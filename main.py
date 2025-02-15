import os
import time
import csv
import numpy as np

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
from tqdm import tqdm

cudnn.benchmark = True

import matplotlib.pyplot as plt

from metrics import AverageMeter, Result
import utils
import criteria
import models
from nyu import NYUDataset 

args = utils.parse_command()
print(args)

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu # Set the GPU.

fieldnames = ['rmse', 'mae', 'delta1', 'absrel',
            'lg10', 'mse', 'delta2', 'delta3', 'data_time', 'gpu_time']
best_fieldnames = ['best_epoch'] + fieldnames
best_result = Result()
best_result.set_to_worst()

##################################################################
def create_data_loaders(args):
    # Data loading code
    print("=> creating data loaders ...")
    home_path = "/home/jinoo5953/FastDepth"
    traindir = os.path.join(home_path, 'data', args.data, 'train')
    valdir = os.path.join(home_path, 'data', args.data, 'val')
    train_loader = None

    max_depth = args.max_depth if args.max_depth >= 0.0 else np.inf

    if args.data == 'nyudepthv2':
        if not args.evaluate:
            train_dataset = NYUDataset(traindir, split='train', modality=args.modality)
        val_dataset = NYUDataset(valdir, split='val', modality=args.modality)
    else:
        raise RuntimeError('Dataset not found.' + 'The dataset must be either of nyudepthv2 or kitti.')

    # set batch size to be 1 for validation
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=args.workers,
                                             pin_memory=True)

    # put construction of train loader here, for those who are interested in testing only
    if not args.evaluate:
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True,
            num_workers=args.workers, pin_memory=True, sampler=None,
            worker_init_fn=lambda work_id: np.random.seed(work_id))
        # worker_init_fn ensures different sampling patterns for each data loading thread

    print("=> data loaders created.")
    return train_loader, val_loader


####################################################################

def main():
    global args, best_result, output_directory, train_csv, test_csv

    # evaluation mode
    if args.evaluate:

        # Data loading code
        
        print("=> creating data loaders...")
        valdir = os.path.join("/home/jinoo5953/FastDepth", 'data', args.data, 'val')

        if args.data == 'nyudepthv2':
            val_dataset = NYUDataset(valdir, split='val', modality=args.modality)
        else:
            raise RuntimeError('Dataset not found.')

        # set batch size to be 1 for validation
        val_loader = torch.utils.data.DataLoader(val_dataset,
                                                 batch_size=1, shuffle=False, num_workers=args.workers, pin_memory=True)
        print("=> data loaders created.")

        assert os.path.isfile(args.evaluate), \
            "=> no model found at '{}'".format(args.evaluate)
        print("=> loading model '{}'".format(args.evaluate))
        checkpoint = torch.load(args.evaluate)
        if type(checkpoint) is dict:
            args.start_epoch = checkpoint['epoch']
            best_result = checkpoint['best_result']
            model = checkpoint['model']
            print("=> loaded best model (epoch {})".format(checkpoint['epoch']))
        else:
            model = checkpoint
            args.start_epoch = 0
        output_directory = os.path.dirname("/home/jinoo5953/FastDepth/results")
        validate(val_loader, model, args.start_epoch, write_to_file=False)
        return

    start_epoch = 0
    if args.train:
        train_loader, val_loader = create_data_loaders(args)
        print("=> creating Model ({}-{}) ...".format(args.arch, args.decoder))

        model = models.MobileNetSkipAdd(output_size=train_loader.dataset.output_size)
        print("=> model created.")
        optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
        
        
    
        # model = torch.nn.DataParallel(model).cuda() # for multi-gpu training
        model = model.cuda()

        # define loss function (criterion) and optimizer
    if args.criterion == 'l2':
        criterion = criteria.MaskedMSELoss().cuda()
    elif args.criterion == 'l1':
        criterion = criteria.MaskedL1Loss().cuda()

        # create results folder, if not already exists
    output_directory = utils.get_output_directory(args)
    
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    train_csv = os.path.join(output_directory, 'train.csv')
    test_csv = os.path.join(output_directory, 'test.csv')
    best_txt = os.path.join(output_directory, 'best.txt')

    # create new csv files with only header
    if not args.resume:
        with open(train_csv, 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
        with open(test_csv, 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

        for epoch in tqdm(range(start_epoch, args.epochs)):
            utils.adjust_learning_rate(optimizer, epoch, args.lr)
            train(train_loader, model, criterion, optimizer, epoch)  # train for one epoch
            result, img_merge = validate(val_loader, model, epoch)  # evaluate on validation set

            # remember best rmse and save checkpoint
            is_best = result.delta1 > best_result.delta1
            if is_best:
                best_result = result
                with open(best_txt, 'w') as txtfile:
                    txtfile.write(
                        "epoch={}\nmse={:.3f}\nrmse={:.3f}\nabsrel={:.3f}\nlg10={:.3f}\nmae={:.3f}\ndelta1={:.3f}\nt_gpu={:.4f}\n".
                            format(epoch, result.mse, result.rmse, result.absrel, result.lg10, result.mae,
                                   result.delta1,
                                   result.gpu_time))
                if img_merge is not None:
                    img_filename = output_directory + '/comparison_best.png'
                    utils.save_image(img_merge, img_filename)

            utils.save_checkpoint({
                'epoch' : epoch,
                'model_state_dict' : model.state_dict(),
                'optimizer_state_dict' : optimizer.state_dict(),
                'args': args,
                'arch': args.arch,
                'best_result': best_result,
            }, is_best, epoch, output_directory)
            
    elif args.resume:
        print("=> saved checkpoint allow.")
        
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        args = checkpoint['args']
        args.arch = checkpoint['arch']
        best_result = checkpoint['best_result']    
        
        with open(train_csv, 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
        with open(test_csv, 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

        for epoch in tqdm(range(epoch, args.epochs)):
            utils.adjust_learning_rate(optimizer, epoch, args.lr)
            train(train_loader, model, criterion, optimizer, epoch)  # train for one epoch
            result, img_merge = validate(val_loader, model, epoch)  # evaluate on validation set

            # remember best rmse and save checkpoint
            is_best = result.delta1 > best_result.delta1
            if is_best:
                best_result = result
                with open(best_txt, 'w') as txtfile:
                    txtfile.write(
                        "epoch={}\nmse={:.3f}\nrmse={:.3f}\nabsrel={:.3f}\nlg10={:.3f}\nmae={:.3f}\ndelta1={:.3f}\nt_gpu={:.4f}\n".
                            format(epoch, result.mse, result.rmse, result.absrel, result.lg10, result.mae,
                                   result.delta1,
                                   result.gpu_time))
                if img_merge is not None:
                    img_filename = output_directory + '/comparison_best.png'
                    utils.save_image(img_merge, img_filename)

            utils.save_checkpoint({
                'epoch' : epoch,
                'model_state_dict' : model.state_dict(),
                'optimizer_state_dict' : optimizer.state_dict(),
                'args': args,
                'arch': args.arch,
                'best_result': best_result,
            }, is_best, epoch, output_directory)            
    torch.save(model, "/home/jinoo5953/FastDepth/train_model/trained_FastDepth.pth.tar")
    
def train(train_loader, model, criterion, optimizer, epoch):
    average_meter = AverageMeter()
    model.train()
    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        input, target = input.cuda(), target.cuda()
        torch.cuda.synchronize()
        data_time = time.time() - end
        
        end = time.time()
        pred = model(input)
        loss = criterion(pred, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        torch.cuda.synchronize()
        gpu_time = time.time() - end
        
        result = Result()
        result.evaluate(pred.data, target.data)
        
        average_meter.update(result, gpu_time, data_time)


def validate(val_loader, model, epoch, write_to_file=True):
    average_meter = AverageMeter()
    model.eval() # switch to evaluate mode
    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        input, target = input.cuda(), target.cuda()
        # torch.cuda.synchronize()
        data_time = time.time() - end

        # compute output
        end = time.time()
        with torch.no_grad():
            pred = model(input)
        # torch.cuda.synchronize()
        gpu_time = time.time() - end

        # measure accuracy and record loss
        result = Result()
        result.evaluate(pred.data, target.data)
        average_meter.update(result, gpu_time, data_time, input.size(0))
        end = time.time()
        
        # save 8 images for visualization
        skip = 50

        if args.modality == 'rgb':
            rgb = input

        if i == 0:
            img_merge = utils.merge_into_row(rgb, target, pred)
        elif (i < 8*skip) and (i % skip == 0):
            row = utils.merge_into_row(rgb, target, pred)
            img_merge = utils.add_row(img_merge, row)
        elif i == 8*skip:
            filename = output_directory + '/train_model/comparison' + '.png'
            utils.save_image(img_merge, filename)
            print("=> comparison result saved!")

        if (i+1) % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  't_GPU={gpu_time:.3f}({average.gpu_time:.3f})\n\t'
                  'RMSE={result.rmse:.2f}({average.rmse:.2f}) '
                  'MAE={result.mae:.2f}({average.mae:.2f}) '
                  'Delta1={result.delta1:.3f}({average.delta1:.3f}) '
                  'REL={result.absrel:.3f}({average.absrel:.3f}) '
                  'Lg10={result.lg10:.3f}({average.lg10:.3f}) '.format(
                   i+1, len(val_loader), gpu_time=gpu_time, result=result, average=average_meter.average()))

    avg = average_meter.average()

    print('\n*\n'
        'RMSE={average.rmse:.3f}\n'
        'MAE={average.mae:.3f}\n'
        'Delta1={average.delta1:.3f}\n'
        'REL={average.absrel:.3f}\n'
        'Lg10={average.lg10:.3f}\n'
        't_GPU={time:.3f}\n'.format(
        average=avg, time=avg.gpu_time))

    if write_to_file:
        with open(test_csv, 'a') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writerow({'mse': avg.mse, 'rmse': avg.rmse, 'absrel': avg.absrel, 'lg10': avg.lg10,
                'mae': avg.mae, 'delta1': avg.delta1, 'delta2': avg.delta2, 'delta3': avg.delta3,
                'data_time': avg.data_time, 'gpu_time': avg.gpu_time})
    return avg, img_merge


if __name__ == '__main__':
    main()