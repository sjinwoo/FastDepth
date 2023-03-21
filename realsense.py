import pyrealsense2 as rs

import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import time

import torch
import torch.optim
import transforms

cmap = plt.cm.viridis

# global variable
global dataset_count, frame_count
dataset_count, frame_count = 0, 0

# function
def colored_depthimage(depth):
    depth_relative = (depth - np.min(depth)) / (np.max(depth) - np.min(depth))
    return cmap(depth_relative)[:,:,:3]

def colored_depthmap(depth):
    depth_relative = (depth - np.min(depth)) / (np.max(depth) - np.min(depth))
    return cmap(depth_relative)[:,:,:3] / np.min(depth) # H, W, C

def transform(rgb, depth):
    transform = transforms.Compose([transforms.Resize(250.0 / 480), 
                                    transforms.CenterCrop((228, 304)), 
                                    transforms.Resize((224, 224)), ])
    rgb_np = transform(rgb)
    rgb_np = np.asfarray(rgb_np, dtype='float') / 255
    depth_np = cv2.resize(depth, dsize=(224, 224), interpolation=cv2.INTER_LINEAR)
    return rgb_np, depth_np

def segmentation(rgb_np, depth_np):
    rgb = rgb_np.copy()
    thres = np.min(depth_np) * 1.25
    rgb[:, :, 1] = np.where(depth_np < thres, 0, rgb[:, :, 1])
    return rgb

def cal_delta1(output, target):
    maxRatio = torch.max(output / target, target / output)
    delta1 = float((maxRatio < 1.25).float().mean())
    if 0.1 < delta1 < 0.6:
        return 1
    return 0

def make_dataset(rgb, depth, path):
    global dataset_count

    rgb_filename = os.path.join(path, "image", "image{0:05d}".format(dataset_count) + ".png")
    depth_filename = os.path.join(path, "depth", "depth{0:05d}".format(dataset_count) + ".png")

    cv2.imwrite(rgb_filename, rgb)
    cv2.imwrite(depth_filename, depth)

    print(f"==> save {dataset_count} data")
    dataset_count += 1

    return

# load pre-trained model
model = torch.load("C:\\Users\\son\\Desktop\\FastDepth\\model\\trained_model.pth.tar", 
                    map_location=torch.device('cpu'))["model"]
model.eval()

# realsense pipeline & configure
pipeline = rs.pipeline()
config = rs.config()

# get color, detph
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# pipeline start
profile = pipeline.start(config)

# get depth scale [meter]
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()

# color, depth align_to
align_to = rs.stream.color
align = rs.align(align_to)

# make local-time directory
print("=> create local-time directory")
base_dir = os.path.join("C:\\Users\\son\\Desktop\\FastDepth\\our_dataset",
                         time.strftime('%Yy_%mm_%dd_%Hh_%Mm_%Ss', time.localtime(time.time())))
os.mkdir(dir)

# make image & depth directory in local-time directory
rgb_image_dir = os.path.join(base_dir, "image")
os.mkdir(rgb_image_dir)
truth_depth_dir = os.path.join(base_dir, "depth")
os.mkdir(truth_depth_dir)

try:
    while True:
        # get aligned color & depth image
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()
        if not depth_frame or not color_frame:
            continue
        frame_count += 1

        # convert to ndarray
        color_image_np = np.asanyarray(color_frame.get_data())
        depth_image_np = np.asanyarray(depth_frame.get_data()) * depth_scale

        # image re-scale
        color_image, depth_image = transform(color_image_np, depth_image_np)

        # convert to Tensor
        to_tensor = transforms.ToTensor()

        # input model : RGB & truth depth
        input = to_tensor(color_image).unsqueeze(0)
        target = to_tensor(depth_image)
        with torch.no_grad():
            output = model(input)
        prediction = np.squeeze(output.data.numpy())
        
        # save dataset
        if cal_delta1(output, target) and (frame_count % 20):
            make_dataset(color_image_np, depth_image_np, base_dir)

        # colorizer depth map & depth image
        colored_depth_image = colored_depthimage(depth_image)
        colored_depth_map_pred = colored_depthmap(prediction)

        # image segmentation
        rgb_image = segmentation(color_image, prediction)
  
        # show image
        images = np.hstack((color_image, rgb_image, colored_depth_map_pred, colored_depth_image))
        cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('RealSense', images)
        cv2.waitKey(1)

finally:
    pipeline.stop()