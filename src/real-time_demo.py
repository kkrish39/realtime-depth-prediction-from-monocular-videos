from networks import VggDepthEstimator
from LKVOLearner import FlipLR
import torch
from torch.autograd import Variable
import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import requests
import cv2
import argparse

# get the commandline arguments to obtain model and ip cam address
parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, help='Path of model to run')
parser.add_argument("--address", type=str, help="ip_address and port number")
args = parser.parse_args()

model_path = args.model_path

# image dimensions
img_size = [128, 416]
vgg_depth_net = VggDepthEstimator(img_size)
vgg_depth_net.load_state_dict(torch.load(model_path))

fliplr = FlipLR(imW=img_size[1], dim_w=2)

# function to process the image data and return depth
def processImage(imgData):

    img = imgData
    img = img.transpose(2, 0, 1)
    img_var = Variable(torch.from_numpy(img).float(), volatile=True)
    pred_depth_pyramid = vgg_depth_net.forward((img_var.unsqueeze(0)-127)/127)
    depth = pred_depth_pyramid[0].data.cpu().squeeze().numpy()
    return depth

# complete URL of the relayed images
URL = "http://" + args.address + "/shot.jpg"

# windows to display image and depth
cv2.namedWindow('image', cv2.WINDOW_NORMAL)
cv2.resizeWindow('image', 128,600)
cv2.namedWindow('depth', cv2.WINDOW_NORMAL)
cv2.resizeWindow('depth', 600,600)

# loop till interrupted
while True:
    # get data from URL
    img_resp = requests.get(URL)
    # obtain the image/content
    img_arr = np.array(bytearray(img_resp.content), dtype=np.uint8)
    img = cv2.imdecode(img_arr, -1)

    # resize it for our required dimensions
    edges = cv2.resize(img,(416,128), interpolation = cv2.INTER_AREA)

    # obtain the depth
    depth = processImage(np.array(edges))

    # display the image and depth
    cv2.imshow('image', np.array(edges))
    cv2.imshow('depth', depth)

    # break condition
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
