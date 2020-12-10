from networks import VggDepthEstimator
from LKVOLearner import FlipLR
import torch
from torch.autograd import Variable
import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import argparse

"""
CUDA_VISIBLE_DEVICES=1 nice -10 python3 testKITTI.py --dataset_root /newfoundland/chaoyang/kitti --ckpt_file /home/chaoyang/LKVOLearner/checkpoints/checkpoints_19_416_scratch/9_model.pth --test_file_list
"""

# define the input arguements and their default values
parser = argparse.ArgumentParser()

parser.add_argument("--dataset_root", type=str, default="/media/ruddha/Games/raw_data_downloader", help="dataset root path")
parser.add_argument("--test_file_list", type=str, default="/media/ruddha/C/LKVOLearner-master/test_files_eigen.txt", help="test file list")
parser.add_argument("--ckpt_file", type=str, default=None, help="checkpoint file")
parser.add_argument("--output_path", type=str, default="pred_depths", help="output path")
parser.add_argument("--use_pp", default=False, action="store_true", help='use post processing')


# parse the arguements given via comman line
FLAGS = parser.parse_args()

dataset_root = FLAGS.dataset_root
model_path = FLAGS.ckpt_file
test_file_list = FLAGS.test_file_list
output_path = FLAGS.output_path

# image region of interest
img_size = [128, 416]
vgg_depth_net = VggDepthEstimator(img_size)

# load the model
vgg_depth_net.load_state_dict(torch.load(model_path))
vgg_depth_net.cuda()

fliplr = FlipLR(imW=img_size[1], dim_w=2).cuda()

# function to read the file containing list of test images
def read_text_lines(file_path):
    f = open(file_path, 'r')
    lines = f.readlines()
    f.close()
    lines = [l.rstrip() for l in lines]
    # return a list of test image filenames
    return lines

# obtain the test image filenames as a list
test_files = read_text_lines(test_file_list)

pred_depths = []
i = 0

# iterate over all test images
for filename in test_files:
    print(i)

    # remove any extra spaces in filenames
    filename = filename.split()[0]

    # join with teh dataset root to obtain complete path
    im_path = os.path.join(dataset_root, filename)

    # resize to required dimension
    img_pil = Image.open(im_path).resize((img_size[1], img_size[0]), Image.ANTIALIAS)
    img = np.array(img_pil)

    # transponse the image
    img = img.transpose(2, 0, 1)

    print(filename)

    img_var = Variable(torch.from_numpy(img).float().cuda(), volatile=True)

    # in case post-processing is set
    if FLAGS.use_pp:
        # flip image
        img_vars = (torch.cat((fliplr(img_var).unsqueeze(0), img_var.unsqueeze(0)), 0)-127)/127
        pred_depth_pyramid = vgg_depth_net.forward(img_vars)
        depth = pred_depth_pyramid[0]
        print(depth.size())
        depth_mean = (fliplr(depth[0:1, :, :]) + depth[1:2, :, :])*.5
        pred_depths.append(depth_mean.data.cpu().squeeze().numpy())
        # compute mean
    else:
        pred_depth_pyramid = vgg_depth_net.forward((img_var.unsqueeze(0)-127)/127)
        pred_depths.append(pred_depth_pyramid[0].data.cpu().squeeze().numpy())
    i = i+1

# make array
pred_depths = np.asarray(pred_depths)
print(pred_depths.shape)

# save output
np.save(output_path, 1/pred_depths)
import scipy.io as sio
sio.savemat(output_path, {'D': pred_depths})
