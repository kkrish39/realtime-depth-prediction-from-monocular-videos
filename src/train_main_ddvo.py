import torch
from torch import optim
import matplotlib.pyplot as plt
import torchvision
import numpy as np
from torch.autograd import Variable
from torch.utils.data import DataLoader
import scipy.io as sio
import os

from LKVOLearner import LKVOLearner
from KITTIdataset import KITTIdataset

from collections import OrderedDict
from options.train_options import TrainOptions
from util.visualizer import Visualizer

from timeit import default_timer as timer

# get the commandline arguments
opt = TrainOptions().parse()

# initialize image dimensions
img_size = [opt.imH, opt.imW]

visualizer = Visualizer(opt)

#set dataset root
dataset = KITTIdataset(data_root_path=opt.dataroot, img_size=img_size, bundle_size=3)

# load the dataset
dataloader = DataLoader(dataset, batch_size=opt.batchSize,
                        shuffle=True, num_workers=opt.nThreads, pin_memory=True)
# list of gpu ids
gpu_ids = list(range(opt.batchSize))

# LKVOLearner trigger
lkvolearner = LKVOLearner(img_size=img_size, ref_frame_idx=1, lambda_S=opt.lambda_S, gpu_ids = gpu_ids, smooth_term = opt.smooth_term, use_ssim=opt.use_ssim)
lkvolearner.init_weights()

if opt.which_epoch >= 0:
    print("load pretrained model")
    lkvolearner.load_model(os.path.join(opt.checkpoints_dir, '%s_model.pth' % (opt.which_epoch)))

lkvolearner.cuda()

ref_frame_idx = 1

# get depth map
def vis_depthmap(input):
    x = (input-input.min()) * (255/(input.max()-input.min()+.00001))
    return x.unsqueeze(2).repeat(1, 1, 3)





optimizer = optim.Adam(lkvolearner.get_parameters(), lr=.0001)

step_num = 0



for epoch in range(max(0, opt.which_epoch), opt.epoch_num+1):
    t = timer()
    for ii, data in enumerate(dataloader):
        optimizer.zero_grad()
        frames = Variable(data[0].float().cuda())
        camparams = Variable(data[1])
        cost, photometric_cost, smoothness_cost, frames, inv_depths = \
            lkvolearner.forward(frames, camparams, max_lk_iter_num=opt.max_lk_iter_num)

        cost_ = cost.data.cpu()
        inv_depths_mean = inv_depths.mean().data.cpu().numpy()

        cost.backward()
        optimizer.step()

        step_num+=1

        if np.mod(step_num, opt.print_freq)==0:
            elapsed_time = timer()-t
            print('%s: %s / %s, ... elapsed time: %f (s)' % (epoch, step_num, int(len(dataset)/opt.batchSize), elapsed_time))
            print(inv_depths_mean)
            t = timer()
            visualizer.plot_current_errors(step_num, 1, opt,
                        OrderedDict([('photometric_cost', photometric_cost.data.cpu()[0]),
                         ('smoothness_cost', smoothness_cost.data.cpu()[0]),
                         ('cost', cost.data.cpu()[0])]))

        if np.mod(step_num, opt.display_freq)==0:

            frame_vis = frames.data.permute(1,2,0).contiguous().cpu().numpy().astype(np.uint8)
            depth_vis = vis_depthmap(inv_depths.data.cpu()).numpy().astype(np.uint8)
            visualizer.display_current_results(
                            OrderedDict([('%s frame' % (opt.name), frame_vis),
                                    ('%s inv_depth' % (opt.name), depth_vis)]),
                                    epoch)
            sio.savemat(os.path.join(opt.checkpoints_dir, 'depth_%s.mat' % (step_num)),
                {'D': inv_depths.data.cpu().numpy(),
                 'I': frame_vis})

        if np.mod(step_num, opt.save_latest_freq)==0:
            print("cache model....")
            lkvolearner.save_model(os.path.join(opt.checkpoints_dir, '%s_model.pth' % (epoch)))
            lkvolearner.cuda()
            print('..... saved')
