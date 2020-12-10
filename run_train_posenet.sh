# script to train initial posenet model
PWD=$(pwd)
mkdir $PWD/checkpoints/
EXPNAME=posenet
# directory to store intermediate models
CHECKPOINT_DIR=$PWD/checkpoints/$EXPNAME
mkdir $CHECKPOINT_DIR
# directory for kitti dataset
DATAROOT_DIR=$PWD/data_kitti
# include python file to run, training data path, path to store intermediate models, batchsize, learning rate,
# port to monitor training progress
CUDA_VISIBLE_DEVICES=0 python src/train_main_posenet.py --dataroot $DATAROOT_DIR\
 --checkpoints_dir $CHECKPOINT_DIR --which_epoch -1 --save_latest_freq 1000\
  --batchSize 1 --display_freq 50 --name $EXPNAME --lambda_S 0.01 --smooth_term 2nd --use_ssim --display_port 8009
