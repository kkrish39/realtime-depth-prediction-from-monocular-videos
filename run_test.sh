# directory for kitti dataset
DATAROOT=../kitti/
PWD=$(pwd)
# cloose model of desire
CKPT=$PWD/checkpoints/finetune/5_model.pth
OUTPUT=$PWD/posenet+ddvo
# generate output for kitti test dataset
CUDA_VISIBLE_DEVICES=0 nice -10 python src/testKITTI.py --dataset_root $DATAROOT --ckpt_file $CKPT --output_path $OUTPUT --test_file_list test_files_eigen.txt
# evaluate using SMFLearner
python2 ~/SfMLearner/kitti_eval/eval_depth.py --kitti_dir=$DATAROOT --pred_file=$OUTPUT.npy --test_file_list test_files_eigen.txt