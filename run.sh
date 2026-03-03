DATASET_NAME="RSTPReid"

CUDA_VISIBLE_DEVICES=0 \
python train.py \
--name LPNC \
--output_dir 'LPNC_log' \
--dataset_name $DATASET_NAME \
--num_epoch 60
