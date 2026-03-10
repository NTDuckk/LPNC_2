DATASET_NAME="Market1501"
ROOT_DIR="data"  # expects data/Market-1501-v15.09.15

# CUDA_VISIBLE_DEVICES=0 \
# python train.py \
# --name LPNC \
# --output_dir 'LPNC_log' \
# --dataset_name $DATASET_NAME \
# --loss_names 'supid+cotrl+cid' \
# --batch_size 64 \
# --accumulation_steps 8 \
# --num_epoch 60

CUDA_VISIBLE_DEVICES=0 \
python train.py \
--name LPNC \
--output_dir 'LPNC_log' \
--dataset_name $DATASET_NAME \
--root_dir $ROOT_DIR \
--loss_names 'supid+cotrl+cid' \
--batch_size 64 \
--accumulation_steps 8 \
--num_epoch 60
