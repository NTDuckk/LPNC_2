DATASET_NAME="RSTPReid"

CUDA_VISIBLE_DEVICES=0,1 \
torchrun --nproc_per_node=2 train.py \
--name LPNC \
--output_dir 'LPNC_log' \
--dataset_name $DATASET_NAME \
--loss_names 'supid+cotrl+cid' \
--num_epoch 60 \
--gradient_checkpointing \
--batch_size 128 \
--gradient_accumulation_steps 4
