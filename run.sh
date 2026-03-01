DATASET_NAME="RSTPReid"

CUDA_VISIBLE_DEVICES=0 \
python train.py \
--name LPNC \
--output_dir 'LPNC_log' \
--dataset_name $DATASET_NAME \
--loss_names 'supid+cotrl+cid' \
--num_epoch 60 \
--fp16 \
--gradient_checkpointing \
--batch_size 64 \
--gradient_accumulation_steps 8
