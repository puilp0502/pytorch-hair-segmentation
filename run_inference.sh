python evaluate.py \
--inference=True \
--dataset figaro --data_dir ./custom_test_dataset/ \
--use_gpu True \
--save_dir ./custom_test_overlay \
--ckpt_dir ./notebooks/ckpt_unet_v2x/unet_v2x_adam_lr_1e-05_epoch_80.pth \
--networks unet \
$@
# --visualize=True
