CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --use_env train_dist.py --dataroot your_data_path --version v1.0-trainval or v1.0-mini


