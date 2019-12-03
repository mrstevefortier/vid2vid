python serve.py --name pose2body_512p_g1 \
--dataroot datasets/pose --dataset_mode pose \
--openpose_only --input_nc 3 --n_scales_spatial 2 --ngf 64 \
--loadSize 512 --no_first_img
