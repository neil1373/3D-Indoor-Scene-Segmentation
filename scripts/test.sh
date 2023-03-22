# generate test time augment predictions
bash scripts/tta_all.sh A_only_22000.pth C_inst_48000.pth C_aug3d_59500.pth augs $1

# ensemble all test time predictions
bash scripts/ensemble.sh augs $2