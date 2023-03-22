for i in $(seq $1 $2);
do
    python3 spatio_temporal/main.py --weights $3 --save_pred_dir $4/$5_$i --is_train False --test_original_pointcloud True --dataset Scannet200Voxelization2cmDataset --scannet_path $6 \
        --model $7 --data_aug_3d $8 --test_elastic_distortion $9
done