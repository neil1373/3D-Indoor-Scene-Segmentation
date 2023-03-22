bash scripts/tta.sh 1 30 $1 $4 a_0 $5 Res16UNet34A True False
bash scripts/tta.sh 31 40 $1 $4 a_0 $5 Res16UNet34A False True

bash scripts/tta.sh 1 30 $2 $4 c_3daug $5 Res16UNet34C True False
bash scripts/tta.sh 31 40 $2 $4 c_3daug $5 Res16UNet34C False True

bash scripts/tta.sh 1 30 $3 $4 c_inst $5 Res16UNet34C True False
bash scripts/tta.sh 31 40 $3 $4 c_inst $5 Res16UNet34C False True