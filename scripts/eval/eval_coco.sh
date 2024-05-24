#! /bin/bash

ckpt=$1
echo $ckpt

# eval dynrefer on coco
python -m torch.distributed.run --nproc_per_node=8 --master_port=29600 train.py --cfg-path configs/eval/coco/coco.yaml --options run.load_ckpt_path=$ckpt