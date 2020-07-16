DETECTRON2_DATASETS='/home/jaeminc/workspace/datasets'
export DETECTRON2_DATASETS=${DETECTRON2_DATASETS}

python train_net.py --num-gpus 4 --config-file configs/X-152-region-c4.yaml


# python train_net.py --num-gpus 4 --config-file configs/X-152-region-c4.yaml \
#     SOLVER.IMS_PER_BATCH 24
