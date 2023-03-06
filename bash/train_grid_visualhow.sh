# The name of this experiment.
MODEL_NAME='baseline'
DATASET_NAME='wikihow'
DATA_PATH='/data/'${DATASET_NAME}
WEIGHT_PATH='/data/weights'

# Save logs and models under snap/; make backup.
output=runs/${DATASET_NAME}_${MODEL_NAME}
mkdir -p $output/src
mkdir -p $output/bash
rsync -av  src/* $output/src/
cp $0 $output/bash/run.bash

CUDA_VISIBLE_DEVICES=0,1 python src/train.py \
  --data_path ${DATA_PATH} --data_name ${DATASET_NAME}  \
  --logger_name runs/${DATASET_NAME}_${MODEL_NAME}/log --model_name runs/${DATASET_NAME}_${MODEL_NAME} \
  --eval_name runs/${DATASET_NAME}_${MODEL_NAME}/eval \
  --num_epochs=25  --batch_size=16 --learning_rate=2e-4 --precomp_enc_type backbone  --workers 8 --backbone_source wsl \
  --backbone_warmup_epochs 25 --embedding_warmup_epochs 1 --optim adam --backbone_lr_factor 0.01  --log_step 200 \
  --input_scale_factor 1.0  --backbone_path ${WEIGHT_PATH}/original_updown_backbone.pth  --resume runs/${DATASET_NAME}_${MODEL_NAME}/checkpoint.pth \
  --progressive_optimization --decay_factor 1.0 --graph_layer_num 3 --loss_type ohem --two_stage_warmup_epochs 5 --two_stage --no_norm --extend_graph\
  --interleave_validation --stepwise_task_completion --stepwise_task_completion_weight 1.0 --attention_supervision --attention_supervision_weight 1.0
