python h-seg_trainer.py \
    --debug \
    --model-type standard \
    --backbone resnet18 \
    --batch-size 4 \
    --num-epochs 200 \
    --lr 5e-4 \
    --weight-decay 1e-5 \
    --no-wandb \
    --use-optuna \
# --wandb-project h-seg \
# --wandb-entity jonas-klein \
# --wandb-name h-seg-resnet18 \