# python h-seg_trainer.py \
#     --model-type standard \
#     --backbone resnet18 \
#     --pretrained \
#     --batch-size 48 \
#     --num-epochs 200 \
#     --lr 5e-4 \
#     --weight-decay 1e-5 \
#     --no-wandb
    # --use-optuna \
    # --n-trials 10 \
    # --prune-threshold 0.8 \
    # --prune-patience 10 \
    # --wandb-name euclidian-optuna  # --no-wandb

# python h-seg_trainer.py \
#     --model-type hyperbolic \
#     --backbone resnet18 \
#     --batch-size 32 \
#     --num-epochs 200 \
#     --lr 5e-4 \
#     --weight-decay 1e-5 \
#     --use-optuna \
#     --n-trials 10 \
#     --prune-threshold 0.8 \
#     --prune-patience 10 \
#     --wandb-name hyperbolic-optuna  # --no-wandb