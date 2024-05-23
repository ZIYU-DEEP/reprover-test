# screen -S 6_240_260
export goal_driven_tactic_ckpt_path="/home/ubuntu/dojo/github/reprover-test/lightning_logs/goal_driven_tactic_ckpt/reprover-goal_driven_tactic/zjnvjqda/checkpoints/checkpoint-epoch=05-step=40000-loss_val=1000.0000-loss_train=0.1496.ckpt"

export goal_ckpt_path='/home/ubuntu/dojo/github/reprover-test/lightning_logs/goal_ckpt/reprover-goal/migvv75v/checkpoints/checkpoint-epoch=05-step=41993-loss_val=0.0538-loss_train=0.0217.ckpt'

CUDA_VISIBLE_DEVICES=3 python prover_rir/evaluate.py \
    --data-path data/minif2f/neurips/ \
    --split test \
    --num-workers 4 \
    --num-gpus 1 \
    --num-theorems 20 \
    --ckpt_path $goal_driven_tactic_ckpt_path \
    --goal-ckpt-path $goal_ckpt_path \
    --num-sampled-goals 2 \
    --gen-type goal_driven_tactic \
    --timeout 1200 \
    --start-ind 1 \
    --verbose