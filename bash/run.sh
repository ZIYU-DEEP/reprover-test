python generator/main.py fit \
    --config generator/confs/cli_lean4_random_modified.yaml

python generator/main_new.py fit \
    --config generator/confs/cli_lean4_random_modified.yaml  

python prover/evaluate.py \
    --data-path data/leandojo_benchmark_4/random/ \
    --ckpt_path leandojo-pl-ckpts/generator_random.ckpt \
    --split test \
    --num-workers 5 \
    --num-gpus 1 \
    --num-theorems 1

CUDA_VISIBLE_DEVICES=0,1 python prover/evaluate.py \
    --data-path data/leandojo_benchmark_4/random/ \
    --ckpt_path leandojo-pl-ckpts/generator_random.ckpt \
    --split test \
    --num-workers 5 \
    --num-gpus 2 \
    --num-theorems 1

python generator/main.py fit \
    --config generator/confs/cli_lean4_random_goal_driven_tactic.yaml

python generator/main.py fit \
    --config generator/confs/cli_lean4_random_goal_driven_tactic.yaml

python generator/main.py fit --config generator/confs/cli_lean4_random_joint.yaml

python generator/main.py fit --config generator/confs/cli_lean4_random_goal.yaml

srun --gres=gpu:A100:1 \
     --mem=80G \
     --ntasks-per-node=1 \
     --account=gts-czhang355 \
     -q inferno \
     -t 1:00:00 \
     -C A100-80GB \
     bash -c "python generator/main.py fit --config generator/confs/cli_lean4_random_goal_driven_tactic.yaml"


CUDA_VISIBLE_DEVICES=0 python prover/evaluate.py \
    --data-path data/leandojo_benchmark_4/random/ \
    --ckpt_path leandojo-pl-ckpts/generator_random.ckpt \
    --split test \
    --num-workers 5 \
    --num-gpus 1 \
    --num-theorems 1

# Exporting the variables
export joint_ckpt_path='/home/ubuntu/dojo/github/reprover-test/lightning_logs/joint_ckpt/reprover-joint/s3upj4dw/checkpoints/checkpoint-epoch=00-step=10000-loss_val=1000.0000-loss_train=0.2334.ckpt'
export goal_ckpt_path='/home/ubuntu/dojo/github/reprover-test/lightning_logs/goal/reprover-goal/lkiucv18/checkpoints/checkpoint-epoch=00-step=300-loss_val=0.0000-loss_train=0.1045-timestamp=0.ckpt'
export tactic_ckpt_path='/home/ubuntu/dojo/github/reprover-test/lightning_logs/goal_driven_tactic_ckpt/reprover-goal_driven_tactic/abmoy18k/checkpoints/checkpoint-epoch=01-step=100000-loss_val=0.2866-loss_train=0.1084.ckpt'
export default_path='/home/ubuntu/dojo/github/reprover-test/leandojo-pl-ckpts/generator_random.ckpt'

