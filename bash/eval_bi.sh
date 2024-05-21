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
export goal_driven_tactic_ckpt_path='/home/ubuntu/dojo/github/reprover-test/lightning_logs/goal_driven_tactic_ckpt/reprover-goal_driven_tactic/abmoy18k/checkpoints/checkpoint-epoch=01-step=100000-loss_val=0.2866-loss_train=0.1084.ckpt'
export goal_driven_tactic_ckpt='/home/ubuntu/dojo/github/reprover-test/lightning_logs/goal_driven_tactic_ckpt/reprover-goal_driven_tactic/gdz01a3r/checkpoints/checkpoint-epoch=02-step=35000-loss_val=0.2795-loss_train=0.1887.ckpt'

export default_path='/home/ubuntu/dojo/github/reprover-test/leandojo-pl-ckpts/generator_random.ckpt'
export joint_ckpt_path='/home/ubuntu/dojo/reprover-joint-ckpt/checkpoint-epoch=01-step=7200-loss_val=0.1362-loss_train=0.1102.ckpt'

export joint_ckpt_path='/home/ubuntu/dojo/reprover-joint-ckpt/checkpoint-epoch=02-step=15000-loss_val=0.1140-loss_train=0.0731.ckpt'
CUDA_VISIBLE_DEVICES=4 python prover/evaluate.py \
    --data-path data/leandojo_benchmark_4/random/ \
    --split test \
    --num-workers 8 \
    --num-gpus 1 \
    --num-theorems 20 \
    --ckpt_path $joint_ckpt_path \
    --gen-type joint \
    --start-ind 0 \
    --timeout 1200 \

CUDA_VISIBLE_DEVICES=5,6 python prover/evaluate.py \
    --data-path data/leandojo_benchmark_4/random/ \
    --split test \
    --num-workers 8 \
    --num-gpus 2 \
    --num-theorems 20 \
    --ckpt_path $default_path \
    --gen-type default \
    --start-ind 0

# Set the gen_type, goal_ckpt_path, and num_goals
# The timeout should be set accordingly
export goal_ckpt_path='/home/ubuntu/dojo/github/reprover-test/lightning_logs/goal_ckpt/reprover-goal/migvv75v/checkpoints/checkpoint-epoch=05-step=41993-loss_val=0.0538-loss_train=0.0217.ckpt'
export goal_driven_tactic_ckpt_path="/home/ubuntu/dojo/github/reprover-test/lightning_logs/goal_driven_tactic_ckpt/reprover-goal_driven_tactic/abmoy18k/checkpoints/checkpoint-epoch=01-step=100000-loss_val=0.2866-loss_train=0.1084.ckpt"
export num_sampled_goal='5'

CUDA_VISIBLE_DEVICES=7 python prover_rir/evaluate.py \
    --data-path data/leandojo_benchmark_4/random/ \
    --split test \
    --num-workers 4 \
    --num-gpus 1 \
    --num-theorems 20 \
    --ckpt_path $goal_driven_tactic_ckpt_path \
    --goal-ckpt-path $goal_ckpt_path \
    --num-sampled-goals 5 \
    --gen-type goal_driven_tactic \
    --timeout 3000 \
    --start-ind 0

CUDA_VISIBLE_DEVICES=5,6 python prover_rir/evaluate.py \
    --data-path data/leandojo_benchmark_4/random/ \
    --split test \
    --num-workers 8 \
    --num-gpus 2 \
    --num-theorems 20 \
    --ckpt_path $goal_driven_tactic_ckpt_path \
    --goal-ckpt-path $goal_ckpt_path \
    --num-sampled-goals 1 \
    --gen-type goal_driven_tactic \
    --timeout 1200 \
    --start-ind 0

CUDA_VISIBLE_DEVICES=7 python prover/evaluate.py \
    --data-path data/leandojo_benchmark_4/random/ \
    --split test \
    --num-workers 5 \
    --num-gpus 1 \
    --num-theorems 1 \
    --ckpt_path $joint_ckpt_path \
    --gen-type joint \
    --start-ind 10


CUDA_VISIBLE_DEVICES=0 python prover/evaluate.py \
    --data-path data/leandojo_benchmark_4/random/ \
    --split test \
    --num-workers 5 \
    --num-gpus 1 \
    --num-theorems 1 \
    --ckpt_path $joint_ckpt_path \
    --gen-type joint \
    --start-ind 10

CUDA_VISIBLE_DEVICES=0 python prover/evaluate.py \
    --data-path data/leandojo_benchmark_4/random/ \
    --split test \
    --num-workers 5 \
    --num-gpus 1 \
    --num-theorems 1 \
    --ckpt_path $tactic_ckpt_path \
    --gen-type goal_driven_tactic \
    --start-ind 0

CUDA_VISIBLE_DEVICES=1 python prover/evaluate.py \
    --data-path data/leandojo_benchmark_4/random/ \
    --split test \
    --num-workers 5 \
    --num-gpus 1 \
    --num-theorems 5 \
    --ckpt_path $default_path \
    --gen-type default \
    --start-ind 0

CUDA_VISIBLE_DEVICES=2 python prover/evaluate.py \
    --data-path data/leandojo_benchmark_4/random/ \
    --split test \
    --num-workers 5 \
    --num-gpus 1 \
    --num-theorems 5 \
    --ckpt_path $joint_ckpt_path \
    --gen-type joint \
    --start-ind 0

## RUNNING
CUDA_VISIBLE_DEVICES=0,1 python prover/evaluate.py \
    --data-path data/leandojo_benchmark_4/random/ \
    --split test \
    --num-workers 5 \
    --num-gpus 2 \
    --num-theorems 20 \
    --ckpt_path $tactic_ckpt_path \
    --gen-type goal_driven_tactic \
    --start-ind 5

CUDA_VISIBLE_DEVICES=2,3 python prover/evaluate.py \
    --data-path data/leandojo_benchmark_4/random/ \
    --split test \
    --num-workers 5 \
    --num-gpus 2 \
    --num-theorems 20 \
    --ckpt_path $default_path \
    --gen-type default \
    --start-ind 5

CUDA_VISIBLE_DEVICES=0,1 python prover/evaluate.py \
    --data-path data/leandojo_benchmark_4/random/ \
    --split test \
    --num-workers 5 \
    --num-gpus 2 \
    --num-theorems 20 \
    --ckpt_path $joint_ckpt_path \
    --gen-type joint \
    --start-ind 5 \
    --timeout 1200

export default_path='/home/ubuntu/dojo/github/reprover-test/leandojo-pl-ckpts/generator_random.ckpt'
CUDA_VISIBLE_DEVICES=4 python prover/evaluate.py \
    --data-path data/leandojo_benchmark_4/random/ \
    --split test \
    --num-workers 6 \
    --num-gpus 1 \
    --num-theorems 150 \
    --ckpt_path $default_path \
    --gen-type default \
    --start-ind 500

export default_path='/home/ubuntu/dojo/github/reprover-test/leandojo-pl-ckpts/generator_random.ckpt'
CUDA_VISIBLE_DEVICES=5 python prover/evaluate.py \
    --data-path data/leandojo_benchmark_4/random/ \
    --split test \
    --num-workers 8 \
    --num-gpus 1 \
    --num-theorems 150 \
    --ckpt_path $default_path \
    --gen-type default \
    --start-ind 650

export default_path='/home/ubuntu/dojo/github/reprover-test/leandojo-pl-ckpts/generator_random.ckpt'
CUDA_VISIBLE_DEVICES=6 python prover/evaluate.py \
    --data-path data/leandojo_benchmark_4/random/ \
    --split test \
    --num-workers 8 \
    --num-gpus 1 \
    --num-theorems 150 \
    --ckpt_path $default_path \
    --gen-type default \
    --start-ind 800

export default_path='/home/ubuntu/dojo/github/reprover-test/leandojo-pl-ckpts/generator_random.ckpt'
CUDA_VISIBLE_DEVICES=7 python prover/evaluate.py \
    --data-path data/leandojo_benchmark_4/random/ \
    --split test \
    --num-workers 4 \
    --num-gpus 1 \
    --num-theorems 150 \
    --ckpt_path $default_path \
    --gen-type default \
    --start-ind 900