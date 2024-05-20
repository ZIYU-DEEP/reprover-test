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