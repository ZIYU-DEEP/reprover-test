python generator/main.py fit \
    --config generator/confs/cli_lean4_random_joint.yaml

python generator/main.py fit \
    --config generator/confs/cli_lean4_random_goal.yaml

python generator/main.py fit \
    --config generator/confs/cli_lean4_random_goal_driven_tactic.yaml

python generator/main.py fit \
    --config generator/confs/cli_lean4_random_joint_ckpt.yaml

python generator/main.py fit \
    --config generator/confs/cli_lean4_random_goal_ckpt.yaml

python generator/main.py fit \
    --config generator/confs/cli_lean4_random_goal_driven_tactic_ckpt.yaml

python generator/main.py fit \
   --config generator/confs/cli_lean4_random_goal_driven_tactic_ckpt_resume_2gpu.yaml

python generator/main.py fit \
    --config generator/confs/cli_lean4_random_joint_ckpt_8gpu.yaml

python generator/main.py fit \
    --config generator/confs/cli_lean4_random_joint_ckpt_8gpu_resume.yaml
# salloc --gres=gpu:A100:1 --mem=80G --ntasks-per-node=1 --account=gts-czhang355 -q inferno -t 24:00:00 -C A100-80GB

python generator/main.py fit \
    --config generator/confs/cli_lean4_random_goal.yaml

CUDA_VISIBLE_DEVICES=4,5,6,7 python generator/main.py fit \
    --config generator/confs/cli_lean4_random_goal_ckpt_4gpu.yaml

CUDA_VISIBLE_DEVICES=0,1,2,3 python generator/main.py fit \
    --config generator/confs/cli_lean4_random_goal_driven_tactic_ckpt_4gpu.yaml

