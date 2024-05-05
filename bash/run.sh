CUDA_VISIBLE_DEVICES=3 python generator/main.py fit --config generator/confs/cli_lean4_random_modified.yaml  
python generator/main.py fit --config generator/confs/cli_lean4_random_modified.yaml  

python prover/evaluate.py \
    --data-path data/leandojo_benchmark_4/random/ \
    --ckpt_path leandojo-pl-ckpts/generator_random.ckpt \
    --split test \
    --num-workers 5 \
    --num-gpus 1 \
    --num-theorems 1