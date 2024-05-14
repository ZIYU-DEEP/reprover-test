# README

## Setups
```bash
# Set up the environment
conda create -n dojo python=3.10
conda activate dojo
pip install -r requirements_lambda.txt

# Set up elan
curl https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh -sSf | sh
source $HOME/.elan/env

# You may add the following to your .bashrc
export PATH="$HOME/.elan/bin:$PATH"
export GITHUB_ACCESS_TOKEN='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'
export PYTHONPATH="$PYTHONPATH:/home/ubuntu/dojo/github/reprover-test"

# Download data and trace the leandojo repo
python scripts/download_data
python scripts/trace_repos.py


# (Optional) Download the checkpoints
sudo apt-get update
sudo apt-get install git-lfs
git lfs install
git clone https://huggingface.co/kaiyuy/leandojo-pl-ckpts.git
cd leandojo-pl-ckpts
git lfs fetch --all
# On some servers, the fetch may fail on latest and zero_to_fp32.py
# And you have to manually modify them according to the repo
```

## Training Settings
```bash
```