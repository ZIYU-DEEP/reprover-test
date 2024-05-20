# README

This repository contains code for evaluating a theorem prover on theorems extracted by LeanDojo. The evaluation script (`evaluate.py`) utilizes best-first search with large language models to search for proofs of the given theorems.

## `evaluate.py`
### Workflow

1. **Retrieve theorems from the specified data source:**
   - The `_get_theorems` function retrieves theorems based on the provided criteria, such as data path, split, file path, full name, name filter, and the number of theorems.
   - It calls `_get_theorems_from_files` to load theorems from JSON files and filters them based on the specified criteria.
   - The retrieved theorems are checked for availability in the cache using `is_available_in_cache`.

2. **Search for proofs using multiple concurrent provers:**
   - The `evaluate` function is the main evaluation function that orchestrates the proof search.
   - It creates a `DistributedProver` object with the specified configuration, including the checkpoint path, indexed corpus path, tactic, module, number of workers, number of GPUs, timeout, and number of sampled tactics.
   - The `search_unordered` method of the `DistributedProver` is called to search for proofs of the theorems using multiple concurrent provers.

3. **Calculate evaluation metrics:**
   - After the proof search is completed, the script calculates the result statistics, including the number of proved theorems, failed theorems, and discarded non-theorems.
   - The pass@1 metric is computed, representing the proportion of proved theorems among the total theorems (excluding discarded ones).

4. **Save evaluation results:**
   - The evaluation results are saved as a pickle file in the `./results` directory with a unique experiment ID.

### Usage

To run the evaluation script, use the following command:

```
python evaluate.py --data-path <path_to_data> [--exp-id <experiment_id>] [--split <split>] [--file-path <file_path>] [--full-name <full_name>] [--name-filter <name_filter>] [--num-theorems <num_theorems>] [--ckpt_path <ckpt_path>] [--indexed-corpus-path <indexed_corpus_path>] [--tactic <tactic>] [--module <module>] [--num-sampled-tactics <num_sampled_tactics>] [--timeout <timeout>] [--num-workers <num_workers>] [--num-gpus <num_gpus>] [--verbose]
```

- `--data-path`: Path to the data extracted by LeanDojo (required).
- `--exp-id`: Experiment ID used for logging (optional).
- `--split`: The data split to use (e.g., 'val', 'test') (default: 'val').
- `--file-path`, `--full-name`, `--name-filter`, `--num-theorems`: Filters for selecting specific theorems (optional).
- `--ckpt_path`: Checkpoint of the tactic generator (optional).
- `--indexed-corpus-path`: Path to a pickled indexed corpus (optional).
- `--tactic`: The tactic to evaluate (optional).
- `--module`: The module to import the tactic (optional).
- `--num-sampled-tactics`: Number of tactics to sample at each node during proof search (default: 64).
- `--timeout`: Maximum number of seconds the proof search can take (default: 600).
- `--num-workers`: The number of concurrent provers (default: 1).
- `--num-gpus`: The number of GPUs for proof search (default: 0).
- `--verbose`: Set the logging level to DEBUG (optional).

### File Structure

- `evaluate.py`: The main evaluation script that orchestrates the proof search and calculates metrics.
- `proof_search.py`: Contains the implementation of the `DistributedProver` and the proof search algorithm.
- `search_tree.py`: Defines the data structures and methods for representing and manipulating the search tree during proof search.


### Acknowledgments

This project is built upon the LeanDojo framework and utilizes large language models for theorem proving. We would like to acknowledge the contributions of the LeanDojo team and the developers of the underlying language models.

For more information about LeanDojo, please refer to the [LeanDojo repository](https://github.com/leanprover/lean-dojo).


### `proof_search.py`

### `search_tree.py`