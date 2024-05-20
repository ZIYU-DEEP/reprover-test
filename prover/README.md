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

```
CUDA_VISIBLE_DEVICES=1 python prover/evaluate.py  \
   --data-path data/leandojo_benchmark_4/random/  \
   --ckpt_path leandojo-pl-ckpts/generator_random.ckpt  \
   --split test  \    
   --num-workers 5  \   
   --num-gpus 1  \   
   --num-theorems 1  \  # set the number of theorems to evaluate
   --start-ind 10  # set the start ind to evaluate
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




## Search Tree Definitions

The `search_tree.py` file contains the definitions of the search tree used by the prover. It defines the necessary classes and methods for representing and manipulating the search tree during the proof search process.

### Status Enum

The `Status` enum represents the status of a node or a proof search. It can have the following values:

- `PROVED`: The node or search has at least one known proof.
- `FAILED`: The node or search has exhausted its options and cannot be proved within the current run.
- `OPEN`: The node or search has not been proven or given up on yet.

### Node Abstract Base Class

The `Node` class is an abstract base class that defines the common properties and methods for all nodes in the search tree. It has the following abstract properties:

- `status`: The status of the node (`Status` enum).
- `distance_to_proof`: The smallest number of steps to a proof from the node.
- `is_terminal`: Indicates whether the node is a terminal node (i.e., a proof finished node or an error node).

### ProofFinishedNode

The `ProofFinishedNode` class represents a node that indicates a finished proof. It has the following properties:

- `inner`: The `ProofFinished` object representing the finished proof.
- `status`: Always set to `Status.PROVED`.
- `distance_to_proof`: Always set to 0.
- `is_terminal`: Always set to `True`.

### ErrorNode

The `ErrorNode` class represents a node that indicates an error during the proof search. It has the following properties:

- `inner`: The error object (`LeanError`, `TimeoutError`, or `ProofGivenUp`).
- `status`: Always set to `Status.FAILED`.
- `distance_to_proof`: Always set to infinity.
- `is_terminal`: Always set to `True`.

### InternalNode

The `InternalNode` class represents an internal node in the search tree, representing a nonterminal state. It has the following properties and methods:

- `state`: The `TacticState` object representing the goal state of the node.
- `cumulative_logprob`: The sum of action log probabilities along the edges from the root to the node.
- `in_edges`: A list of `Edge` objects representing the edges leading to the node.
- `out_edges`: A list of `Edge` objects representing the edges leading out of the node. When a node is explored, this list is populated and must not change after that.
- `is_terminal`: Always set to `False`.
- `status`: The status of the node, computed based on the status of its children.
- `distance_to_proof`: The number of steps separating the node from the end of a proof along the optimal path. Updated as needed by children.
- `is_explored`: Indicates whether the node has been explored (i.e., its outgoing edges have been generated).
- `priority`: The priority of the node, used for sorting in the best-first search. Nodes are compared by inverse priority, so a node with higher priority has a lower value.
- `extract_proof`: Extracts a proof of the node as a sequence of edges if the node is proved.
- `check_invariants`: Performs sanity checks on the node and its properties.

### Edge

The `Edge` class represents an edge in the search tree, representing a tactic. It has the following properties:

- `tactic`: The tactic string associated with the edge.
- `src`: The source `InternalNode` of the edge.
- `dst`: The destination `Node` of the edge.
- `distance_to_proof`: The distance to a proof from the edge, computed as 1 plus the distance to proof of the destination node.

## Best-First Search

The `search_tree.py` file provides the necessary data structures and methods for performing best-first search in the proof search process. The `InternalNode` class represents the nodes in the search tree, and the `Edge` class represents the edges connecting the nodes.

During the best-first search, the prover maintains a priority queue of `InternalNode` objects. The priority of a node is determined by its `cumulative_logprob`, which is the sum of the action log probabilities along the edges from the root to the node. The nodes with higher `cumulative_logprob` are considered to have higher priority and are explored first.

The search starts from the root node and expands the nodes in the priority queue based on their priority. When a node is expanded, its outgoing edges are generated using the tactic generator. The destination nodes of the edges are added to the priority queue if they are not terminal nodes (i.e., not `ProofFinishedNode` or `ErrorNode`).

The search continues until a proof is found (`ProofFinishedNode` is reached), the search space is exhausted (no more nodes to explore), or a timeout occurs. The `extract_proof` method can be used to extract the sequence of edges that form a proof if the search is successful.

The `status` and `distance_to_proof` properties of the nodes are updated recursively based on the status and distances of their children. This allows the prover to efficiently determine the status of the search and the optimal path to a proof.

The `check_invariants` method is used for debugging purposes to perform sanity checks on the nodes and ensure the integrity of the search tree.

## Acknowledgments

This project is built upon the LeanDojo framework and utilizes large language models for theorem proving. We would like to acknowledge the contributions of the LeanDojo team and the developers of the underlying language models.

For more information about LeanDojo, please refer to the [LeanDojo repository](https://github.com/leanprover/lean-dojo).