# README by GPT

## `datamodule.py`

### Overview
The `datamodule.py` file defines the data handling infrastructure for a PyTorch Lightning-based tactic generator system. It facilitates the loading, preprocessing, and batching of data for training a model that generates tactics as part of a theorem proving process.

### Key Components

#### `GeneratorDataset`
- **Purpose**: Manages individual data items, preparing them for processing by the model. It formats the theorem proving data into a usable format for machine learning models.
- **Functionality**:
  - Parses and preprocesses theorems and associated tactics from JSON files.
  - Supports optional data augmentation through retrieved premises if available.

#### `GeneratorDataModule`
- **Purpose**: Orchestrates the overall data pipeline for both training and validation phases, using the `GeneratorDataset`.
- **Functionality**:
  - Sets up training and validation datasets.
  - Provides DataLoader configurations that handle the efficient loading and batching of data.

### Methodologies

- **Data Loading**: Data is loaded from structured JSON files, ensuring each theorem and its tactics are correctly parsed and formatted.
- **Tokenization**: Uses the ByT5 tokenizer to handle a wide range of tokens, including those specific to mathematical and programming contexts.
- **Batch Processing**: Implements a custom collate function to prepare data batches in a format compatible with the model's input requirements.

### Usage
This module is used within a larger training script to manage data operations, abstracting away the complexities of data handling from the main model training logic.

### Configuration
- Configurable to work with or without retrieval augmentation, allowing for flexibility depending on the availability of additional contextual data.
- Parameterizable to adjust tokenization lengths, dropout probabilities, and batch sizes to suit different training setups and hardware capacities.


## `model.py`

### Overview
The `model.py` file contains the implementation of a PyTorch Lightning module named `RetrievalAugmentedGenerator`, designed to generate tactics for theorem proving. This module integrates a T5 model for generating tactic predictions, potentially augmented with premises retrieved by a `PremiseRetriever`.

### Key Components

- **`TopkAccuracy` (Class):** A custom metric class for evaluating the top-k accuracy of model predictions.
- **`TacticGenerator` (Abstract Base Class):** Defines the interface for tactic generators, requiring implementations for `generate` and `batch_generate` methods.
- **`RetrievalAugmentedGenerator` (Class):** Inherits from `TacticGenerator` and `pytorch_lightning.LightningModule`, encapsulating the main functionality of the tactic generation model.

### Main Methods

- **`__init__`:** Initializes the model, setting up the T5 transformer, tokenizer, and optional premise retriever.
- **`forward`:** Defines the forward pass of the model, computing the loss directly from input and output tensors.
- **`training_step`:** Conducts a single step of model training, calculating loss and logging key information.
- **`validation_step`:** Similar to `training_step`, but also includes generation of tactics using beam search to evaluate the model during validation.
- **`generate` and `batch_generate`:** Methods for generating tactics based on input state; `generate` handles single instances, while `batch_generate` is designed for batch processing.

### Utilities

- **`_log_io_texts`:** Logs the input and output texts to TensorBoard for monitoring during training or validation phases.
- **`configure_optimizers`:** Sets up optimizers and learning rate schedules based on the model configuration.

### Usage
This module is typically used in a training script where instances of `RetrievalAugmentedGenerator` are created, trained on batches of theorem proving tasks, and evaluated on their ability to generate appropriate tactics.

### Files and Dependencies
- Relies on `common.py` for utility functions and `retrieval.model` for optional premise retrieval functionality.
- Utilizes libraries such as `torch`, `pytorch_lightning`, `transformers`, and `torchmetrics`.

## The Look of the Data
Here we show one example entry from the `test.json` in the `random` split of the LeanDojo Benchmark from the [Mathlib](https://github.com/leanprover-community/mathlib4/blob/fe4454af900584467d21f4fd4fe951d29d9332a7/Mathlib/Data/Complex/Abs.lean#L327) repository.

The original theorem is:
```bash
theorem lim_im (f : CauSeq ℂ Complex.abs) : lim (cauSeqIm f) = (lim f).im := by
  rw [lim_eq_lim_im_add_lim_re]; simp [ofReal']
```
Basically, for each theorem, the following information is recorded:
- `url` / `commit` / `file_path` / `full_name` / `start` / `end`
- `traced_tactics`
    - `tactic` / `annotated tactic` (an annotation and a list) / `state_before` / `state_after`
```bash
{
"url": "https://github.com/leanprover-community/mathlib4", 
"commit": "fe4454af900584467d21f4fd4fe951d29d9332a7", 
"file_path": "Mathlib/Data/Complex/Abs.lean", 
"full_name": "Complex.lim_im", 
"start": [327, 1], 
"end": [328, 48], 
"traced_tactics": [
    {
        "tactic": "rw [lim_eq_lim_im_add_lim_re]",
    
        "annotated_tactic": [
            "rw [<a>lim_eq_lim_im_add_lim_re</a>]", 
            [{"full_name": "Complex.lim_eq_lim_im_add_lim_re", 
              "def_path": "Mathlib/Data/Complex/Abs.lean", 
              "def_pos": [313, 9], 
              "def_end_pos": [313, 33]}]], 
             
        "state_before": "f : CauSeq \u2102 \u21d1abs'\n\u22a2 CauSeq.lim (cauSeqIm f) = (CauSeq.lim f).im", 
           
        "state_after": "f : CauSeq \u2102 \u21d1abs'\n\u22a2 CauSeq.lim (cauSeqIm f) = (\u2191(CauSeq.lim (cauSeqRe f)) + \u2191(CauSeq.lim (cauSeqIm f)) * I).im"
    }, 
           
    {
        "tactic": "simp [ofReal']", 
      
        "annotated_tactic": [
            "simp [<a>ofReal'</a>]", 
            [{"full_name": "Complex.ofReal'", 
              "def_path": "Mathlib/Data/Complex/Basic.lean", 
              "def_pos": [89, 5], 
              "def_end_pos": [89, 12]}]], 
          
        "state_before":"f : CauSeq \u2102 \u21d1abs'\n\u22a2 CauSeq.lim (cauSeqIm f) = (\u2191(CauSeq.lim (cauSeqRe f)) + \u2191(CauSeq.lim (cauSeqIm f)) * I).im"}, 

        "state_after": "no goals"
    }]
}
```

