"""Data module for the tactic generator."""

import os
import json
import pickle
from tqdm import tqdm
from loguru import logger
import pytorch_lightning as pl
from typing import Optional, List, Dict, Any
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, ByT5Tokenizer

from common import (
    Batch,
    Corpus,
    Example,
    format_state,
    remove_marks,
    format_tactic,
    format_augmented_state,
)


class GeneratorDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        corpus: Corpus,
        keep_marks: bool,
        preds: List[Dict[str, Any]],
        max_inp_seq_len: int,
        max_oup_seq_len: int,
        p_drop: float,
        normalize_tactics: bool,
        tokenizer: ByT5Tokenizer,
        is_train: bool,
        task_type: str="default",  # Allow different patterns
    ) -> None:
        super().__init__()
        self.corpus = corpus
        self.keep_marks = keep_marks
        self.preds = preds
        self.max_inp_seq_len = max_inp_seq_len
        self.max_oup_seq_len = max_oup_seq_len
        self.p_drop = p_drop
        self.tokenizer = tokenizer
        self.is_train = is_train
        self.data = self._load_data(data_path, normalize_tactics)
        self.task_type = task_type

    def _load_data(self, data_path: str, normalize_tactics: bool) -> List[Example]:
        data = []
        for thm in tqdm(json.load(open(data_path))):
            for tac in thm["traced_tactics"]:
                # Get the string tactic
                if "annotated_tactic" in tac:
                    tactic = format_tactic(*tac["annotated_tactic"], normalize_tactics)
                else:
                    tactic = format_tactic(tac["tactic"], [], normalize_tactics)
                if not self.keep_marks:
                    tactic = remove_marks(tactic)
                
                # Get the data
                data.append(
                    {
                        "url": thm["url"],
                        "commit": thm["commit"],
                        "file_path": thm["file_path"],
                        "full_name": thm["full_name"],
                        "state": format_state(tac["state_before"]),  # Current state
                        "tactic": tactic,                            # Tactic to apply
                        "target_state": format_state(tac["state_after"]),  # Target state
                    }
                )

        logger.info(f"{len(data)} examples loaded")
        return data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Example:
        ex = self.data[idx]

        # ---------------------------------------------------------------
        # PREMISE-RELEVANT
        # Augment the state with the premises
        if self.preds is not None:
            file_path = ex["file_path"]
            pred = self.preds[(file_path, ex["full_name"], ex["state"])]
            ex["state"] = format_augmented_state(
                ex["state"],
                pred["retrieved_premises"],
                self.max_inp_seq_len,
                self.p_drop if self.is_train else 0.0,
            )
        # ---------------------------------------------------------------

        if not self.keep_marks:
            ex["state"] = remove_marks(ex["state"])
            ex["target_state"] = remove_marks(ex["target_state"])

        return ex

    def collate(self, examples: List[Example]) -> Batch:
        """
        Grouping multiple examples in a single batch.
        Each key in the dictionary corresponds to a list of examples.
        """
        # Tokenize states
        inputs = [ex["state"] for ex in examples]
        tokenized_state = self.tokenizer(
            inputs,
            padding="longest",
            max_length=self.max_inp_seq_len,
            truncation=True,
            return_tensors="pt",
        )
        
        # Tokenize tactics
        outputs = [ex["tactic"] for ex in examples]
        tokenized_tactic = self.tokenizer(
            outputs,
            padding="longest",
            max_length=self.max_oup_seq_len,
            truncation=True,
            return_tensors="pt",
        )
        output_ids = tokenized_tactic.input_ids
        output_ids[output_ids == self.tokenizer.pad_token_id] = -100  # Let the loss to ignore

        
        batch = {}
        batch["inputs"] = inputs
        batch["input_ids"] = tokenized_state.input_ids
        batch["input_mask"] = tokenized_state.attention_mask
        
        batch["outputs"] = outputs
        batch["output_ids"] = output_ids
        batch["output_mask"] = tokenized_tactic.attention_mask

        # Copy other fields.
        for k in examples[0].keys():
            if k not in batch:
                batch[k] = [ex[k] for ex in examples]

        return batch


class GeneratorDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_path: str,
        keep_marks: bool,
        model_name: str,
        batch_size: int,
        eval_batch_size: int,
        max_inp_seq_len: int,
        max_oup_seq_len: int,
        p_drop: float,
        normalize_tactics: bool,
        num_workers: int,
        corpus_path: Optional[str] = None,
        preds_path: Optional[str] = None,
        task_type: str="default",  # Allow different patterns
    ) -> None:
        super().__init__()
        self.data_path = data_path
        if corpus_path is not None:
            self.corpus = Corpus(corpus_path)
        else:
            self.corpus = None
        self.keep_marks = keep_marks
        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size
        self.max_inp_seq_len = max_inp_seq_len
        self.max_oup_seq_len = max_oup_seq_len
        self.p_drop = p_drop
        self.normalize_tactics = normalize_tactics
        self.num_workers = num_workers
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.task_type = task_type

        if preds_path is None:
            logger.info("Without retrieval data")
            self.preds = None
        else:
            logger.info("With retrieval data")
            self.preds = {}
            for pred in pickle.load(open(preds_path, "rb")):
                ctx = pred["context"]
                self.preds[ctx.path, ctx.theorem_full_name, ctx.state] = pred

    def prepare_data(self) -> None:
        pass

    def setup(self, stage: Optional[str] = None) -> None:
        if stage in (None, "fit"):
            self.ds_train = GeneratorDataset(
                data_path=os.path.join(self.data_path, "train.json"),
                corpus=self.corpus,
                keep_marks=self.keep_marks,
                preds=self.preds,
                max_inp_seq_len=self.max_inp_seq_len,
                max_oup_seq_len=self.max_oup_seq_len,
                p_drop=self.p_drop,
                normalize_tactics=self.normalize_tactics,
                tokenizer=self.tokenizer,
                is_train=True,
                task_type=self.task_type,
            )

        if stage in (None, "fit", "validate"):
            self.ds_val = GeneratorDataset(
                data_path=os.path.join(self.data_path, "val.json"),
                corpus=self.corpus,
                keep_marks=self.keep_marks,
                preds=self.preds,
                max_inp_seq_len=self.max_inp_seq_len,
                max_oup_seq_len=self.max_oup_seq_len,
                p_drop=self.p_drop,
                normalize_tactics=self.normalize_tactics,
                tokenizer=self.tokenizer,
                is_train=False,
                task_type=self.task_type,
            )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.ds_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self.ds_train.collate,
            shuffle=True,
            pin_memory=True,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.ds_val,
            batch_size=self.eval_batch_size,
            num_workers=self.num_workers,
            collate_fn=self.ds_val.collate,
            shuffle=False,
            pin_memory=True,
            drop_last=False,
        )
