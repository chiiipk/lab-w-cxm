# data_loader.py
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import datasets
from typing import Dict, Any, List, Optional

class BGEDataset(torch.utils.data.Dataset):
    """
    Supports:
      - sts: needs fields sentence1, sentence2, score
      - pair_classification: sentence1, sentence2, label
      - classification: text, label
      - retrieval (default for msmarco/mteb/*): query + passage (no labels)
    """
    def __init__(self, dataset_name: str, split: str = "train",
                 tokenizer_name: str = "BAAI/bge-m3", max_length: int = 512,
                 task: Optional[str] = None,
                 # flexible fields for custom datasets
                 text_field: str = "text",
                 label_field: str = "label",
                 sentence1_field: str = "sentence1",
                 sentence2_field: str = "sentence2",
                 score_field: str = "score"):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length
        self.text_field = text_field
        self.label_field = label_field
        self.s1 = sentence1_field
        self.s2 = sentence2_field
        self.sc = score_field

        # load HF dataset
        self.dataset = datasets.load_dataset(dataset_name, split=split)

        # infer / set task
        if task is not None:
            self.task_type = task
        else:
            if dataset_name == "sts":
                self.task_type = "sts"
            elif dataset_name == "msmarco" or dataset_name.startswith("mteb/"):
                self.task_type = "retrieval"
            else:
                # default fallback
                self.task_type = "retrieval"

    def __len__(self) -> int:
        return len(self.dataset)

    def _tok(self, text: str):
        return self.tokenizer(
            text, max_length=self.max_length,
            padding='max_length', truncation=True, return_tensors='pt'
        )

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        ex = self.dataset[idx]

        if self.task_type == "sts":
            t1, t2 = ex[self.s1], ex[self.s2]
            y = float(ex[self.sc])
            if y > 1.5:  # normalize 0..5 -> 0..1
                y = y / 5.0
            tok1, tok2 = self._tok(t1), self._tok(t2)
            return {
                'input_ids_1': tok1['input_ids'].squeeze(0),
                'attention_mask_1': tok1['attention_mask'].squeeze(0),
                'input_ids_2': tok2['input_ids'].squeeze(0),
                'attention_mask_2': tok2['attention_mask'].squeeze(0),
                'similarity_score': torch.tensor(y, dtype=torch.float),
                'task_type': 'sts'
            }

        if self.task_type == "pair_classification":
            t1, t2 = ex[self.s1], ex[self.s2]
            y = int(ex[self.label_field])
            tok1, tok2 = self._tok(t1), self._tok(t2)
            return {
                'input_ids_1': tok1['input_ids'].squeeze(0),
                'attention_mask_1': tok1['attention_mask'].squeeze(0),
                'input_ids_2': tok2['input_ids'].squeeze(0),
                'attention_mask_2': tok2['attention_mask'].squeeze(0),
                'label': torch.tensor(y, dtype=torch.long),
                'task_type': 'pair_classification'
            }

        if self.task_type == "classification":
            t = ex[self.text_field]
            y = int(ex[self.label_field])
            tok = self._tok(t)
            return {
                'input_ids': tok['input_ids'].squeeze(0),
                'attention_mask': tok['attention_mask'].squeeze(0),
                'label': torch.tensor(y, dtype=torch.long),
                'task_type': 'classification'
            }

        # retrieval (query + passage) – keep old behavior
        query_key = 'query' if 'query' in ex else 'question'
        passage_key = 'passage' if 'passage' in ex else 'text' if 'text' in ex else 'positive_passages'
        q = ex[query_key]
        p = ex[passage_key][0] if isinstance(ex[passage_key], list) else ex[passage_key]
        tokq, tokp = self._tok(q), self._tok(p)
        return {
            'input_ids_1': tokq['input_ids'].squeeze(0),
            'attention_mask_1': tokq['attention_mask'].squeeze(0),
            'input_ids_2': tokp['input_ids'].squeeze(0),
            'attention_mask_2': tokp['attention_mask'].squeeze(0),
            'task_type': 'retrieval'
        }


def _collate_interleave_pairs(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    """For STS / retrieval / pair_classification: interleave [x1, x2, x1, x2, ...]"""
    ids1 = torch.stack([b['input_ids_1'] for b in batch])
    am1  = torch.stack([b['attention_mask_1'] for b in batch])
    ids2 = torch.stack([b['input_ids_2'] for b in batch])
    am2  = torch.stack([b['attention_mask_2'] for b in batch])

    B, L = ids1.size(0), ids1.size(1)
    input_ids = torch.zeros(B * 2, L, dtype=ids1.dtype)
    attention_mask = torch.zeros(B * 2, L, dtype=am1.dtype)
    input_ids[0::2], input_ids[1::2] = ids1, ids2
    attention_mask[0::2], attention_mask[1::2] = am1, am2

    out = {'input_ids': input_ids, 'attention_mask': attention_mask}
    if 'similarity_score' in batch[0]:
        out['similarity_scores'] = torch.stack([b['similarity_score'] for b in batch])  # [B]
    if 'label' in batch[0]:
        out['labels'] = torch.stack([b['label'] for b in batch])  # [B]
    return out


def _collate_single(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    return {
        'input_ids': torch.stack([b['input_ids'] for b in batch]),
        'attention_mask': torch.stack([b['attention_mask'] for b in batch]),
        'labels': torch.stack([b['label'] for b in batch])
    }


def create_dataloader(dataset_name: str, split: str = "train", batch_size: int = 16,
                      tokenizer_name: str = "BAAI/bge-m3", max_length: int = 512,
                      num_workers: int = 2,
                      task: Optional[str] = None,
                      text_field: str = "text",
                      label_field: str = "label",
                      sentence1_field: str = "sentence1",
                      sentence2_field: str = "sentence2",
                      score_field: str = "score") -> DataLoader:
    """
    Now supports: task in {None/infer, 'sts', 'retrieval', 'classification', 'pair_classification'}.
    Composer’s train_clear.py will keep working:
      - It only reads 'input_ids','attention_mask', and (optionally) 'similarity_scores'.
      - For classification tasks, 'labels' is also included for later eval or if your model uses it.
    """
    dataset = BGEDataset(dataset_name, split, tokenizer_name, max_length, task,
                         text_field, label_field, sentence1_field, sentence2_field, score_field)

    # choose collate
    if dataset.task_type in ["sts", "retrieval", "pair_classification"]:
        collate_fn = _collate_interleave_pairs
    else:
        collate_fn = _collate_single

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn
    )
