# """
# Production BGE-M3 Dataset Loader with MTEB Support
# Clean implementation for STS similarity and MTEB retrieval tasks
# """

# import torch
# from torch.utils.data import DataLoader
# from transformers import AutoTokenizer
# import datasets
# from typing import Dict, Any

# class BGEDataset(torch.utils.data.Dataset):
#     """Production dataset for BGE-M3 training with proper tensor handling"""
    
#     def __init__(self, dataset_name: str, split: str = "train", 
#                  tokenizer_name: str = "BAAI/bge-m3", max_length: int = 512):
#         self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
#         self.max_length = max_length
        
#         # Production dataset loading
#         if dataset_name == "sts":
#             self.dataset = datasets.load_dataset("mteb/stsbenchmark-sts", split=split)
#             self.task_type = "similarity"
#         elif dataset_name == "msmarco":
#             self.dataset = datasets.load_dataset("ms_marco", "v2.1", split=split)
#             self.task_type = "retrieval"
#         elif dataset_name.startswith("mteb/"):
#             # MTEB benchmark datasets
#             self.dataset = datasets.load_dataset(dataset_name, split=split)
#             self.task_type = "retrieval"
#         else:
#             raise ValueError(f"Unsupported dataset: {dataset_name}. Supported: sts, msmarco, mteb/*")
    
#     def __len__(self) -> int:
#         return len(self.dataset)
    
#     def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
#         item = self.dataset[idx]
        
#         if self.task_type == "similarity":
#             # STS: Clean sentence pair processing
#             inputs1 = self.tokenizer(
#                 item['sentence1'], max_length=self.max_length,
#                 padding='max_length', truncation=True, return_tensors='pt'
#             )
#             inputs2 = self.tokenizer(
#                 item['sentence2'], max_length=self.max_length,
#                 padding='max_length', truncation=True, return_tensors='pt'
#             )
            
#             return {
#                 'input_ids_1': inputs1['input_ids'].squeeze(0),  # [seq_length]
#                 'attention_mask_1': inputs1['attention_mask'].squeeze(0),  # [seq_length]
#                 'input_ids_2': inputs2['input_ids'].squeeze(0),  # [seq_length]
#                 'attention_mask_2': inputs2['attention_mask'].squeeze(0),  # [seq_length]
#                 'similarity_score': torch.tensor(item['score'], dtype=torch.float),
#                 'task_type': 'sts'
#             }
        
#         else:  # retrieval
#             # MTEB/MSMARCO: Clean query-passage processing
#             query_key = 'query' if 'query' in item else 'question'
#             passage_key = 'passage' if 'passage' in item else 'text' if 'text' in item else 'positive_passages'
            
#             query_inputs = self.tokenizer(
#                 item[query_key], max_length=self.max_length,
#                 padding='max_length', truncation=True, return_tensors='pt'
#             )
#             passage_inputs = self.tokenizer(
#                 item[passage_key][0] if isinstance(item[passage_key], list) else item[passage_key], 
#                 max_length=self.max_length, padding='max_length', truncation=True, return_tensors='pt'
#             )
            
#             return {
#                 'input_ids_1': query_inputs['input_ids'].squeeze(0),  # [seq_length]
#                 'attention_mask_1': query_inputs['attention_mask'].squeeze(0),  # [seq_length]
#                 'input_ids_2': passage_inputs['input_ids'].squeeze(0),  # [seq_length]
#                 'attention_mask_2': passage_inputs['attention_mask'].squeeze(0),  # [seq_length]
#                 'task_type': 'retrieval'
#             }

# def create_dataloader(dataset_name: str, split: str = "train", batch_size: int = 16, 
#                      tokenizer_name: str = "BAAI/bge-m3", max_length: int = 512,
#                      num_workers: int = 2) -> DataLoader:
#     """Create production DataLoader with proper tensor handling"""
#     dataset = BGEDataset(dataset_name, split, tokenizer_name, max_length)
    
#     def collate_fn(batch):
#         """Production collate function - maintains paired structure"""
#         # Stack tensors to create proper batch dimensions
#         input_ids_1 = torch.stack([item['input_ids_1'] for item in batch])  # [batch_size, seq_length]
#         attention_mask_1 = torch.stack([item['attention_mask_1'] for item in batch])  # [batch_size, seq_length]
#         input_ids_2 = torch.stack([item['input_ids_2'] for item in batch])  # [batch_size, seq_length]
#         attention_mask_2 = torch.stack([item['attention_mask_2'] for item in batch])  # [batch_size, seq_length]
        
#         # Interleave pairs: [sent1_1, sent2_1, sent1_2, sent2_2, ...]
#         batch_size = len(batch)
#         input_ids = torch.zeros(batch_size * 2, input_ids_1.size(1), dtype=input_ids_1.dtype)
#         attention_mask = torch.zeros(batch_size * 2, attention_mask_1.size(1), dtype=attention_mask_1.dtype)
        
#         input_ids[0::2] = input_ids_1  # Even indices: first sentences
#         input_ids[1::2] = input_ids_2  # Odd indices: second sentences
#         attention_mask[0::2] = attention_mask_1
#         attention_mask[1::2] = attention_mask_2
        
#         result = {
#             'input_ids': input_ids,
#             'attention_mask': attention_mask,
#         }
        
#         # Add task-specific targets (infer task from data, no string fields for Composer)
#         if batch[0]['task_type'] == 'sts':
#             similarity_scores = torch.stack([item['similarity_score'] for item in batch])
#             result['similarity_scores'] = similarity_scores
        
#         return result
    
#     return DataLoader(
#         dataset, batch_size=batch_size, shuffle=(split == "train"),
#         num_workers=num_workers, collate_fn=collate_fn, pin_memory=True
#     )

import argparse
from dataclasses import dataclass
from typing import Dict, Any, Optional, List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel
import datasets
from datasets import load_dataset
from scipy.stats import spearmanr
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score


# ----------------------------
# Utilities
# ----------------------------
def mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    # last_hidden_state: [B, L, H], attention_mask: [B, L]
    mask = attention_mask.unsqueeze(-1)  # [B, L, 1]
    summed = (last_hidden_state * mask).sum(dim=1)  # [B, H]
    counts = mask.sum(dim=1).clamp(min=1)           # [B, 1]
    return summed / counts


def pair_feature(e1: torch.Tensor, e2: torch.Tensor) -> torch.Tensor:
    # Robust pair features for sentence-pair classification
    return torch.cat([e1, e2, torch.abs(e1 - e2), e1 * e2], dim=-1)


def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


# ----------------------------
# Datasets
# ----------------------------
class STSDataset(torch.utils.data.Dataset):
    """For STS: expects fields sentence1, sentence2, score (0..5 or 0..1)."""
    def __init__(self, hf_ds, tokenizer, max_length: int = 512, s1_field="sentence1", s2_field="sentence2", score_field="score"):
        self.ds = hf_ds
        self.tok = tokenizer
        self.L = max_length
        self.s1, self.s2, self.sc = s1_field, s2_field, score_field

    def __len__(self): return len(self.ds)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        ex = self.ds[idx]
        t1, t2 = ex[self.s1], ex[self.s2]
        y = float(ex[self.sc])
        # Normalize to [0,1] if it looks like 0..5
        if y > 1.5:  # heuristic for STS Benchmark style scores
            y = y / 5.0
        tok1 = self.tok(t1, max_length=self.L, truncation=True, padding="max_length", return_tensors="pt")
        tok2 = self.tok(t2, max_length=self.L, truncation=True, padding="max_length", return_tensors="pt")
        return {
            "input_ids_1": tok1["input_ids"].squeeze(0),
            "attention_mask_1": tok1["attention_mask"].squeeze(0),
            "input_ids_2": tok2["input_ids"].squeeze(0),
            "attention_mask_2": tok2["attention_mask"].squeeze(0),
            "label": torch.tensor(y, dtype=torch.float)
        }


class SingleClassDataset(torch.utils.data.Dataset):
    """Single-sentence classification. Fields: text_field, label_field."""
    def __init__(self, hf_ds, tokenizer, max_length=256, text_field="text", label_field="label"):
        self.ds = hf_ds
        self.tok = tokenizer
        self.L = max_length
        self.text_field, self.label_field = text_field, label_field

    def __len__(self): return len(self.ds)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        ex = self.ds[idx]
        text = ex[self.text_field]
        label = int(ex[self.label_field])
        tok = self.tok(text, max_length=self.L, truncation=True, padding="max_length", return_tensors="pt")
        return {
            "input_ids": tok["input_ids"].squeeze(0),
            "attention_mask": tok["attention_mask"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.long)
        }


class PairClassDataset(torch.utils.data.Dataset):
    """Sentence-pair classification. Fields: sentence1, sentence2, label."""
    def __init__(self, hf_ds, tokenizer, max_length=256, s1_field="sentence1", s2_field="sentence2", label_field="label"):
        self.ds = hf_ds
        self.tok = tokenizer
        self.L = max_length
        self.s1, self.s2, self.label_field = s1_field, s2_field, label_field

    def __len__(self): return len(self.ds)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        ex = self.ds[idx]
        t1, t2 = ex[self.s1], ex[self.s2]
        label = int(ex[self.label_field])
        tok1 = self.tok(t1, max_length=self.L, truncation=True, padding="max_length", return_tensors="pt")
        tok2 = self.tok(t2, max_length=self.L, truncation=True, padding="max_length", return_tensors="pt")
        return {
            "input_ids_1": tok1["input_ids"].squeeze(0),
            "attention_mask_1": tok1["attention_mask"].squeeze(0),
            "input_ids_2": tok2["input_ids"].squeeze(0),
            "attention_mask_2": tok2["attention_mask"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.long)
        }


# ----------------------------
# Collate fns
# ----------------------------
def collate_sts(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "input_ids_1": torch.stack([b["input_ids_1"] for b in batch]),
        "attention_mask_1": torch.stack([b["attention_mask_1"] for b in batch]),
        "input_ids_2": torch.stack([b["input_ids_2"] for b in batch]),
        "attention_mask_2": torch.stack([b["attention_mask_2"] for b in batch]),
        "labels": torch.stack([b["label"] for b in batch])  # float in [0,1]
    }

def collate_single(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "input_ids": torch.stack([b["input_ids"] for b in batch]),
        "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
        "labels": torch.stack([b["label"] for b in batch])
    }

def collate_pair(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "input_ids_1": torch.stack([b["input_ids_1"] for b in batch]),
        "attention_mask_1": torch.stack([b["attention_mask_1"] for b in batch]),
        "input_ids_2": torch.stack([b["input_ids_2"] for b in batch]),
        "attention_mask_2": torch.stack([b["attention_mask_2"] for b in batch]),
        "labels": torch.stack([b["label"] for b in batch])
    }


# ----------------------------
# Models (BGE backbone + light heads)
# ----------------------------
class BGEBackbone(nn.Module):
    def __init__(self, model_name: str = "BAAI/bge-m3"):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)

    def forward_embed(self, input_ids, attention_mask) -> torch.Tensor:
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        emb = mean_pool(out.last_hidden_state, attention_mask)
        emb = nn.functional.normalize(emb, p=2, dim=-1)
        return emb


class SingleClassifier(nn.Module):
    def __init__(self, backbone: BGEBackbone, num_labels: int):
        super().__init__()
        self.backbone = backbone
        hidden = self.backbone.encoder.config.hidden_size
        self.head = nn.Linear(hidden, num_labels)

    def forward(self, input_ids, attention_mask):
        emb = self.backbone.forward_embed(input_ids, attention_mask)  # [B, H]
        logits = self.head(emb)  # [B, C]
        return logits


class PairClassifier(nn.Module):
    def __init__(self, backbone: BGEBackbone, num_labels: int):
        super().__init__()
        self.backbone = backbone
        hidden = self.backbone.encoder.config.hidden_size
        self.head = nn.Linear(hidden * 4, num_labels)  # concat features
    def forward(self, input_ids_1, attention_mask_1, input_ids_2, attention_mask_2):
        e1 = self.backbone.forward_embed(input_ids_1, attention_mask_1)
        e2 = self.backbone.forward_embed(input_ids_2, attention_mask_2)
        feat = pair_feature(e1, e2)
        logits = self.head(feat)
        return logits


class STSRegressor(nn.Module):
    """Predict similarity via cosine(e1,e2); train with MSE to gold in [0,1]."""
    def __init__(self, backbone: BGEBackbone):
        super().__init__()
        self.backbone = backbone

    def forward(self, input_ids_1, attention_mask_1, input_ids_2, attention_mask_2):
        e1 = self.backbone.forward_embed(input_ids_1, attention_mask_1)
        e2 = self.backbone.forward_embed(input_ids_2, attention_mask_2)
        cos = torch.sum(e1 * e2, dim=-1)  # cosine since embeddings are normalized
        return cos  # in [-1,1], but typically ~[0,1] for similar pairs


# ----------------------------
# Train/Eval loops
# ----------------------------
def eval_classification(model, dataloader, device) -> Dict[str, float]:
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(**{k: batch[k] for k in batch if k in ["input_ids", "attention_mask",
                                                                  "input_ids_1", "attention_mask_1",
                                                                  "input_ids_2", "attention_mask_2"]})
            preds = torch.argmax(logits, dim=-1)
            all_preds.append(preds.cpu().numpy())
            all_labels.append(batch["labels"].cpu().numpy())
    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_labels)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    }


def eval_sts(model: STSRegressor, dataloader, device) -> Dict[str, float]:
    model.eval()
    preds, gold = [], []
    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            cos = model(batch["input_ids_1"], batch["attention_mask_1"],
                        batch["input_ids_2"], batch["attention_mask_2"])  # [-1,1]
            # map to [0,1] to compare with labels
            pred = (cos.clamp(-1, 1) + 1.0) / 2.0
            preds.append(pred.cpu().numpy())
            gold.append(batch["labels"].cpu().numpy())
    y_pred = np.concatenate(preds)
    y_true = np.concatenate(gold)
    # Convert to 0..5 scale for reporting Spearman like STS-B
    spearman = spearmanr(y_true * 5.0, y_pred * 5.0).statistic
    mse = float(np.mean((y_true - y_pred) ** 2))
    return {"spearman": float(spearman), "mse": mse}


def train_loop(model, train_loader, valid_loader, task: str, device, epochs: int, lr: float):
    model.to(device)
    if task in ["classification", "pair_classification"]:
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    else:  # sts
        criterion = nn.MSELoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    for ep in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            if task == "classification":
                logits = model(batch["input_ids"], batch["attention_mask"])
                loss = criterion(logits, batch["labels"])
            elif task == "pair_classification":
                logits = model(batch["input_ids_1"], batch["attention_mask_1"],
                               batch["input_ids_2"], batch["attention_mask_2"])
                loss = criterion(logits, batch["labels"])
            else:  # sts
                cos = model(batch["input_ids_1"], batch["attention_mask_1"],
                            batch["input_ids_2"], batch["attention_mask_2"])
                pred = (cos.clamp(-1, 1) + 1.0) / 2.0  # [0,1]
                loss = criterion(pred, batch["labels"])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / max(1, len(train_loader))

        # Eval each epoch
        with torch.no_grad():
            if task == "classification":
                metrics = eval_classification(model, valid_loader, device)
            elif task == "pair_classification":
                metrics = eval_classification(model, valid_loader, device)
            else:
                metrics = eval_sts(model, valid_loader, device)

        print(f"[Epoch {ep}] train_loss={avg_loss:.4f} | {metrics}")

    return model


# ----------------------------
# Main
# ----------------------------
def build_dataloaders(args, tokenizer):
    # Load raw HF dataset
    hf = load_dataset(args.dataset_name)
    # Choose splits (fallbacks)
    train_split = args.train_split or ("train" if "train" in hf else list(hf.keys())[0])
    valid_split = args.valid_split or ("validation" if "validation" in hf else
                                       "dev" if "dev" in hf else
                                       "test" if "test" in hf else list(hf.keys())[-1])

    if args.task == "sts":
        ds_train = STSDataset(hf[train_split], tokenizer, max_length=args.max_length,
                              s1_field=args.sentence1_field, s2_field=args.sentence2_field,
                              score_field=args.score_field)
        ds_valid = STSDataset(hf[valid_split], tokenizer, max_length=args.max_length,
                              s1_field=args.sentence1_field, s2_field=args.sentence2_field,
                              score_field=args.score_field)
        collate_fn = collate_sts

    elif args.task == "classification":
        ds_train = SingleClassDataset(hf[train_split], tokenizer, max_length=args.max_length,
                                      text_field=args.text_field, label_field=args.label_field)
        ds_valid = SingleClassDataset(hf[valid_split], tokenizer, max_length=args.max_length,
                                      text_field=args.text_field, label_field=args.label_field)
        collate_fn = collate_single

    elif args.task == "pair_classification":
        ds_train = PairClassDataset(hf[train_split], tokenizer, max_length=args.max_length,
                                    s1_field=args.sentence1_field, s2_field=args.sentence2_field,
                                    label_field=args.label_field)
        ds_valid = PairClassDataset(hf[valid_split], tokenizer, max_length=args.max_length,
                                    s1_field=args.sentence1_field, s2_field=args.sentence2_field,
                                    label_field=args.label_field)
        collate_fn = collate_pair
    else:
        raise ValueError(f"Unknown task: {args.task}")

    train_loader = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True, collate_fn=collate_fn)
    valid_loader = DataLoader(ds_valid, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True, collate_fn=collate_fn)
    return train_loader, valid_loader


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, required=True,
                        choices=["sts", "classification", "pair_classification"])
    parser.add_argument("--dataset_name", type=str, required=True,
                        help="HuggingFace dataset name, e.g. mteb/stsbenchmark-sts, ag_news, glue/mrpc, etc.")
    parser.add_argument("--model_name", type=str, default="BAAI/bge-m3")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)

    # Flexible field names
    parser.add_argument("--text_field", type=str, default="text")
    parser.add_argument("--label_field", type=str, default="label")
    parser.add_argument("--sentence1_field", type=str, default="sentence1")
    parser.add_argument("--sentence2_field", type=str, default="sentence2")
    parser.add_argument("--score_field", type=str, default="score")

    # Optional splits
    parser.add_argument("--train_split", type=str, default=None)
    parser.add_argument("--valid_split", type=str, default=None)

    # For classification tasks
    parser.add_argument("--num_labels", type=int, default=None,
                        help="If not set, infer from dataset unique labels at load time.")

    args = parser.parse_args()
    set_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # Build loaders
    train_loader, valid_loader = build_dataloaders(args, tokenizer)

    # Infer num_labels if needed (classification only)
    if args.task in ["classification", "pair_classification"]:
        if args.num_labels is None:
            # Peek labels from a small slice
            lab = []
            for _, batch in zip(range(4), train_loader):
                lab.append(batch["labels"].view(-1).tolist())
            labs = set([int(x) for sub in lab for x in sub])
            num_labels = max(labs) + 1 if labs else 2
        else:
            num_labels = args.num_labels
    else:
        num_labels = None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build model
    backbone = BGEBackbone(args.model_name)
    if args.task == "classification":
        model = SingleClassifier(backbone, num_labels=num_labels)
    elif args.task == "pair_classification":
        model = PairClassifier(backbone, num_labels=num_labels)
    else:  # sts
        model = STSRegressor(backbone)

    # Train
    model = train_loop(model, train_loader, valid_loader, args.task, device, args.epochs, args.lr)

    # Final Eval (same as per-epoch)
    print("\n=== Final evaluation ===")
    if args.task == "sts":
        metrics = eval_sts(model, valid_loader, device)
        print(metrics)
    else:
        metrics = eval_classification(model, valid_loader, device)
        print(metrics)


if __name__ == "__main__":
    main()
