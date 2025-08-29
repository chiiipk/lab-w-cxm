#!/usr/bin/env python3
"""
Production BGE-M3 Pruning Training Script
Focused on head/layer/intermediate pruning with MTEB support
"""

import torch
import argparse
from pathlib import Path
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent))

# Fix datasets import conflict by importing before local modules
import datasets as hf_datasets

from composer import Trainer, DataSpec
from composer.utils import reproducibility
from omegaconf import OmegaConf

# Import BGE components
from models.composer_bge_m3 import ComposerBGEM3
from callbacks.pruning_callback import PruningCallback
print("[BOOT] train_clean.py started", flush=True)



# def get_num_samples_in_batch(batch):
#     """Get actual batch size from interleaved tensor pairs"""
#     return batch['input_ids'].size(0) // 2

# def split_batch(batch, microbatch_size):
#     """Custom batch splitting for interleaved sentence pairs"""
#     if microbatch_size is None:
#         return [batch]
    
#     actual_batch_size = batch['input_ids'].size(0) // 2  # True batch size
#     if actual_batch_size <= microbatch_size:
#         return [batch]
    
#     # Split into microbatches
#     microbatches = []
#     for start_idx in range(0, actual_batch_size, microbatch_size):
#         end_idx = min(start_idx + microbatch_size, actual_batch_size)
        
#         # For interleaved tensors, we need indices: [start*2, start*2+1, ..., end*2-1]
#         interleaved_indices = []
#         for i in range(start_idx, end_idx):
#             interleaved_indices.extend([i * 2, i * 2 + 1])
        
#         microbatch = {
#             'input_ids': batch['input_ids'][interleaved_indices],
#             'attention_mask': batch['attention_mask'][interleaved_indices],
#         }
        
#         # Handle similarity_scores (has original batch dimension)
#         if 'similarity_scores' in batch:
#             microbatch['similarity_scores'] = batch['similarity_scores'][start_idx:end_idx]
        
#         microbatches.append(microbatch)
    
#     return microbatches

def setup_optimizer(model, cfg):
    """Production optimizer with stable L0 learning rates"""
    if not hasattr(model, 'l0_module'):
        return torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    
    # Stable parameter groups for L0 pruning
    model_params = [p for n, p in model.named_parameters() 
                   if not n.startswith('l0_module') and p.requires_grad]
    mask_params = [p for p in model.l0_module.masks.parameters() if p.requires_grad]
    lambda_params = [p for p in model.l0_module.lambdas.parameters() if p.requires_grad]
    
    param_groups = [
        {'params': model_params, 'lr': cfg.lr, 'weight_decay': cfg.weight_decay},
        {'params': mask_params, 'lr': cfg.lr * 2, 'weight_decay': 0.0},  # Reduced from 10x
        {'params': lambda_params, 'lr': cfg.lr * 5, 'weight_decay': 0.0}  # Reduced from 100x
    ]
    
    return torch.optim.AdamW(param_groups, betas=cfg.get('betas', [0.9, 0.999]), eps=cfg.get('eps', 1e-8))

# def main():
#     parser = argparse.ArgumentParser(description='BGE-M3 Pruning Training')
#     parser.add_argument('config', type=str, help='Path to configuration file')
#     parser.add_argument('--seed', type=int, default=42, help='Random seed')
#     parser.add_argument('--dataset', type=str, default='sts', 
#                        help='Dataset to use (sts, msmarco, or mteb/dataset-name)')
    
#     args = parser.parse_args()
    
#     # Set seed
#     reproducibility.seed_all(args.seed)
    
#     # Load configuration
#     cfg = OmegaConf.load(args.config)
#     print(f"🔥 BGE-M3 Production Pruning Training")
#     print(f"📁 Config: {args.config}")
#     print(f"📊 Dataset: {args.dataset}")
#     print(f"🎯 Pruning: head/layer/intermediate only")
    
#     # Setup model
#     print("🚀 Loading BGE-M3 with L0 pruning...")
#     model = ComposerBGEM3(cfg.model)
    
#     # Print model info
#     total_params = sum(p.numel() for p in model.parameters())
#     trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
#     print(f"📈 Total parameters: {total_params:,}")
#     print(f"📈 Trainable parameters: {trainable_params:,}")
    
#     # Setup data with MTEB support
#     print(f"📊 Loading {args.dataset} dataset from HuggingFace...")
#     train_dataloader = create_dataloader(
#         dataset_name=args.dataset,
#         split="train",
#         batch_size=cfg.get('batch_size', 16),
#         max_length=cfg.get('max_length', 512)
#     )
    
#     # Use test split for datasets that don't have validation
#     eval_split = "validation" if args.dataset == "sts" else "test"
#     eval_dataloader = create_dataloader(
#         dataset_name=args.dataset,
#         split=eval_split,
#         batch_size=cfg.get('batch_size', 16),
#         max_length=cfg.get('max_length', 512)
#     )
    
#     # Wrap dataloaders in DataSpec for Composer batch size detection and custom splitting
#     train_dataspec = DataSpec(
#         dataloader=train_dataloader, 
#         get_num_samples_in_batch=get_num_samples_in_batch,
#         split_batch=split_batch
#     )
#     eval_dataspec = DataSpec(
#         dataloader=eval_dataloader, 
#         get_num_samples_in_batch=get_num_samples_in_batch,
#         split_batch=split_batch
#     )
    
#     # Setup optimizer
#     optimizer = setup_optimizer(model, cfg.optimizer)
    
#     # Setup callbacks
#     callbacks = [PruningCallback(log_interval=500)]
    
#     # Convert PyTorch device names to Composer device names
#     device = cfg.get('device', 'gpu')  # Default to 'gpu' for Composer
    
#     # If user specified 'cuda' or torch.cuda.is_available(), use 'gpu'
#     if device == 'cuda' or (device == 'auto' and torch.cuda.is_available()):
#         device = 'gpu'
#     elif device == 'cpu':
#         device = 'cpu'
#     else:
#         # Default fallback
#         device = 'gpu' if torch.cuda.is_available() else 'cpu'
    
#     # Setup trainer with DataSpec for proper batch size detection
#     trainer = Trainer(
#         model=model,
#         train_dataloader=train_dataspec,
#         eval_dataloader=eval_dataspec,
#         max_duration=cfg.max_duration,
#         eval_interval=cfg.eval_interval,
#         save_interval=cfg.get('save_interval', '500ba'),
#         save_folder=cfg.get('save_folder', 'experiments/checkpoints'),
#         device=device,
#         callbacks=callbacks,
#         optimizers=optimizer,
#         precision=cfg.get('precision', 'amp_bf16'),
#     )
#     # Start training
#     print("🎯 Starting BGE-M3 pruning training...")
#     trainer.fit()
    
#     # Print final results
#     print(f"✅ Production training completed!")
#     if hasattr(model, 'l0_module'):
#         zs = model.l0_module()
#         print(f"🎯 Pruning results:")
#         for mask_name, mask_tensor in zs.items():
#             sparsity = (mask_tensor == 0).float().mean().item()
#             print(f"  {mask_name}: {sparsity:.1%} sparsity")
    
#     # Save pruned model in HuggingFace format
#     hf_save_path = cfg.get('save_folder', 'experiments/production') + '_hf'
#     model.save_pruned_hf_model(hf_save_path)
#     print(f"💾 Model ready for deployment")

# if __name__ == "__main__":
#     main()

# ... (imports giữ nguyên)

from data_loader import create_dataloader

def get_num_samples_in_batch(batch):
    """
    Get logical batch size.
    - If interleaved (2*B x L): return B
    - Else (single classification B x L): return B
    """
    if batch['input_ids'].dim() == 2:  # [N, L]
        N = batch['input_ids'].size(0)
        # Heuristic: if even and came from interleave, mask length equal and divisible by 2
        if 'similarity_scores' in batch or ('labels' in batch and N % 2 == 0 and batch['attention_mask'].size(0) == N):
            return N // 2
        return N
    return batch['input_ids'].size(0)

def split_batch(batch, microbatch_size):
    """
    Custom microbatch splitting.
    - If interleaved pairs: slice by pair (2 indices per example).
    - Else (single classification): normal slicing by rows.
    """
    if microbatch_size is None:
        return [batch]

    N = batch['input_ids'].size(0)
    interleaved = False
    if 'similarity_scores' in batch:
        interleaved = True
        true_B = batch['similarity_scores'].size(0)
    elif 'labels' in batch and N % 2 == 0 and batch['attention_mask'].size(0) == N and batch['labels'].size(0) == N // 2:
        # pair_classification case (labels size B, tensors size 2B)
        interleaved = True
        true_B = batch['labels'].size(0)
    else:
        true_B = N  # single classification or retrieval without labels

    if not interleaved:
        # Single-sentence classification or plain sequences — slice rows directly
        chunks = []
        for start in range(0, true_B, microbatch_size):
            end = min(start + microbatch_size, true_B)
            mb = {
                'input_ids': batch['input_ids'][start:end],
                'attention_mask': batch['attention_mask'][start:end],
            }
            if 'labels' in batch:
                mb['labels'] = batch['labels'][start:end]
            chunks.append(mb)
        return chunks

    # Interleaved case
    if true_B <= microbatch_size:
        return [batch]

    microbatches = []
    for start_idx in range(0, true_B, microbatch_size):
        end_idx = min(start_idx + microbatch_size, true_B)
        interleaved_indices = []
        for i in range(start_idx, end_idx):
            interleaved_indices.extend([i * 2, i * 2 + 1])

        microbatch = {
            'input_ids': batch['input_ids'][interleaved_indices],
            'attention_mask': batch['attention_mask'][interleaved_indices],
        }
        if 'similarity_scores' in batch:
            microbatch['similarity_scores'] = batch['similarity_scores'][start_idx:end_idx]
        if 'labels' in batch and batch['labels'].size(0) == true_B:
            microbatch['labels'] = batch['labels'][start_idx:end_idx]
        microbatches.append(microbatch)
    return microbatches

def main():
    print("[1] parse args...", flush=True)
    parser = argparse.ArgumentParser(description='BGE-M3 Pruning Training')
    parser.add_argument('config', type=str, help='Path to configuration file')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')

    # NEW: explicit task selection
    parser.add_argument('--task', type=str, default='retrieval',
                        choices=['retrieval', 'sts', 'classification', 'pair_classification'],
                        help='Choose training/eval task type.')

    parser.add_argument('--dataset', type=str, default='sts',
                        help='HF dataset name (e.g., sts, msmarco, mteb/stsbenchmark-sts, ag_news, glue/mrpc)')

    # Optional field mapping for custom datasets
    parser.add_argument('--text_field', type=str, default='text')
    parser.add_argument('--label_field', type=str, default='label')
    parser.add_argument('--sentence1_field', type=str, default='sentence1')
    parser.add_argument('--sentence2_field', type=str, default='sentence2')
    parser.add_argument('--score_field', type=str, default='score')

    args = parser.parse_args()
    reproducibility.seed_all(args.seed)

    cfg = OmegaConf.load(args.config)
    print(f"🔥 BGE-M3 Production Pruning Training")
    print(f"📁 Config: {args.config}")
    print(f"📊 Dataset: {args.dataset}")
    print(f"🧩 Task: {args.task}")
    print(f"🎯 Pruning: head/layer/intermediate only")

    print("🚀 Loading BGE-M3 with L0 pruning...")
    model = ComposerBGEM3(cfg.model)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"📈 Total parameters: {total_params:,}")
    print(f"📈 Trainable parameters: {trainable_params:,}")

    print(f"📊 Loading {args.dataset} dataset from HuggingFace...")

    # Train loader
    train_dataloader = create_dataloader(
        dataset_name=args.dataset,
        split="train",
        batch_size=cfg.get('batch_size', 4),
        max_length=cfg.get('max_length', 256),
        task=args.task,
        text_field=args.text_field,
        label_field=args.label_field,
        sentence1_field=args.sentence1_field,
        sentence2_field=args.sentence2_field,
        score_field=args.score_field,
    )

    # Eval split heuristic
    # - If specific datasets provide 'validation' use that; else 'test'; else fallback 'validation' for sts
    eval_split = cfg.get('eval_split', None)
    if eval_split is None:
        eval_split = "validation" if args.task in ["sts", "pair_classification", "classification"] else "test"

    eval_dataloader = create_dataloader(
        dataset_name=args.dataset,
        split=eval_split,
        batch_size=cfg.get('batch_size', 4),
        max_length=cfg.get('max_length', 256),
        task=args.task,
        text_field=args.text_field,
        label_field=args.label_field,
        sentence1_field=args.sentence1_field,
        sentence2_field=args.sentence2_field,
        score_field=args.score_field,
    )

    train_dataspec = DataSpec(
        dataloader=train_dataloader,
        get_num_samples_in_batch=get_num_samples_in_batch,
        split_batch=split_batch
    )
    eval_dataspec = DataSpec(
        dataloader=eval_dataloader,
        get_num_samples_in_batch=get_num_samples_in_batch,
        split_batch=split_batch
    )

    optimizer = setup_optimizer(model, cfg.optimizer)
    callbacks = [PruningCallback(log_interval=500)]

    device = cfg.get('device', 'gpu')
    if device == 'cuda' or (device == 'auto' and torch.cuda.is_available()):
        device = 'gpu'
    elif device == 'cpu':
        device = 'cpu'
    else:
        device = 'gpu' if torch.cuda.is_available() else 'cpu'

    trainer = Trainer(
        model=model,
        train_dataloader=train_dataspec,
        eval_dataloader=eval_dataspec,
        max_duration=cfg.max_duration,
        eval_interval=cfg.eval_interval,
        save_interval=cfg.get('save_interval', '500ba'),
        save_folder=cfg.get('save_folder', 'experiments/checkpoints'),
        device=device,
        callbacks=callbacks,
        optimizers=optimizer,
        precision=cfg.get('precision', 'amp_bf16'),
    )

    print("🎯 Starting BGE-M3 pruning training...")
    trainer.fit()

    print(f"✅ Production training completed!")
    if hasattr(model, 'l0_module'):
        zs = model.l0_module()
        print(f"🎯 Pruning results:")
        for mask_name, mask_tensor in zs.items():
            sparsity = (mask_tensor == 0).float().mean().item()
            print(f"  {mask_name}: {sparsity:.1%} sparsity")

    # (Optional) Post-hoc HF export
    hf_save_path = cfg.get('save_folder', 'experiments/production') + '_hf'
    model.save_pruned_hf_model(hf_save_path)
    print(f"💾 Model ready for deployment")
if __name__ == "__main__":
    main()
