#!/usr/bin/env python3
"""
Multi-Level Training Script for Akkadian-English Translation

Implements various training strategies:
1. Document-level: Train on full document context first
2. Sentence-level: Fine-tune on sentence pairs (matches test format)
3. Curriculum: Train easy→hard or by length
4. Mixed: Combine document and sentence examples

Usage:
    # Document-level training first
    python multi_level_trainer.py --mode document --epochs 10
    
    # Then fine-tune on sentences
    python multi_level_trainer.py --mode sentence --epochs 5 --resume_from models/custom_transformer/checkpoint.pt
    
    # Curriculum learning
    python multi_level_trainer.py --mode curriculum --epochs 10 --sort_by length
    
    # Mixed training
    python multi_level_trainer.py --mode mixed --doc_ratio 0.3 --epochs 10
"""

import argparse
import os
import sys
import subprocess

# Add scripts directory to path for imports
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from dataset_loader import (
    get_akkadian_dataset,
    get_document_level_dataset,
    get_sentence_level_dataset,
    get_curriculum_dataset,
    get_mixed_dataset
)


def run_training(args, dataset_fn, dataset_name):
    """Run training with the specified dataset."""
    print(f"\n{'='*60}")
    print(f"Training Mode: {dataset_name}")
    print(f"{'='*60}")
    
    # Load dataset based on mode
    if args.mode == 'curriculum':
        dataset = dataset_fn(sort_by=args.sort_by, ascending=args.ascending)
    elif args.mode == 'mixed':
        dataset = dataset_fn(doc_ratio=args.doc_ratio)
    else:
        dataset = dataset_fn()
    
    print(f"Train examples: {len(dataset['train'])}")
    print(f"Validation examples: {len(dataset['validation'])}")
    
    # Build command for train_custom.py
    cmd = [
        sys.executable, os.path.join(SCRIPT_DIR, "train_custom.py"),
        "--epochs", str(args.epochs),
        "--batch_size", str(args.batch_size),
        "--output_dir", args.output_dir,
        "--tensorboard_dir", args.tensorboard_dir,
        "--experiment_name", f"{dataset_name}_{args.experiment_suffix}",
        "--optimizer", args.optimizer,
        "--label_smoothing", str(args.label_smoothing),
        "--dropout", str(args.dropout),
        "--gradient_accumulation_steps", str(args.gradient_accumulation_steps),
        "--patience", str(args.patience),
    ]
    
    if args.use_alibi:
        cmd.append("--use_alibi")
    
    if args.resume_from:
        cmd.extend(["--resume_from", args.resume_from])
    
    if args.smoke_test:
        cmd.append("--smoke_test")
    
    print(f"\nRunning: {' '.join(cmd)}")
    
    # Set environment variable for dataset mode
    env = os.environ.copy()
    env['TRAINING_MODE'] = args.mode
    env['DOC_RATIO'] = str(args.doc_ratio) if args.mode == 'mixed' else '0.5'
    env['CURRICULUM_SORT'] = args.sort_by if args.mode == 'curriculum' else 'length'
    
    # Run training
    result = subprocess.run(cmd, env=env)
    return result.returncode


def main():
    parser = argparse.ArgumentParser(description="Multi-level training for Akkadian translation")
    
    # Training mode
    parser.add_argument("--mode", type=str, required=True,
                        choices=["document", "sentence", "curriculum", "mixed", "full_pipeline"],
                        help="Training mode")
    
    # Dataset options
    parser.add_argument("--doc_ratio", type=float, default=0.5,
                        help="Ratio of document examples in mixed mode")
    parser.add_argument("--sort_by", type=str, default="length", choices=["length", "ratio"],
                        help="Difficulty metric for curriculum learning")
    parser.add_argument("--ascending", action="store_true", default=True,
                        help="Sort easy→hard (True) or hard→easy (False)")
    
    # Training parameters
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--output_dir", type=str, default="models/custom_transformer")
    parser.add_argument("--tensorboard_dir", type=str, default="runs")
    parser.add_argument("--experiment_suffix", type=str, default="exp")
    parser.add_argument("--resume_from", type=str, default=None)
    parser.add_argument("--smoke_test", action="store_true")
    
    # Optimization
    parser.add_argument("--optimizer", type=str, default="adam", choices=["adam", "adamw", "adafactor"])
    parser.add_argument("--label_smoothing", type=float, default=0.0)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--use_alibi", action="store_true")
    
    args = parser.parse_args()
    
    # Map mode to dataset function
    mode_map = {
        'document': (get_document_level_dataset, 'document_level'),
        'sentence': (get_sentence_level_dataset, 'sentence_level'),
        'curriculum': (get_curriculum_dataset, f'curriculum_{args.sort_by}'),
        'mixed': (get_mixed_dataset, f'mixed_{int(args.doc_ratio*100)}'),
    }
    
    if args.mode == 'full_pipeline':
        # Full pipeline: document → sentence
        print("\n" + "="*60)
        print("FULL MULTI-LEVEL PIPELINE")
        print("Stage 1: Document-level training")
        print("Stage 2: Sentence-level fine-tuning")
        print("="*60)
        
        # Stage 1: Document level
        args.mode = 'document'
        doc_output_dir = os.path.join(args.output_dir, "stage1_document")
        os.makedirs(doc_output_dir, exist_ok=True)
        
        stage1_args = argparse.Namespace(**vars(args))
        stage1_args.output_dir = doc_output_dir
        stage1_args.experiment_suffix = "stage1_doc"
        
        dataset_fn, dataset_name = mode_map['document']
        ret = run_training(stage1_args, dataset_fn, dataset_name)
        
        if ret != 0:
            print("Stage 1 failed!")
            return 1
        
        # Stage 2: Sentence level (fine-tune from stage 1)
        args.mode = 'sentence'
        sent_output_dir = os.path.join(args.output_dir, "stage2_sentence")
        os.makedirs(sent_output_dir, exist_ok=True)
        
        stage2_args = argparse.Namespace(**vars(args))
        stage2_args.output_dir = sent_output_dir
        stage2_args.experiment_suffix = "stage2_sent"
        stage2_args.resume_from = os.path.join(doc_output_dir, "best_model.pt")
        stage2_args.epochs = args.epochs // 2  # Fewer epochs for fine-tuning
        
        dataset_fn, dataset_name = mode_map['sentence']
        ret = run_training(stage2_args, dataset_fn, dataset_name)
        
        print("\n" + "="*60)
        print("FULL PIPELINE COMPLETE")
        print(f"Stage 1 model: {doc_output_dir}")
        print(f"Stage 2 model: {sent_output_dir}")
        print("="*60)
        
        return ret
    
    else:
        # Single mode training
        dataset_fn, dataset_name = mode_map[args.mode]
        return run_training(args, dataset_fn, dataset_name)


if __name__ == "__main__":
    sys.exit(main())
