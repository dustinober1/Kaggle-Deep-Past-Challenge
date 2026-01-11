#!/usr/bin/env python3
"""
Optimization Experiments for Akkadian-English Translation

Runs systematic experiments testing different optimization settings:
- Optimizers: Adam, AdamW, Adafactor
- Label smoothing: 0.0, 0.1, 0.2
- Dropout rates: 0.1, 0.2, 0.3
- Gradient accumulation: 1, 2, 4

Results are logged to TensorBoard and summarized in a JSON report.

Usage:
    # Run all experiments (caution: takes time!)
    python optimization_experiments.py --all
    
    # Run specific experiment
    python optimization_experiments.py --optimizer adamw --label_smoothing 0.1
    
    # Smoke test all experiments
    python optimization_experiments.py --all --smoke_test
"""

import argparse
import os
import sys
import subprocess
import json
import time
from itertools import product
from datetime import datetime

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def run_experiment(config, base_args):
    """Run a single experiment with given config."""
    exp_name = "_".join([f"{k}={v}" for k, v in config.items()])
    print(f"\n{'='*60}")
    print(f"Experiment: {exp_name}")
    print(f"{'='*60}")
    
    output_dir = os.path.join(base_args.output_dir, exp_name)
    os.makedirs(output_dir, exist_ok=True)
    
    cmd = [
        sys.executable, os.path.join(SCRIPT_DIR, "train_custom.py"),
        "--epochs", str(base_args.epochs),
        "--batch_size", str(base_args.batch_size),
        "--output_dir", output_dir,
        "--tensorboard_dir", base_args.tensorboard_dir,
        "--experiment_name", exp_name,
        "--optimizer", config.get('optimizer', 'adam'),
        "--label_smoothing", str(config.get('label_smoothing', 0.0)),
        "--dropout", str(config.get('dropout', 0.1)),
        "--gradient_accumulation_steps", str(config.get('grad_accum', 1)),
        "--patience", str(base_args.patience),
    ]
    
    if base_args.use_alibi:
        cmd.append("--use_alibi")
    
    if base_args.smoke_test:
        cmd.append("--smoke_test")
    
    start_time = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - start_time
    
    # Load metrics if available
    metrics_path = os.path.join(output_dir, "metrics.json")
    best_val_loss = None
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            metrics = json.load(f)
            if metrics:
                best_val_loss = min(m['val_loss'] for m in metrics)
    
    return {
        'config': config,
        'exp_name': exp_name,
        'success': result.returncode == 0,
        'elapsed': elapsed,
        'best_val_loss': best_val_loss,
        'output_dir': output_dir
    }


def main():
    parser = argparse.ArgumentParser(description="Run optimization experiments")
    
    # Experiment selection
    parser.add_argument("--all", action="store_true", help="Run all experiments in grid")
    parser.add_argument("--optimizer_sweep", action="store_true", help="Sweep optimizers only")
    parser.add_argument("--smoothing_sweep", action="store_true", help="Sweep label smoothing only")
    parser.add_argument("--dropout_sweep", action="store_true", help="Sweep dropout rates only")
    parser.add_argument("--grad_accum_sweep", action="store_true", help="Sweep gradient accumulation")
    
    # Single experiment params
    parser.add_argument("--optimizer", type=str, default=None)
    parser.add_argument("--label_smoothing", type=float, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--grad_accum", type=int, default=None)
    
    # Base training params
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--output_dir", type=str, default="models/experiments")
    parser.add_argument("--tensorboard_dir", type=str, default="runs/experiments")
    parser.add_argument("--smoke_test", action="store_true")
    parser.add_argument("--use_alibi", action="store_true")
    parser.add_argument("--patience", type=int, default=3)
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.tensorboard_dir, exist_ok=True)
    
    # Define experiment grids
    experiments = []
    
    if args.all:
        # Full grid (large!)
        for opt, ls, do, ga in product(
            ['adam', 'adamw', 'adafactor'],
            [0.0, 0.1],
            [0.1, 0.2],
            [1, 2]
        ):
            experiments.append({'optimizer': opt, 'label_smoothing': ls, 'dropout': do, 'grad_accum': ga})
    
    elif args.optimizer_sweep:
        for opt in ['adam', 'adamw', 'adafactor']:
            experiments.append({'optimizer': opt, 'label_smoothing': 0.0, 'dropout': 0.1, 'grad_accum': 1})
    
    elif args.smoothing_sweep:
        for ls in [0.0, 0.05, 0.1, 0.15, 0.2]:
            experiments.append({'optimizer': 'adamw', 'label_smoothing': ls, 'dropout': 0.1, 'grad_accum': 1})
    
    elif args.dropout_sweep:
        for do in [0.1, 0.15, 0.2, 0.25, 0.3]:
            experiments.append({'optimizer': 'adamw', 'label_smoothing': 0.1, 'dropout': do, 'grad_accum': 1})
    
    elif args.grad_accum_sweep:
        for ga in [1, 2, 4, 8]:
            experiments.append({'optimizer': 'adamw', 'label_smoothing': 0.1, 'dropout': 0.1, 'grad_accum': ga})
    
    else:
        # Single experiment from args
        config = {}
        if args.optimizer: config['optimizer'] = args.optimizer
        else: config['optimizer'] = 'adam'
        if args.label_smoothing is not None: config['label_smoothing'] = args.label_smoothing
        else: config['label_smoothing'] = 0.0
        if args.dropout is not None: config['dropout'] = args.dropout
        else: config['dropout'] = 0.1
        if args.grad_accum is not None: config['grad_accum'] = args.grad_accum
        else: config['grad_accum'] = 1
        experiments.append(config)
    
    print(f"\nPlanned experiments: {len(experiments)}")
    for i, exp in enumerate(experiments, 1):
        print(f"  {i}. {exp}")
    
    # Run experiments
    results = []
    for exp in experiments:
        result = run_experiment(exp, args)
        results.append(result)
        print(f"Result: {'✓' if result['success'] else '✗'} | Val Loss: {result['best_val_loss']:.4f if result['best_val_loss'] else 'N/A'} | Time: {result['elapsed']:.1f}s")
    
    # Summary report
    print("\n" + "="*60)
    print("EXPERIMENT SUMMARY")
    print("="*60)
    
    successful = [r for r in results if r['success'] and r['best_val_loss'] is not None]
    if successful:
        best = min(successful, key=lambda r: r['best_val_loss'])
        print(f"\nBest configuration:")
        print(f"  Config: {best['config']}")
        print(f"  Val Loss: {best['best_val_loss']:.4f}")
        print(f"  Model: {best['output_dir']}")
        
        # Ranking
        print("\nAll results (sorted by val loss):")
        for i, r in enumerate(sorted(successful, key=lambda x: x['best_val_loss']), 1):
            print(f"  {i}. {r['exp_name']}: {r['best_val_loss']:.4f}")
    
    # Save report
    report = {
        'timestamp': datetime.now().isoformat(),
        'args': vars(args),
        'results': results,
        'best': best['config'] if successful else None
    }
    
    report_path = os.path.join(args.output_dir, "experiment_report.json")
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    print(f"\nReport saved to: {report_path}")
    print(f"TensorBoard: tensorboard --logdir {args.tensorboard_dir}")


if __name__ == "__main__":
    main()
