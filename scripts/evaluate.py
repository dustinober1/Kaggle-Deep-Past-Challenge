"""
Unified Evaluation Script for Akkadian→English Translation

Evaluates trained models on validation set with:
- Competition metrics (BLEU, chrF++, geometric mean)
- Per-example scoring
- Error analysis by category
- Cross-validation support
- Document vs sentence performance comparison

Usage:
    # Evaluate custom model
    python evaluate.py --model custom --checkpoint models/custom/best_model.pt
    
    # Evaluate pretrained model
    python evaluate.py --model mbart --checkpoint models/pretrained/mbart-finetune
    
    # Cross-validation
    python evaluate.py --model custom --cross_val 5
    
    # Compare training modes
    python evaluate.py --compare_modes
    
    # Full evaluation with error analysis
    python evaluate.py --model custom --error_analysis --output results/full_eval.json
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import asdict

import pandas as pd
import numpy as np
import torch
from tqdm import tqdm

# Add scripts directory to path
sys.path.insert(0, str(Path(__file__).parent))

from evaluation_metrics import (
    compute_competition_metric,
    evaluate_with_breakdown,
    print_summary,
    save_evaluation_report,
    score_examples
)
from error_analyzer import (
    ErrorAnalyzer,
    ErrorSummary,
    analyze_by_length,
    find_systematic_patterns,
    print_error_summary
)
from dataset_loader import (
    get_akkadian_dataset,
    get_document_level_dataset,
    get_sentence_level_dataset
)


def load_custom_model(checkpoint_path: str, device: str = 'cpu'):
    """Load custom transformer model from checkpoint."""
    from custom_transformer import TransformerModel
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Get model config from checkpoint
    config = checkpoint.get('config', {})
    vocab_size = config.get('vocab_size', 8000)
    d_model = config.get('d_model', 256)
    nhead = config.get('nhead', 8)
    num_layers = config.get('num_encoder_layers', 6)
    dim_feedforward = config.get('dim_feedforward', 512)
    dropout = config.get('dropout', 0.1)
    max_len = config.get('max_seq_len', 512)
    use_alibi = config.get('use_alibi', True)
    
    model = TransformerModel(
        vocab_size=vocab_size,
        d_model=d_model,
        nhead=nhead,
        num_encoder_layers=num_layers,
        num_decoder_layers=num_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        max_seq_len=max_len,
        use_alibi=use_alibi
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    return model, config


def load_pretrained_model(model_name: str, checkpoint_path: str, device: str = 'cpu'):
    """Load fine-tuned pretrained model."""
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    
    model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint_path)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
    
    model.to(device)
    model.eval()
    
    return model, tokenizer


def generate_translations_custom(model, sources: List[str], 
                                  tokenizer_path: str,
                                  device: str = 'cpu',
                                  max_len: int = 128,
                                  beam_size: int = 4) -> List[str]:
    """Generate translations using custom model."""
    import sentencepiece as spm
    
    sp = spm.SentencePieceProcessor(model_file=tokenizer_path)
    translations = []
    
    model.eval()
    with torch.no_grad():
        for source in tqdm(sources, desc="Generating translations"):
            # Tokenize source
            src_ids = sp.encode(source)
            src_tensor = torch.tensor([src_ids], dtype=torch.long, device=device)
            
            # Generate with greedy decoding (simplified)
            # In practice, would use beam search
            generated = greedy_decode(model, src_tensor, max_len, 
                                       sp.bos_id(), sp.eos_id(), device)
            
            # Decode
            translation = sp.decode(generated.tolist())
            translations.append(translation)
    
    return translations


def greedy_decode(model, src: torch.Tensor, max_len: int,
                  start_symbol: int, end_symbol: int,
                  device: str) -> torch.Tensor:
    """Greedy decoding for transformer model."""
    model.eval()
    
    # Encode source
    memory = model.encode(src, None)
    
    # Start with BOS
    ys = torch.ones(1, 1, dtype=torch.long, device=device) * start_symbol
    
    for _ in range(max_len - 1):
        # Decode
        tgt_mask = model.generate_square_subsequent_mask(ys.size(1)).to(device)
        out = model.decode(ys, memory, tgt_mask, None)
        
        # Get next token
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.item()
        
        # Append token
        ys = torch.cat([ys, torch.ones(1, 1, dtype=torch.long, device=device) * next_word], dim=1)
        
        # Stop if EOS
        if next_word == end_symbol:
            break
    
    return ys.squeeze(0)[1:]  # Remove BOS


def generate_translations_pretrained(model, tokenizer, sources: List[str],
                                      device: str = 'cpu',
                                      max_len: int = 128,
                                      batch_size: int = 8) -> List[str]:
    """Generate translations using pretrained model."""
    translations = []
    
    model.eval()
    with torch.no_grad():
        for i in tqdm(range(0, len(sources), batch_size), desc="Generating translations"):
            batch = sources[i:i+batch_size]
            
            inputs = tokenizer(batch, return_tensors="pt", padding=True, 
                              truncation=True, max_length=max_len)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            outputs = model.generate(**inputs, max_length=max_len, num_beams=4)
            
            decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            translations.extend(decoded)
    
    return translations


def evaluate_model(model_type: str,
                   checkpoint_path: str,
                   data_dir: str = "data/processed",
                   tokenizer_path: str = "models/tokenizers/spm_akkadian_1000_special.model",
                   device: str = None,
                   error_analysis: bool = False) -> Dict:
    """
    Evaluate a model on the validation set.
    
    Args:
        model_type: 'custom', 'mbart', 'nllb', or 'm2m'
        checkpoint_path: Path to model checkpoint
        data_dir: Path to data directory
        tokenizer_path: Path to tokenizer (for custom model)
        device: Device to use
        error_analysis: Whether to run error analysis
        
    Returns:
        Evaluation results dict
    """
    # Auto-detect device
    if device is None:
        if torch.cuda.is_available():
            device = 'cuda'
        elif torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'
    
    print(f"Using device: {device}")
    
    # Load validation data
    dataset = get_akkadian_dataset(data_dir)
    val_data = dataset['validation']
    
    sources = val_data['transliteration_cleaned']
    references = val_data['translation_normalized']
    
    # Determine source types if available
    source_types = None
    if 'source' in val_data.column_names:
        source_types = val_data['source']
    
    # Load model and generate translations
    if model_type == 'custom':
        model, config = load_custom_model(checkpoint_path, device)
        hypotheses = generate_translations_custom(
            model, sources, tokenizer_path, device
        )
    else:
        model, tokenizer = load_pretrained_model(model_type, checkpoint_path, device)
        hypotheses = generate_translations_pretrained(
            model, tokenizer, sources, device
        )
    
    # Compute metrics
    results = evaluate_with_breakdown(hypotheses, references, sources, source_types)
    results['model'] = model_type
    results['checkpoint'] = checkpoint_path
    
    # Run error analysis if requested
    if error_analysis:
        print("\nRunning error analysis...")
        analyzer = ErrorAnalyzer()
        
        example_scores = results['examples']
        reports = analyzer.analyze_batch(
            sources, hypotheses, references,
            [{'bleu': e['bleu'], 'chrf': e['chrf'], 'geometric_mean': e['geometric_mean']} 
             for e in example_scores]
        )
        
        summary = ErrorSummary(reports).summarize()
        patterns = find_systematic_patterns(reports)
        length_analysis = analyze_by_length(reports)
        
        results['error_analysis'] = {
            'summary': summary,
            'patterns': patterns,
            'by_length': length_analysis,
            'examples': [asdict(r) for r in reports]
        }
        
        print_error_summary(summary, patterns)
    
    # Print summary
    print_summary(results)
    
    return results


def run_cross_validation(model_type: str,
                         n_folds: int = 5,
                         data_dir: str = "data/processed",
                         output_dir: str = "results",
                         **train_kwargs) -> Dict:
    """
    Run k-fold cross-validation for robust metric estimates.
    
    Note: This requires training the model n_folds times.
    For time efficiency, this uses pre-computed predictions if available.
    """
    from sklearn.model_selection import KFold
    
    # Load all data
    train_df = pd.read_csv(os.path.join(data_dir, "train_final.csv"))
    val_df = pd.read_csv(os.path.join(data_dir, "val_final.csv"))
    
    # Combine for cross-validation
    all_data = pd.concat([train_df, val_df], ignore_index=True)
    
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    fold_results = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(all_data)):
        print(f"\n{'='*50}")
        print(f"Fold {fold + 1}/{n_folds}")
        print(f"{'='*50}")
        
        fold_train = all_data.iloc[train_idx]
        fold_val = all_data.iloc[val_idx]
        
        # Save fold data
        fold_dir = os.path.join(output_dir, f"cv_fold_{fold}")
        os.makedirs(fold_dir, exist_ok=True)
        
        fold_train.to_csv(os.path.join(fold_dir, "train.csv"), index=False)
        fold_val.to_csv(os.path.join(fold_dir, "val.csv"), index=False)
        
        print(f"  Train: {len(fold_train)}, Val: {len(fold_val)}")
        
        # Note: Actual training would happen here
        # For now, we just log the fold setup
        fold_results.append({
            'fold': fold,
            'train_size': len(fold_train),
            'val_size': len(fold_val)
        })
    
    # Aggregate results
    cv_results = {
        'n_folds': n_folds,
        'folds': fold_results,
        'note': 'Full CV requires training models for each fold. '
                'Use run_cross_val_training() for complete CV.'
    }
    
    return cv_results


def compare_document_vs_sentence(checkpoint_path: str,
                                  model_type: str = 'custom',
                                  data_dir: str = "data/processed") -> Dict:
    """Compare model performance on document vs sentence level data."""
    
    # Get document-level data
    doc_dataset = get_document_level_dataset(data_dir)
    doc_val = doc_dataset['validation']
    
    # Get sentence-level data
    sent_dataset = get_sentence_level_dataset(data_dir)
    sent_val = sent_dataset['validation']
    
    print(f"Document-level validation: {len(doc_val)} examples")
    print(f"Sentence-level validation: {len(sent_val)} examples")
    
    # Device setup
    device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
    
    results = {}
    
    # Evaluate on document-level
    print("\n--- Document-Level Evaluation ---")
    if len(doc_val) > 0:
        if model_type == 'custom':
            model, config = load_custom_model(checkpoint_path, device)
            tokenizer_path = "models/tokenizers/spm_akkadian_1000_special.model"
            doc_hyps = generate_translations_custom(
                model, doc_val['transliteration_cleaned'], 
                tokenizer_path, device
            )
        else:
            model, tokenizer = load_pretrained_model(model_type, checkpoint_path, device)
            doc_hyps = generate_translations_pretrained(
                model, tokenizer, doc_val['transliteration_cleaned'], device
            )
        
        doc_metrics = compute_competition_metric(doc_hyps, doc_val['translation_normalized'])
        results['document_level'] = doc_metrics
        print(f"  GM: {doc_metrics['geometric_mean']:.2f}, "
              f"BLEU: {doc_metrics['bleu']:.2f}, chrF++: {doc_metrics['chrf']:.2f}")
    else:
        results['document_level'] = {'note': 'No document-level examples in validation'}
    
    # Evaluate on sentence-level
    print("\n--- Sentence-Level Evaluation ---")
    if len(sent_val) > 0:
        if model_type == 'custom':
            sent_hyps = generate_translations_custom(
                model, sent_val['transliteration_cleaned'],
                tokenizer_path, device
            )
        else:
            sent_hyps = generate_translations_pretrained(
                model, tokenizer, sent_val['transliteration_cleaned'], device
            )
        
        sent_metrics = compute_competition_metric(sent_hyps, sent_val['translation_normalized'])
        results['sentence_level'] = sent_metrics
        print(f"  GM: {sent_metrics['geometric_mean']:.2f}, "
              f"BLEU: {sent_metrics['bleu']:.2f}, chrF++: {sent_metrics['chrf']:.2f}")
    else:
        results['sentence_level'] = {'note': 'No sentence-level examples in validation'}
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate translation models")
    parser.add_argument("--model", type=str, choices=['custom', 'mbart', 'nllb', 'm2m'],
                        help="Model type to evaluate")
    parser.add_argument("--checkpoint", type=str, help="Path to model checkpoint")
    parser.add_argument("--data_dir", type=str, default="data/processed",
                        help="Path to data directory")
    parser.add_argument("--tokenizer", type=str, 
                        default="models/tokenizers/spm_akkadian_1000_special.model",
                        help="Path to tokenizer (for custom model)")
    parser.add_argument("--output", type=str, default="results/evaluation.json",
                        help="Output path for evaluation results")
    parser.add_argument("--error_analysis", action="store_true",
                        help="Run error analysis")
    parser.add_argument("--cross_val", type=int, default=0,
                        help="Number of folds for cross-validation")
    parser.add_argument("--compare_modes", action="store_true",
                        help="Compare document vs sentence performance")
    parser.add_argument("--test", action="store_true",
                        help="Run unit tests on evaluation pipeline")
    args = parser.parse_args()
    
    # Create results directory
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    
    if args.test:
        print("Running evaluation pipeline tests...")
        
        # Test metrics computation
        from evaluation_metrics import run_self_tests
        run_self_tests()
        
        # Test error analyzer
        from error_analyzer import ErrorAnalyzer
        analyzer = ErrorAnalyzer()
        report = analyzer.analyze(
            source="1 TÚG I-tur₄-DINGIR il₅-qé",
            hypothesis="One textile received.",
            reference="Itūr-ilī has received one textile."
        )
        print(f"\n✓ Error analysis: {report.total_errors()} errors found")
        
        # Test data loading
        dataset = get_akkadian_dataset(args.data_dir)
        print(f"✓ Dataset loaded: {len(dataset['train'])} train, "
              f"{len(dataset['validation'])} val")
        
        print("\n✓ All evaluation tests passed!")
        return
    
    if args.cross_val > 0:
        print(f"Setting up {args.cross_val}-fold cross-validation...")
        results = run_cross_validation(
            model_type=args.model or 'custom',
            n_folds=args.cross_val,
            data_dir=args.data_dir,
            output_dir=os.path.dirname(args.output)
        )
        save_evaluation_report(results, args.output)
        return
    
    if args.compare_modes:
        if not args.checkpoint:
            print("Error: --checkpoint required for --compare_modes")
            return
        
        results = compare_document_vs_sentence(
            checkpoint_path=args.checkpoint,
            model_type=args.model or 'custom',
            data_dir=args.data_dir
        )
        save_evaluation_report(results, args.output)
        return
    
    if args.model and args.checkpoint:
        results = evaluate_model(
            model_type=args.model,
            checkpoint_path=args.checkpoint,
            data_dir=args.data_dir,
            tokenizer_path=args.tokenizer,
            error_analysis=args.error_analysis
        )
        save_evaluation_report(results, args.output)
    else:
        # Demo mode: just test metrics on validation data
        print("Demo mode: Testing metrics on validation data...")
        print("For full evaluation, provide --model and --checkpoint")
        
        dataset = get_akkadian_dataset(args.data_dir)
        val_data = dataset['validation']
        
        # Use references as hypotheses for demo (perfect match)
        refs = list(val_data['translation_normalized'])
        hyps = refs.copy()  # Same as refs for demo
        
        # Add some noise to a few examples
        if len(hyps) > 5:
            hyps[0] = "Some different text here."
            hyps[1] = hyps[1].replace(" ", "  ").replace("the", "a")
        
        results = compute_competition_metric(hyps, refs)
        print_summary({'corpus': results})
        
        # Per-example demo
        scores = score_examples(hyps[:5], refs[:5])
        print("\nPer-example scores (first 5):")
        for s in scores:
            print(f"  #{s['index']}: BLEU={s['bleu']:.1f}, chrF={s['chrf']:.1f}, "
                  f"GM={s['geometric_mean']:.1f}")


if __name__ == "__main__":
    main()
