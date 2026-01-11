"""
Evaluation Metrics for Akkadian→English Translation

Implements competition metrics:
- BLEU (SacreBLEU micro-averaged corpus BLEU)
- chrF++ (character F-score with word order)
- Geometric Mean (competition metric: sqrt(BLEU × chrF++))

Usage:
    python evaluation_metrics.py --test  # Run self-tests
    
    # From Python:
    from evaluation_metrics import compute_competition_metric
    result = compute_competition_metric(hypotheses, references)
"""

import sacrebleu
from sacrebleu.metrics import BLEU, CHRF
import numpy as np
from typing import List, Dict, Tuple, Optional, Union
import json
import argparse


def compute_bleu(hypotheses: List[str], references: List[str]) -> Dict:
    """
    Compute corpus BLEU score using SacreBLEU.
    
    Args:
        hypotheses: List of predicted translations
        references: List of reference translations
        
    Returns:
        Dict with 'score' (0-100), 'precisions', 'bp' (brevity penalty), etc.
    """
    # SacreBLEU expects references as list of lists (for multiple references per example)
    refs = [[ref] for ref in references]
    
    bleu = BLEU()
    result = bleu.corpus_score(hypotheses, [[r[0] for r in refs]])
    
    return {
        'score': result.score,
        'precisions': result.precisions,
        'bp': result.bp,
        'sys_len': result.sys_len,
        'ref_len': result.ref_len,
        'signature': str(result)
    }


def compute_chrf(hypotheses: List[str], references: List[str], 
                 word_order: int = 2) -> Dict:
    """
    Compute chrF++ score using SacreBLEU.
    
    chrF++ uses character n-grams plus word unigrams/bigrams for better
    evaluation of morphologically rich languages.
    
    Args:
        hypotheses: List of predicted translations
        references: List of reference translations
        word_order: Word n-gram order (2 for chrF++, 0 for chrF)
        
    Returns:
        Dict with 'score' (0-100), plus detailed metrics
    """
    refs = [[ref] for ref in references]
    
    chrf = CHRF(word_order=word_order)
    result = chrf.corpus_score(hypotheses, [[r[0] for r in refs]])
    
    return {
        'score': result.score,
        'signature': str(result)
    }


def compute_geometric_mean(bleu_score: float, chrf_score: float) -> float:
    """
    Compute geometric mean of BLEU and chrF++ (competition metric).
    
    Args:
        bleu_score: BLEU score (0-100)
        chrf_score: chrF++ score (0-100)
        
    Returns:
        Geometric mean score (0-100)
    """
    if bleu_score <= 0 or chrf_score <= 0:
        return 0.0
    return np.sqrt(bleu_score * chrf_score)


def compute_competition_metric(hypotheses: List[str], 
                                references: List[str]) -> Dict:
    """
    Compute the full competition metric suite.
    
    Args:
        hypotheses: List of predicted translations
        references: List of reference translations
        
    Returns:
        Dict with BLEU, chrF++, geometric mean, and detailed stats
    """
    bleu_result = compute_bleu(hypotheses, references)
    chrf_result = compute_chrf(hypotheses, references)
    geo_mean = compute_geometric_mean(bleu_result['score'], chrf_result['score'])
    
    return {
        'geometric_mean': geo_mean,
        'bleu': bleu_result['score'],
        'chrf': chrf_result['score'],
        'bleu_details': bleu_result,
        'chrf_details': chrf_result,
        'num_examples': len(hypotheses)
    }


def compute_sentence_bleu(hypothesis: str, reference: str) -> float:
    """
    Compute sentence-level BLEU score.
    
    Note: Sentence-level BLEU is less reliable than corpus BLEU.
    Use smoothing for better estimates on short sentences.
    
    Args:
        hypothesis: Single predicted translation
        reference: Single reference translation
        
    Returns:
        Sentence BLEU score (0-100)
    """
    bleu = BLEU(smooth_method='exp', smooth_value=0.0)
    result = bleu.sentence_score(hypothesis, [reference])
    return result.score


def compute_sentence_chrf(hypothesis: str, reference: str) -> float:
    """
    Compute sentence-level chrF++ score.
    
    Args:
        hypothesis: Single predicted translation
        reference: Single reference translation
        
    Returns:
        Sentence chrF++ score (0-100)
    """
    chrf = CHRF(word_order=2)
    result = chrf.sentence_score(hypothesis, [reference])
    return result.score


def score_examples(hypotheses: List[str], 
                   references: List[str],
                   sources: Optional[List[str]] = None,
                   ids: Optional[List[str]] = None) -> List[Dict]:
    """
    Compute per-example scores.
    
    Args:
        hypotheses: List of predicted translations
        references: List of reference translations
        sources: Optional list of source (Akkadian) texts
        ids: Optional list of example IDs
        
    Returns:
        List of dicts, each containing scores for one example
    """
    results = []
    
    for i, (hyp, ref) in enumerate(zip(hypotheses, references)):
        example_result = {
            'index': i,
            'bleu': compute_sentence_bleu(hyp, ref),
            'chrf': compute_sentence_chrf(hyp, ref),
            'hypothesis': hyp,
            'reference': ref,
            'hypothesis_len': len(hyp.split()),
            'reference_len': len(ref.split())
        }
        
        # Add geometric mean
        example_result['geometric_mean'] = compute_geometric_mean(
            example_result['bleu'], example_result['chrf']
        )
        
        # Add optional fields
        if sources is not None:
            example_result['source'] = sources[i]
            example_result['source_len'] = len(sources[i].split())
        
        if ids is not None:
            example_result['id'] = ids[i]
            
        results.append(example_result)
    
    return results


def evaluate_with_breakdown(hypotheses: List[str],
                            references: List[str],
                            sources: Optional[List[str]] = None,
                            source_types: Optional[List[str]] = None) -> Dict:
    """
    Full evaluation with corpus and per-example breakdown.
    
    Args:
        hypotheses: Predicted translations
        references: Reference translations
        sources: Optional source texts
        source_types: Optional source type labels ('document' or 'sentence')
        
    Returns:
        Comprehensive evaluation dict
    """
    # Corpus-level metrics
    corpus_metrics = compute_competition_metric(hypotheses, references)
    
    # Per-example scores
    example_scores = score_examples(hypotheses, references, sources)
    
    # Score distribution stats
    bleu_scores = [e['bleu'] for e in example_scores]
    chrf_scores = [e['chrf'] for e in example_scores]
    geo_scores = [e['geometric_mean'] for e in example_scores]
    
    result = {
        'corpus': corpus_metrics,
        'distribution': {
            'bleu': {
                'mean': np.mean(bleu_scores),
                'std': np.std(bleu_scores),
                'min': np.min(bleu_scores),
                'max': np.max(bleu_scores),
                'median': np.median(bleu_scores)
            },
            'chrf': {
                'mean': np.mean(chrf_scores),
                'std': np.std(chrf_scores),
                'min': np.min(chrf_scores),
                'max': np.max(chrf_scores),
                'median': np.median(chrf_scores)
            },
            'geometric_mean': {
                'mean': np.mean(geo_scores),
                'std': np.std(geo_scores),
                'min': np.min(geo_scores),
                'max': np.max(geo_scores),
                'median': np.median(geo_scores)
            }
        },
        'examples': example_scores
    }
    
    # Add breakdown by source type if provided
    if source_types is not None:
        type_metrics = {}
        for src_type in set(source_types):
            indices = [i for i, t in enumerate(source_types) if t == src_type]
            type_hyps = [hypotheses[i] for i in indices]
            type_refs = [references[i] for i in indices]
            type_metrics[src_type] = compute_competition_metric(type_hyps, type_refs)
        result['by_source_type'] = type_metrics
    
    return result


def save_evaluation_report(results: Dict, output_path: str):
    """Save evaluation results to JSON file."""
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=float)
    print(f"Saved evaluation report to {output_path}")


def print_summary(results: Dict):
    """Print a summary of evaluation results."""
    corpus = results['corpus']
    
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"  Geometric Mean (Competition): {corpus['geometric_mean']:.2f}")
    print(f"  BLEU:                         {corpus['bleu']:.2f}")
    print(f"  chrF++:                       {corpus['chrf']:.2f}")
    print(f"  Num Examples:                 {corpus['num_examples']}")
    print("=" * 60)
    
    if 'distribution' in results:
        dist = results['distribution']
        print("\nScore Distribution (per-example):")
        print(f"  BLEU:   mean={dist['bleu']['mean']:.2f}, std={dist['bleu']['std']:.2f}")
        print(f"  chrF++: mean={dist['chrf']['mean']:.2f}, std={dist['chrf']['std']:.2f}")
    
    if 'by_source_type' in results:
        print("\nBy Source Type:")
        for src_type, metrics in results['by_source_type'].items():
            print(f"  {src_type}: GM={metrics['geometric_mean']:.2f}, "
                  f"BLEU={metrics['bleu']:.2f}, chrF++={metrics['chrf']:.2f}")
    print()


def run_self_tests():
    """Run self-tests to verify metric implementations."""
    print("Running evaluation metric tests...")
    
    # Test cases
    test_cases = [
        {
            'name': 'Exact match',
            'hyps': ['The cat sat on the mat.'],
            'refs': ['The cat sat on the mat.'],
            'expected_bleu_approx': 100.0
        },
        {
            'name': 'Partial match',
            'hyps': ['The cat sat on the rug.'],
            'refs': ['The cat sat on the mat.'],
            'expected_bleu_approx': 60.0  # Rough approximation
        },
        {
            'name': 'No match',
            'hyps': ['Hello world'],
            'refs': ['Goodbye universe'],
            'expected_bleu_approx': 0.0
        },
        {
            'name': 'Multiple examples',
            'hyps': [
                'Itūr-ilī received one textile.',
                'He sent silver to the city.',
                'The merchant went to Kanesh.'
            ],
            'refs': [
                'Itūr-ilī has received one textile.',
                'He sent silver to the city.',
                'The merchant traveled to Kanesh.'
            ],
            'expected_bleu_approx': 50.0  # Mixed quality
        }
    ]
    
    all_passed = True
    for tc in test_cases:
        result = compute_competition_metric(tc['hyps'], tc['refs'])
        
        # Check BLEU is in expected range
        bleu_ok = (tc['expected_bleu_approx'] == 0 and result['bleu'] < 5) or \
                  (tc['expected_bleu_approx'] == 100 and result['bleu'] > 95) or \
                  (5 < tc['expected_bleu_approx'] < 95 and 
                   abs(result['bleu'] - tc['expected_bleu_approx']) < 40)
        
        status = "✓" if bleu_ok else "✗"
        print(f"  {status} {tc['name']}: BLEU={result['bleu']:.2f}, "
              f"chrF++={result['chrf']:.2f}, GM={result['geometric_mean']:.2f}")
        
        if not bleu_ok:
            all_passed = False
    
    # Test per-example scoring
    print("\n  Testing per-example scoring...")
    examples = score_examples(test_cases[3]['hyps'], test_cases[3]['refs'])
    if len(examples) == 3 and all('bleu' in e and 'chrf' in e for e in examples):
        print("  ✓ Per-example scoring works")
    else:
        print("  ✗ Per-example scoring failed")
        all_passed = False
    
    # Test geometric mean edge cases
    print("\n  Testing geometric mean edge cases...")
    assert compute_geometric_mean(0, 50) == 0.0, "Zero BLEU should give 0"
    assert compute_geometric_mean(50, 0) == 0.0, "Zero chrF should give 0"
    assert abs(compute_geometric_mean(100, 100) - 100) < 0.01, "Perfect scores"
    assert abs(compute_geometric_mean(25, 100) - 50) < 0.01, "sqrt(25*100)=50"
    print("  ✓ Geometric mean edge cases pass")
    
    print("\n" + ("All tests passed!" if all_passed else "Some tests failed!"))
    return all_passed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluation metrics for translation")
    parser.add_argument("--test", action="store_true", help="Run self-tests")
    args = parser.parse_args()
    
    if args.test:
        run_self_tests()
    else:
        # Demo usage
        print("Evaluation Metrics Demo")
        print("-" * 40)
        
        hyps = [
            "Itūr-ilī has received one textile of ordinary quality.",
            "He did not give you the textile.",
            "Send the silver to Kanesh."
        ]
        refs = [
            "Itūr-ilī has received one textile of ordinary quality.",
            "... he did not give you a textile.",
            "Send the silver to the City of Kanesh."
        ]
        
        result = compute_competition_metric(hyps, refs)
        print_summary({'corpus': result})
        
        print("Per-example scores:")
        for ex in score_examples(hyps, refs):
            print(f"  #{ex['index']}: BLEU={ex['bleu']:.1f}, chrF={ex['chrf']:.1f}")
