"""
Error Analyzer for Akkadian→English Translation

Categorizes and analyzes translation errors to identify:
- Named entity errors (personal names, geographic names)
- Number/measurement errors (shekels, talents, textiles)
- Gap handling errors (<gap>, <big_gap> markers)
- Word order errors
- Length-based patterns
- Systematic issues

Usage:
    python error_analyzer.py --input results/predictions.json --output results/error_analysis.json
"""

import re
import json
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple, Optional, Set
import os


@dataclass
class ErrorReport:
    """Report for a single example's errors."""
    index: int
    source: str
    hypothesis: str
    reference: str
    source_len: int
    ref_len: int
    hyp_len: int
    
    # Error categories
    named_entity_errors: List[Dict]
    number_errors: List[Dict]
    gap_errors: List[Dict]
    word_order_score: float
    
    # Scores
    bleu: float = 0.0
    chrf: float = 0.0
    geometric_mean: float = 0.0
    
    def total_errors(self) -> int:
        return (len(self.named_entity_errors) + 
                len(self.number_errors) + 
                len(self.gap_errors))
    
    def has_errors(self) -> bool:
        return self.total_errors() > 0 or self.word_order_score < 0.8


class ErrorAnalyzer:
    """Analyzes translation errors by category."""
    
    def __init__(self, 
                 pn_path: str = "models/tokenizers/personal_names.txt",
                 gn_path: str = "models/tokenizers/geographic_names.txt",
                 entities_path: str = "data/processed/named_entities.json"):
        """
        Initialize error analyzer with name lists.
        
        Args:
            pn_path: Path to personal names file
            gn_path: Path to geographic names file  
            entities_path: Path to named entities JSON
        """
        self.personal_names = self._load_names(pn_path)
        self.geographic_names = self._load_names(gn_path)
        self.entities = self._load_entities(entities_path)
        
        # Measurement units common in Old Assyrian
        self.measurement_units = {
            'shekel', 'shekels', 'talent', 'talents', 'mina', 'minas',
            'textile', 'textiles', 'donkey', 'donkeys', 'tin', 'silver',
            'gold', 'copper', 'year', 'years', 'month', 'months',
            'day', 'days', 'cubit', 'cubits'
        }
        
        # Number patterns
        self.number_pattern = re.compile(
            r'\b(\d+(?:/\d+)?(?:\s*(?:and|plus))?\s*\d*(?:/\d+)?)\b'
            r'|'
            r'\b(one|two|three|four|five|six|seven|eight|nine|ten|'
            r'eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|'
            r'eighteen|nineteen|twenty|thirty|forty|fifty|sixty|'
            r'seventy|eighty|ninety|hundred|thousand)\b',
            re.IGNORECASE
        )
    
    def _load_names(self, path: str) -> Set[str]:
        """Load names from text file."""
        names = set()
        if os.path.exists(path):
            with open(path, 'r') as f:
                for line in f:
                    name = line.strip()
                    if name:
                        names.add(name.lower())
                        # Also add normalized versions
                        names.add(name.lower().replace('-', ''))
                        names.add(name.lower().replace('ī', 'i').replace('ū', 'u'))
        return names
    
    def _load_entities(self, path: str) -> Dict:
        """Load named entities JSON."""
        if os.path.exists(path):
            with open(path, 'r') as f:
                return json.load(f)
        return {'personal_names': {}, 'geographic_names': {}}
    
    def extract_proper_nouns(self, text: str) -> List[str]:
        """Extract potential proper nouns from text."""
        # Words starting with capital letter (not at sentence start)
        words = text.split()
        proper_nouns = []
        
        for i, word in enumerate(words):
            # Skip first word and common sentence starters
            if i == 0:
                continue
            # Check if starts with capital
            if word and word[0].isupper():
                # Clean punctuation
                clean = re.sub(r'[^\w-]', '', word)
                if clean and len(clean) > 1:
                    proper_nouns.append(clean)
        
        return proper_nouns
    
    def extract_numbers(self, text: str) -> List[str]:
        """Extract numbers and measurements from text."""
        matches = self.number_pattern.findall(text)
        # Flatten tuples from regex groups
        numbers = []
        for match in matches:
            for group in match:
                if group:
                    numbers.append(group)
        return numbers
    
    def extract_measurements(self, text: str) -> List[Tuple[str, str]]:
        """Extract number + unit pairs."""
        measurements = []
        words = text.lower().split()
        
        for i, word in enumerate(words):
            # Check if this word is a unit
            clean_word = re.sub(r'[^\w]', '', word)
            if clean_word in self.measurement_units:
                # Look for preceding number
                if i > 0:
                    prev = words[i-1]
                    if self.number_pattern.match(prev) or prev.isdigit():
                        measurements.append((prev, clean_word))
        
        return measurements
    
    def check_gap_handling(self, source: str, hypothesis: str, 
                           reference: str) -> List[Dict]:
        """Check for gap marker errors."""
        errors = []
        
        # Count gap markers
        source_gaps = source.count('<gap>') + source.count('…')
        source_big_gaps = source.count('<big_gap>')
        
        # Check if reference mentions gaps/breaks
        ref_has_gap = any(m in reference.lower() for m in 
                         ['...', '…', '[gap]', '[broken]', '[damaged]', '[?]'])
        
        # Check hypothesis handling
        hyp_has_gap = any(m in hypothesis.lower() for m in 
                        ['...', '…', '[gap]', '[broken]', '[damaged]', '[?]'])
        
        # Error: source has gaps but translation doesn't acknowledge
        if source_gaps > 0 or source_big_gaps > 0:
            if ref_has_gap and not hyp_has_gap:
                errors.append({
                    'type': 'missing_gap_marker',
                    'source_gaps': source_gaps,
                    'source_big_gaps': source_big_gaps,
                    'description': 'Source has gaps but hypothesis lacks gap markers'
                })
        
        # Error: hypothesis has gaps but shouldn't
        if not source_gaps and not source_big_gaps:
            if hyp_has_gap and not ref_has_gap:
                errors.append({
                    'type': 'spurious_gap_marker',
                    'description': 'Hypothesis has gap markers but source is complete'
                })
        
        return errors
    
    def check_named_entities(self, hypothesis: str, 
                             reference: str) -> List[Dict]:
        """Check for named entity errors."""
        errors = []
        
        # Extract proper nouns from both
        hyp_names = set(self.extract_proper_nouns(hypothesis))
        ref_names = set(self.extract_proper_nouns(reference))
        
        # Names in reference but not in hypothesis
        missing = ref_names - hyp_names
        for name in missing:
            # Check if it's a known name with variant spelling
            name_lower = name.lower()
            found_variant = False
            
            for hyp_name in hyp_names:
                hyp_lower = hyp_name.lower()
                # Check for close match (edit distance or substring)
                if (name_lower in hyp_lower or hyp_lower in name_lower or
                    self._similar_names(name_lower, hyp_lower)):
                    found_variant = True
                    break
            
            if not found_variant:
                errors.append({
                    'type': 'missing_name',
                    'name': name,
                    'is_known_pn': name_lower in self.personal_names,
                    'is_known_gn': name_lower in self.geographic_names
                })
        
        # Names in hypothesis but not in reference (hallucinated)
        extra = hyp_names - ref_names
        for name in extra:
            name_lower = name.lower()
            errors.append({
                'type': 'extra_name',
                'name': name,
                'is_known_pn': name_lower in self.personal_names,
                'is_known_gn': name_lower in self.geographic_names
            })
        
        return errors
    
    def _similar_names(self, name1: str, name2: str) -> bool:
        """Check if two names are variants of each other."""
        # Normalize common Akkadian transliteration variants
        def normalize(n):
            return (n.replace('ī', 'i').replace('ū', 'u')
                     .replace('š', 's').replace('ṭ', 't')
                     .replace('ṣ', 's').replace('-', ''))
        
        n1 = normalize(name1)
        n2 = normalize(name2)
        
        return n1 == n2 or n1 in n2 or n2 in n1
    
    def check_numbers(self, hypothesis: str, reference: str) -> List[Dict]:
        """Check for number/measurement errors."""
        errors = []
        
        # Extract measurements
        hyp_measurements = set(self.extract_measurements(hypothesis))
        ref_measurements = set(self.extract_measurements(reference))
        
        # Missing measurements
        for num, unit in (ref_measurements - hyp_measurements):
            errors.append({
                'type': 'missing_measurement',
                'number': num,
                'unit': unit,
                'description': f'Missing "{num} {unit}" from reference'
            })
        
        # Extra measurements
        for num, unit in (hyp_measurements - ref_measurements):
            errors.append({
                'type': 'extra_measurement',
                'number': num,
                'unit': unit,
                'description': f'Extra "{num} {unit}" not in reference'
            })
        
        # Also check standalone numbers
        hyp_nums = set(self.extract_numbers(hypothesis))
        ref_nums = set(self.extract_numbers(reference))
        
        # Numbers in reference but not hypothesis (and not already counted)
        for num in (ref_nums - hyp_nums):
            if not any(num in str(m) for m in ref_measurements):
                errors.append({
                    'type': 'missing_number',
                    'number': num
                })
        
        return errors
    
    def compute_word_order_score(self, hypothesis: str, 
                                  reference: str) -> float:
        """
        Compute a word order similarity score using n-gram overlap.
        
        Returns a score between 0 and 1, where 1 means identical word order.
        """
        def get_ngrams(text: str, n: int) -> List[Tuple]:
            words = text.lower().split()
            return [tuple(words[i:i+n]) for i in range(len(words) - n + 1)]
        
        # Use bigrams and trigrams
        hyp_bigrams = set(get_ngrams(hypothesis, 2))
        ref_bigrams = set(get_ngrams(reference, 2))
        
        hyp_trigrams = set(get_ngrams(hypothesis, 3))
        ref_trigrams = set(get_ngrams(reference, 3))
        
        # Compute overlap
        if not ref_bigrams or not ref_trigrams:
            return 1.0  # Empty reference
        
        bigram_overlap = len(hyp_bigrams & ref_bigrams) / len(ref_bigrams) if ref_bigrams else 1.0
        trigram_overlap = len(hyp_trigrams & ref_trigrams) / len(ref_trigrams) if ref_trigrams else 1.0
        
        # Weighted average
        return 0.6 * bigram_overlap + 0.4 * trigram_overlap
    
    def analyze(self, source: str, hypothesis: str, reference: str,
                index: int = 0, scores: Optional[Dict] = None) -> ErrorReport:
        """
        Analyze a single translation for errors.
        
        Args:
            source: Source (Akkadian) text
            hypothesis: Model prediction
            reference: Gold reference translation
            index: Example index
            scores: Optional dict with 'bleu', 'chrf', 'geometric_mean'
            
        Returns:
            ErrorReport with categorized errors
        """
        report = ErrorReport(
            index=index,
            source=source,
            hypothesis=hypothesis,
            reference=reference,
            source_len=len(source.split()),
            ref_len=len(reference.split()),
            hyp_len=len(hypothesis.split()),
            named_entity_errors=self.check_named_entities(hypothesis, reference),
            number_errors=self.check_numbers(hypothesis, reference),
            gap_errors=self.check_gap_handling(source, hypothesis, reference),
            word_order_score=self.compute_word_order_score(hypothesis, reference)
        )
        
        if scores:
            report.bleu = scores.get('bleu', 0.0)
            report.chrf = scores.get('chrf', 0.0)
            report.geometric_mean = scores.get('geometric_mean', 0.0)
        
        return report
    
    def analyze_batch(self, sources: List[str], hypotheses: List[str],
                      references: List[str], 
                      scores: Optional[List[Dict]] = None) -> List[ErrorReport]:
        """Analyze a batch of translations."""
        reports = []
        for i, (src, hyp, ref) in enumerate(zip(sources, hypotheses, references)):
            score = scores[i] if scores else None
            reports.append(self.analyze(src, hyp, ref, i, score))
        return reports


class ErrorSummary:
    """Aggregate summary of errors across examples."""
    
    def __init__(self, reports: List[ErrorReport]):
        self.reports = reports
        self.num_examples = len(reports)
    
    def summarize(self) -> Dict:
        """Generate comprehensive error summary."""
        # Count error types
        error_counts = Counter()
        examples_with_errors = 0
        
        named_entity_errors = []
        number_errors = []
        gap_errors = []
        word_order_scores = []
        
        for report in self.reports:
            if report.has_errors():
                examples_with_errors += 1
            
            for err in report.named_entity_errors:
                error_counts[err['type']] += 1
                named_entity_errors.append(err)
            
            for err in report.number_errors:
                error_counts[err['type']] += 1
                number_errors.append(err)
            
            for err in report.gap_errors:
                error_counts[err['type']] += 1
                gap_errors.append(err)
            
            word_order_scores.append(report.word_order_score)
        
        return {
            'total_examples': self.num_examples,
            'examples_with_errors': examples_with_errors,
            'error_rate': examples_with_errors / self.num_examples if self.num_examples > 0 else 0,
            'error_counts': dict(error_counts),
            'total_errors': sum(error_counts.values()),
            'word_order': {
                'mean': np.mean(word_order_scores),
                'std': np.std(word_order_scores),
                'min': np.min(word_order_scores),
                'median': np.median(word_order_scores)
            },
            'by_category': {
                'named_entity': len(named_entity_errors),
                'number': len(number_errors),
                'gap': len(gap_errors)
            }
        }


def analyze_by_length(reports: List[ErrorReport], 
                      bins: List[int] = [30, 60, 100, 200]) -> Dict:
    """Analyze errors by source text length."""
    results = {}
    
    # Create length bins
    bin_labels = ['very_short', 'short', 'medium', 'long', 'very_long']
    bins = [0] + bins + [float('inf')]
    
    for i, label in enumerate(bin_labels):
        lo, hi = bins[i], bins[i+1]
        bin_reports = [r for r in reports if lo <= r.source_len < hi]
        
        if bin_reports:
            summary = ErrorSummary(bin_reports).summarize()
            summary['length_range'] = f"{lo}-{hi if hi != float('inf') else '∞'}"
            results[label] = summary
    
    return results


def analyze_by_source_type(reports: List[ErrorReport],
                           source_types: List[str]) -> Dict:
    """Analyze errors by source type (document vs sentence)."""
    results = {}
    
    type_reports = defaultdict(list)
    for report, src_type in zip(reports, source_types):
        type_reports[src_type].append(report)
    
    for src_type, type_reps in type_reports.items():
        results[src_type] = ErrorSummary(type_reps).summarize()
    
    return results


def find_systematic_patterns(reports: List[ErrorReport], 
                             min_count: int = 3) -> List[Dict]:
    """Find systematic error patterns that appear frequently."""
    patterns = []
    
    # Collect all missing names
    missing_names = Counter()
    for report in reports:
        for err in report.named_entity_errors:
            if err['type'] == 'missing_name':
                missing_names[err['name']] += 1
    
    # Add frequent missing names as patterns
    for name, count in missing_names.most_common(20):
        if count >= min_count:
            patterns.append({
                'type': 'frequently_missing_name',
                'name': name,
                'count': count,
                'percentage': count / len(reports) * 100
            })
    
    # Collect missing units
    missing_units = Counter()
    for report in reports:
        for err in report.number_errors:
            if err['type'] == 'missing_measurement':
                missing_units[err.get('unit', 'unknown')] += 1
    
    for unit, count in missing_units.most_common(10):
        if count >= min_count:
            patterns.append({
                'type': 'frequently_missing_unit',
                'unit': unit,
                'count': count,
                'percentage': count / len(reports) * 100
            })
    
    # Gap handling issues
    gap_issues = sum(1 for r in reports if r.gap_errors)
    if gap_issues >= min_count:
        patterns.append({
            'type': 'gap_handling_issues',
            'count': gap_issues,
            'percentage': gap_issues / len(reports) * 100
        })
    
    # Word order problems
    poor_word_order = sum(1 for r in reports if r.word_order_score < 0.5)
    if poor_word_order >= min_count:
        patterns.append({
            'type': 'poor_word_order',
            'count': poor_word_order,
            'percentage': poor_word_order / len(reports) * 100,
            'description': 'Examples with word order score < 0.5'
        })
    
    return patterns


def print_error_summary(summary: Dict, patterns: List[Dict]):
    """Print formatted error summary."""
    print("\n" + "=" * 60)
    print("ERROR ANALYSIS SUMMARY")
    print("=" * 60)
    
    print(f"\nTotal Examples: {summary['total_examples']}")
    print(f"Examples with Errors: {summary['examples_with_errors']} "
          f"({summary['error_rate']*100:.1f}%)")
    
    print(f"\nError Counts by Type:")
    for err_type, count in sorted(summary['error_counts'].items(), 
                                   key=lambda x: -x[1]):
        print(f"  {err_type}: {count}")
    
    print(f"\nError Counts by Category:")
    for cat, count in summary['by_category'].items():
        print(f"  {cat}: {count}")
    
    print(f"\nWord Order Score: mean={summary['word_order']['mean']:.3f}, "
          f"median={summary['word_order']['median']:.3f}")
    
    if patterns:
        print("\nSystematic Patterns:")
        for p in patterns[:10]:
            print(f"  - {p['type']}: {p.get('name', p.get('unit', ''))} "
                  f"({p['count']} occurrences, {p['percentage']:.1f}%)")
    
    print()


def main():
    parser = argparse.ArgumentParser(description="Analyze translation errors")
    parser.add_argument("--input", type=str, help="Input predictions JSON file")
    parser.add_argument("--output", type=str, default="results/error_analysis.json",
                        help="Output analysis JSON file")
    parser.add_argument("--test", action="store_true", help="Run self-tests")
    args = parser.parse_args()
    
    if args.test:
        print("Running error analyzer tests...")
        
        analyzer = ErrorAnalyzer()
        
        # Test cases
        test_cases = [
            {
                'source': '1 TÚG ša qá-tim I-tur₄-DINGIR il₅-qé',
                'hypothesis': 'One textile received.',
                'reference': 'Itūr-ilī has received one textile of ordinary quality.',
            },
            {
                'source': 'a-na <gap> KÙ.BABBAR šu-qí-il',
                'hypothesis': 'Pay silver.',
                'reference': 'Pay ... silver.',
            },
            {
                'source': '5 GÍN KÙ.BABBAR a-na Aš-šur šu-qí-il',
                'hypothesis': 'Pay silver to Aššur.',
                'reference': 'Pay 5 shekels of silver to Aššur.',
            }
        ]
        
        for i, tc in enumerate(test_cases):
            report = analyzer.analyze(tc['source'], tc['hypothesis'], 
                                       tc['reference'], i)
            print(f"\nTest {i+1}:")
            print(f"  Named entity errors: {len(report.named_entity_errors)}")
            print(f"  Number errors: {len(report.number_errors)}")
            print(f"  Gap errors: {len(report.gap_errors)}")
            print(f"  Word order score: {report.word_order_score:.3f}")
        
        print("\n✓ Error analyzer tests complete")
        return
    
    if args.input:
        # Load predictions and run analysis
        with open(args.input, 'r') as f:
            data = json.load(f)
        
        analyzer = ErrorAnalyzer()
        
        sources = data.get('sources', [])
        hypotheses = data.get('hypotheses', [])
        references = data.get('references', [])
        scores = data.get('scores', [])
        
        reports = analyzer.analyze_batch(sources, hypotheses, references, scores)
        summary = ErrorSummary(reports).summarize()
        patterns = find_systematic_patterns(reports)
        
        # Length analysis
        length_analysis = analyze_by_length(reports)
        
        # Save results
        results = {
            'summary': summary,
            'patterns': patterns,
            'by_length': length_analysis,
            'examples': [asdict(r) for r in reports]
        }
        
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2, default=float)
        
        print(f"Saved error analysis to {args.output}")
        print_error_summary(summary, patterns)
    else:
        print("Usage: python error_analyzer.py --input predictions.json [--output analysis.json]")
        print("       python error_analyzer.py --test")


if __name__ == "__main__":
    main()
