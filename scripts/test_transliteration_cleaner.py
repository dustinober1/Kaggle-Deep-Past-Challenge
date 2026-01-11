#!/usr/bin/env python3
"""
Tests for transliteration_cleaner.py

Run with: python -m pytest scripts/test_transliteration_cleaner.py -v
Or directly: python scripts/test_transliteration_cleaner.py
"""

import sys
import re
from pathlib import Path


# =============================================================================
# INLINE IMPLEMENTATIONS FOR STANDALONE TESTING
# =============================================================================
# These mirror the implementations in transliteration_cleaner.py
# This allows testing without pandas dependency

def remove_scribal_annotations(text: str) -> str:
    """Remove modern scribal annotations from transliteration."""
    if not text:
        return ""
    
    # Remove errant/erroneous signs completely
    text = re.sub(r'<<[^>]*>>', '', text)
    
    # Remove scribal insertion markers but keep content
    text = re.sub(r'<([^<>]+)>', r'\1', text)
    
    # Remove half brackets
    text = re.sub(r'˹([^˺]*)˺', r'\1', text)
    text = text.replace('˹', '').replace('˺', '')
    
    # Remove certain reading marker
    text = text.replace('!', '')
    
    # Remove uncertain reading marker
    text = re.sub(r'\?(?![^\[]*\])', '', text)
    
    # Remove line divider
    text = text.replace('/', ' ')
    
    # Remove word divider
    text = re.sub(r'(?<![A-ZŠṢṬḪa-zšṣṭḫ]): *', ' ', text)
    
    return text


def standardize_gaps(text: str) -> str:
    """Standardize gap markers in transliteration."""
    if not text:
        return ""
    
    # Handle [... ...] patterns
    text = re.sub(r'\[\s*[\.…]+\s*[\.…]*\s*\]', '<big_gap>', text)
    
    # Handle multiple x patterns
    text = re.sub(r'\[\s*x(\s+x)+\s*\]', '<big_gap>', text, flags=re.IGNORECASE)
    
    # Handle single [x]
    text = re.sub(r'\[\s*x\s*\]', '<gap>', text, flags=re.IGNORECASE)
    
    # Handle ellipsis
    text = re.sub(r'\.{3,}', '<big_gap>', text)
    text = text.replace('…', '<big_gap>')
    
    # Handle broken markers
    text = re.sub(r'\(\s*(?:rest\s+)?(?:broken|damaged|destroyed|missing|illegible)\s*\)', 
                  '<big_gap>', text, flags=re.IGNORECASE)
    
    text = re.sub(r'\(\s*\d+\s*(?:lines?\s+)?(?:broken|missing|destroyed)\s*\)', 
                  '<big_gap>', text, flags=re.IGNORECASE)
    
    # Clean up multiple consecutive gaps (but preserve single gaps)
    text = re.sub(r'(<gap>\s*){2,}', '<gap> ', text)
    text = re.sub(r'(<big_gap>\s*){2,}', '<big_gap> ', text)
    text = re.sub(r'<gap>\s*<big_gap>', '<big_gap>', text)
    text = re.sub(r'<big_gap>\s*<gap>', '<big_gap>', text)
    
    return text


def normalize_brackets(text: str) -> str:
    """Normalize square brackets in transliteration."""
    if not text:
        return ""
    
    # Protect gap markers
    text = text.replace('<gap>', '<<<GAP>>>')
    text = text.replace('<big_gap>', '<<<BIG_GAP>>>')
    
    # Remove brackets
    text = re.sub(r'\[([^\[\]]+)\]', r'\1', text)
    while '[' in text and ']' in text:
        text = re.sub(r'\[([^\[\]]+)\]', r'\1', text)
    
    # Restore gap markers
    text = text.replace('<<<GAP>>>', '<gap>')
    text = text.replace('<<<BIG_GAP>>>', '<big_gap>')
    
    return text


def normalize_determinatives(text: str) -> str:
    """Normalize determinatives in curly brackets."""
    if not text:
        return ""
    
    subscript_map = {
        '0': '₀', '1': '₁', '2': '₂', '3': '₃', '4': '₄',
        '5': '₅', '6': '₆', '7': '₇', '8': '₈', '9': '₉',
    }
    
    def normalize_det_content(match):
        content = match.group(1).lower()
        content = re.sub(r'(\d+)$', 
                        lambda m: ''.join(subscript_map.get(c, c) for c in m.group(1)), 
                        content)
        return '{' + content + '}'
    
    text = re.sub(r'\{([^}]+)\}', normalize_det_content, text)
    return text


def remove_line_numbers(text: str) -> str:
    """Remove line numbers from transliteration."""
    if not text:
        return ""
    
    text = re.sub(r"^\s*\d+['\u2033\u2032\u2019]*\.\s*", '', text, flags=re.MULTILINE)
    text = re.sub(r"\s+\d+['\u2033\u2032\u2019]*\.\s+", ' ', text)
    return text


def clean_whitespace(text: str) -> str:
    """Clean up whitespace in the text."""
    if not text:
        return ""
    
    text = re.sub(r'\s*-\s*', '-', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text


def convert_parenthetical_determinatives(text: str) -> str:
    """Convert parenthetical determinatives to curly-bracket format."""
    if not text:
        return ""
    
    paren_determinatives = [
        'mul', 'lu₂', 'lu2', 'lú', 'e₂', 'e2', 'uru', 'kur',
        'geš', 'ĝeš', 'giš', 'tug₂', 'tug2', 'dub', 'id₂', 'id2',
        'mušen', 'na₄', 'na4', 'kuš', 'u₂', 'u2', 'urudu', 'zabar',
        'ki', 'mi', 'd', 'm', 'f',
    ]
    
    for det in paren_determinatives:
        pattern_before = r'\((' + re.escape(det) + r')\)(?=[A-Za-zŠṢṬḪšṣṭḫÀÁÂÃÄÅàáâãäå₀₁₂₃₄₅₆₇₈₉])'
        text = re.sub(pattern_before, r'{\1}', text, flags=re.IGNORECASE)
        pattern_after = r'(?<=[A-Za-zŠṢṬḪšṣṭḫÀÁÂÃÄÅàáâãäå₀₁₂₃₄₅₆₇₈₉-])\((' + re.escape(det) + r')\)'
        text = re.sub(pattern_after, r'{\1}', text, flags=re.IGNORECASE)
    
    return text


def remove_parenthetical_comments(text: str) -> str:
    """Remove or handle parenthetical comments."""
    if not text:
        return ""
    
    text = re.sub(r'\(\s*erasure\s*\)', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\(\s*(?:blank|uninscribed|space)\s*\)', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\(\s*\d+\s*signs?\s*(?:broken|missing|destroyed)?\s*\)', 
                  '<gap>', text, flags=re.IGNORECASE)
    return text


def clean_transliteration(text: str, verbose: bool = False) -> str:
    """Apply full cleaning pipeline to a transliteration."""
    if not text:
        return ""
    
    text = remove_line_numbers(text)
    text = remove_scribal_annotations(text)
    text = convert_parenthetical_determinatives(text)  # NEW: (d) → {d}
    text = remove_parenthetical_comments(text)
    text = standardize_gaps(text)
    text = normalize_brackets(text)
    text = normalize_determinatives(text)
    text = clean_whitespace(text)
    
    return text


# =============================================================================
# TEST FUNCTIONS
# =============================================================================


def test_remove_scribal_annotations():
    """Test removal of scribal annotations."""
    
    # Test ! removal (certain reading)
    assert remove_scribal_annotations("a-na!") == "a-na"
    assert remove_scribal_annotations("KÙ!.BABBAR") == "KÙ.BABBAR"
    
    # Test ? removal (uncertain reading)
    assert remove_scribal_annotations("a-na?") == "a-na"
    assert remove_scribal_annotations("šum-ma?") == "šum-ma"
    
    # Test half bracket removal ˹ ˺
    assert remove_scribal_annotations("˹a-na˺") == "a-na"
    assert remove_scribal_annotations("˹KÙ˺.BABBAR") == "KÙ.BABBAR"
    
    # Test scribal insertion < > (keep content)
    assert remove_scribal_annotations("<a>-na") == "a-na"
    assert remove_scribal_annotations("a-<na>") == "a-na"
    
    # Test erroneous signs << >> (remove completely)
    assert remove_scribal_annotations("a-na <<error>> text") == "a-na  text"
    
    # Test combined
    assert remove_scribal_annotations("˹a!-na?˺") == "a-na"
    
    print("✓ remove_scribal_annotations tests passed")


def test_standardize_gaps():
    """Test gap standardization."""
    
    # Single unknown sign [x] → <gap>
    assert standardize_gaps("[x]") == "<gap>"
    assert standardize_gaps("[X]") == "<gap>"
    assert standardize_gaps("a-na [x] text") == "a-na <gap> text"
    
    # Multiple unknown signs [x x] → <big_gap>
    assert standardize_gaps("[x x]") == "<big_gap>"
    assert standardize_gaps("[x x x]") == "<big_gap>"
    
    # Ellipsis → <big_gap>
    assert standardize_gaps("...") == "<big_gap>"
    assert standardize_gaps("…") == "<big_gap>"
    assert standardize_gaps("[...]") == "<big_gap>"
    assert standardize_gaps("[… …]") == "<big_gap>"
    
    # Broken markers
    assert standardize_gaps("(broken)") == "<big_gap>"
    assert standardize_gaps("(rest broken)") == "<big_gap>"
    assert standardize_gaps("(3 lines broken)") == "<big_gap>"
    
    # Multiple gaps consolidation
    result = standardize_gaps("[x] [x]")
    assert "<gap>" in result
    
    print("✓ standardize_gaps tests passed")


def test_normalize_brackets():
    """Test bracket normalization."""
    
    # Remove brackets but keep content
    assert normalize_brackets("[KÙ.BABBAR]") == "KÙ.BABBAR"
    assert normalize_brackets("[a]-na") == "a-na"
    assert normalize_brackets("a-[na]") == "a-na"
    assert normalize_brackets("[a-na]") == "a-na"
    
    # Preserve gap markers
    assert normalize_brackets("<gap>") == "<gap>"
    assert normalize_brackets("<big_gap>") == "<big_gap>"
    assert normalize_brackets("a-na <gap> text") == "a-na <gap> text"
    
    print("✓ normalize_brackets tests passed")


def test_normalize_determinatives():
    """Test determinative normalization."""
    
    # Preserve valid determinatives
    assert "{d}" in normalize_determinatives("{d}UTU")
    assert "{ki}" in normalize_determinatives("A-šur{ki}")
    assert "{m}" in normalize_determinatives("{m}A-šur-i-dí")
    
    # Normalize subscript numbers
    assert "{lu₂}" in normalize_determinatives("{lu2}")
    assert "{e₂}" in normalize_determinatives("{e2}")
    
    # Lowercase normalization
    assert "{d}" in normalize_determinatives("{D}")
    assert "{ki}" in normalize_determinatives("{KI}")
    
    print("✓ normalize_determinatives tests passed")


def test_remove_line_numbers():
    """Test line number removal."""
    
    # Standard line numbers
    assert remove_line_numbers("1. a-na") == "a-na"
    assert remove_line_numbers("5. KÙ.BABBAR") == "KÙ.BABBAR"
    assert remove_line_numbers("10. text here") == "text here"
    
    # Line numbers with primes
    assert remove_line_numbers("1'. a-na") == "a-na"
    assert remove_line_numbers("1''. text") == "text"
    
    print("✓ remove_line_numbers tests passed")


def test_clean_whitespace():
    """Test whitespace cleaning."""
    
    # Multiple spaces
    assert clean_whitespace("a    na") == "a na"
    assert clean_whitespace("  text  ") == "text"
    
    # Spaces around hyphens
    assert clean_whitespace("a - na") == "a-na"
    assert clean_whitespace("a- na") == "a-na"
    assert clean_whitespace("a -na") == "a-na"
    
    print("✓ clean_whitespace tests passed")


def test_convert_parenthetical_determinatives():
    """Test conversion of parenthetical determinatives to curly-bracket format."""
    
    # Test (d) before word - deity determinative
    assert "{d}" in convert_parenthetical_determinatives("(d)UTU")
    assert "{d}" in convert_parenthetical_determinatives("(d)IŠKUR")
    assert "{d}" in convert_parenthetical_determinatives("(d)IM-ba-ni")
    
    # Test (ki) after word - location determinative
    assert "{ki}" in convert_parenthetical_determinatives("a-lim(ki)")
    assert "{ki}" in convert_parenthetical_determinatives("kà-ni-iš(ki)")
    
    # Test (m) before word - masculine name marker
    assert "{m}" in convert_parenthetical_determinatives("(m)A-šur-i-dí")
    assert "{m}" in convert_parenthetical_determinatives("(m)a-bi₄-a")
    
    # Test (f) before word - feminine name marker
    assert "{f}" in convert_parenthetical_determinatives("(f)IŠTAR")
    
    # Test preservation of content
    result = convert_parenthetical_determinatives("(d)UTU-tap-pá-i")
    assert "UTU-tap-pá-i" in result
    assert "{d}" in result
    
    # Test combined
    result = convert_parenthetical_determinatives("a-lim(ki) (d)UTU")
    assert "{ki}" in result
    assert "{d}" in result
    
    print("✓ convert_parenthetical_determinatives tests passed")


def test_remove_parenthetical_comments():
    """Test parenthetical comment handling."""
    
    # Erasure removal
    assert remove_parenthetical_comments("a-na (erasure) text") == "a-na  text"
    
    # Blank removal
    assert remove_parenthetical_comments("a-na (blank) text") == "a-na  text"
    
    # Signs markers → gap
    assert "<gap>" in remove_parenthetical_comments("(2 signs)")
    
    print("✓ remove_parenthetical_comments tests passed")


def test_clean_transliteration_full_pipeline():
    """Test the full cleaning pipeline."""
    
    # Complex example with multiple issues
    text1 = "1. ˹a!˺-na? [x] {D}UTU a-[na] (erasure)"
    cleaned1 = clean_transliteration(text1)
    
    # Should have:
    # - No line number (1.)
    # - No half brackets ˹ ˺
    # - No ! or ?
    # - Gap marker for [x]
    # - Normalized determinative {d}
    # - No brackets around a-na
    # - No erasure marker
    
    assert "1." not in cleaned1
    assert "˹" not in cleaned1 and "˺" not in cleaned1
    assert "!" not in cleaned1
    assert "?" not in cleaned1
    assert "<gap>" in cleaned1
    assert "{d}" in cleaned1
    assert "[" not in cleaned1 and "]" not in cleaned1
    assert "(erasure)" not in cleaned1
    
    # Test with ellipsis
    text2 = "[... ...] a-na ..."
    cleaned2 = clean_transliteration(text2)
    assert "<big_gap>" in cleaned2
    
    # Test empty/None handling
    assert clean_transliteration("") == ""
    assert clean_transliteration(None) == ""
    
    print("✓ Full pipeline tests passed")


def test_real_world_examples():
    """Test with realistic Akkadian transliteration examples."""
    
    # Example 1: Letter opening
    text1 = "1. a-na A-šur-na-da um-ma I-dí-A-šur-ma"
    cleaned1 = clean_transliteration(text1)
    assert "a-na" in cleaned1
    assert "A-šur-na-da" in cleaned1
    assert "1." not in cleaned1
    
    # Example 2: With determinatives and damage
    text2 = "{d}UTU {d}EN.ZU a-na [x x] ˹šarri˺"
    cleaned2 = clean_transliteration(text2)
    assert "{d}" in cleaned2
    assert "<big_gap>" in cleaned2
    assert "šarri" in cleaned2
    assert "˹" not in cleaned2
    
    # Example 3: Mixed logograms and syllabic
    text3 = "KÙ.BABBAR! ša a-na? dumu-šu i-din"
    cleaned3 = clean_transliteration(text3)
    assert "KÙ.BABBAR" in cleaned3
    assert "!" not in cleaned3
    assert "?" not in cleaned3
    
    # Example 4: Long break markers
    text4 = "beginning text ... (rest broken)"
    cleaned4 = clean_transliteration(text4)
    assert "<big_gap>" in cleaned4
    assert "(rest broken)" not in cleaned4
    
    print("✓ Real-world example tests passed")


def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("TRANSLITERATION CLEANER TESTS")
    print("=" * 60 + "\n")
    
    test_remove_scribal_annotations()
    test_standardize_gaps()
    test_normalize_brackets()
    test_normalize_determinatives()
    test_remove_line_numbers()
    test_clean_whitespace()
    test_convert_parenthetical_determinatives()  # NEW
    test_remove_parenthetical_comments()
    test_clean_transliteration_full_pipeline()
    test_real_world_examples()
    
    print("\n" + "=" * 60)
    print("ALL TESTS PASSED ✓")
    print("=" * 60)


if __name__ == '__main__':
    run_all_tests()
