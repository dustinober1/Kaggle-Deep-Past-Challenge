"""
Akkadian-English Translation Dataset Loader

Provides data loading functions for multilingual translation training including:
- Standard dataset loading
- Document-level filtering (for more context)
- Sentence-level filtering (to match test format)
- Curriculum learning (sorted by difficulty)
"""

import pandas as pd
from datasets import Dataset, DatasetDict
import os


def get_akkadian_dataset(data_dir="data/processed"):
    """
    Loads the train and validation datasets for Akkadian-English translation.
    
    Args:
        data_dir (str): Path to the directory containing processed CSVs.
        
    Returns:
        DatasetDict: A HuggingFace DatasetDict containing 'train' and 'validation' splits.
    """
    train_path = os.path.join(data_dir, "train_final.csv")
    val_path = os.path.join(data_dir, "val_final.csv")
    
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Training file not found at {train_path}")
    if not os.path.exists(val_path):
        raise FileNotFoundError(f"Validation file not found at {val_path}")
        
    print(f"Loading training data from {train_path}...")
    train_df = pd.read_csv(train_path)
    print(f"Loaded {len(train_df)} training examples.")
    
    print(f"Loading validation data from {val_path}...")
    val_df = pd.read_csv(val_path)
    print(f"Loaded {len(val_df)} validation examples.")
    
    required_cols = ['transliteration', 'translation']
    for col in required_cols:
        if col not in train_df.columns:
            raise ValueError(f"Training data missing required column: {col}")
        if col not in val_df.columns:
            raise ValueError(f"Validation data missing required column: {col}")

    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)
    
    return DatasetDict({
        'train': train_dataset,
        'validation': val_dataset
    })


def get_document_level_dataset(data_dir="data/processed"):
    """
    Load dataset filtered to document-level examples only.
    
    Document-level examples have more context (full tablets/letters) and are better
    for initial training to learn patterns before fine-tuning on sentences.
    
    Returns:
        DatasetDict: Filtered dataset with only document-level examples.
    """
    dataset = get_akkadian_dataset(data_dir)
    
    # Filter for document-level: typically source == 'document' or longer texts
    def is_document_level(example):
        # Document-level examples typically have source_type == 'document' 
        # or are longer (>200 chars in transliteration)
        if 'source_type' in example and example['source_type'] == 'document':
            return True
        # Fallback: use length heuristic (documents are typically longer)
        translit = example.get('transliteration_cleaned', example.get('transliteration', ''))
        return len(translit) > 200
    
    train_filtered = dataset['train'].filter(is_document_level)
    val_filtered = dataset['validation'].filter(is_document_level)
    
    print(f"Document-level: {len(train_filtered)} train, {len(val_filtered)} val")
    
    return DatasetDict({
        'train': train_filtered,
        'validation': val_filtered
    })


def get_sentence_level_dataset(data_dir="data/processed"):
    """
    Load dataset filtered to sentence-level examples only.
    
    Sentence-level examples match the test format better and are used for
    fine-tuning after learning patterns from documents.
    
    Returns:
        DatasetDict: Filtered dataset with only sentence-level examples.
    """
    dataset = get_akkadian_dataset(data_dir)
    
    # Filter for sentence-level: typically source == 'sentence' or shorter texts
    def is_sentence_level(example):
        if 'source_type' in example and example['source_type'] == 'sentence':
            return True
        # Fallback: use length heuristic (sentences are typically shorter)
        translit = example.get('transliteration_cleaned', example.get('transliteration', ''))
        return len(translit) <= 200
    
    train_filtered = dataset['train'].filter(is_sentence_level)
    val_filtered = dataset['validation'].filter(is_sentence_level)
    
    print(f"Sentence-level: {len(train_filtered)} train, {len(val_filtered)} val")
    
    return DatasetDict({
        'train': train_filtered,
        'validation': val_filtered
    })


def get_curriculum_dataset(data_dir="data/processed", sort_by='length', ascending=True):
    """
    Load dataset sorted for curriculum learning.
    
    Curriculum learning starts with easier examples and progressively moves to harder ones.
    This can help the model learn basic patterns before tackling complex examples.
    
    Args:
        data_dir: Path to data directory
        sort_by: Metric to sort by - 'length' (shorter = easier) or 'ratio' (translation/translit ratio)
        ascending: If True, sort easy→hard; if False, hard→easy
        
    Returns:
        DatasetDict: Dataset sorted by difficulty.
    """
    train_path = os.path.join(data_dir, "train_final.csv")
    val_path = os.path.join(data_dir, "val_final.csv")
    
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    
    # Calculate difficulty metric
    translit_col = 'transliteration_cleaned' if 'transliteration_cleaned' in train_df.columns else 'transliteration'
    trans_col = 'translation_normalized' if 'translation_normalized' in train_df.columns else 'translation'
    
    if sort_by == 'length':
        train_df['_difficulty'] = train_df[translit_col].str.len()
        val_df['_difficulty'] = val_df[translit_col].str.len()
    elif sort_by == 'ratio':
        # Higher ratio = more expansion = potentially more complex
        train_df['_difficulty'] = train_df[trans_col].str.len() / (train_df[translit_col].str.len() + 1)
        val_df['_difficulty'] = val_df[trans_col].str.len() / (val_df[translit_col].str.len() + 1)
    else:
        raise ValueError(f"Unknown sort_by: {sort_by}. Use 'length' or 'ratio'.")
    
    # Sort by difficulty
    train_df = train_df.sort_values('_difficulty', ascending=ascending).reset_index(drop=True)
    # Note: validation set typically shouldn't be sorted as we want consistent evaluation
    
    # Remove helper column
    train_df = train_df.drop(columns=['_difficulty'])
    val_df = val_df.drop(columns=['_difficulty'], errors='ignore')
    
    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)
    
    order = "easy→hard" if ascending else "hard→easy"
    print(f"Curriculum ({sort_by}, {order}): {len(train_dataset)} train, {len(val_dataset)} val")
    
    return DatasetDict({
        'train': train_dataset,
        'validation': val_dataset
    })


def get_mixed_dataset(data_dir="data/processed", doc_ratio=0.5, shuffle=True, seed=42):
    """
    Load dataset with a mix of document-level and sentence-level examples.
    
    This combines the benefits of both: document context and test-format matching.
    
    Args:
        data_dir: Path to data directory
        doc_ratio: Ratio of document-level examples (0.0-1.0)
        shuffle: Whether to shuffle the combined dataset
        seed: Random seed for shuffling
        
    Returns:
        DatasetDict: Mixed dataset with controlled proportions.
    """
    doc_ds = get_document_level_dataset(data_dir)
    sent_ds = get_sentence_level_dataset(data_dir)
    
    # Calculate how many to sample from each
    total_train = len(doc_ds['train']) + len(sent_ds['train'])
    n_docs = int(total_train * doc_ratio)
    n_sents = total_train - n_docs
    
    # Sample from each (or take all if not enough)
    doc_train = doc_ds['train'].select(range(min(n_docs, len(doc_ds['train']))))
    sent_train = sent_ds['train'].select(range(min(n_sents, len(sent_ds['train']))))
    
    # Combine
    from datasets import concatenate_datasets
    combined_train = concatenate_datasets([doc_train, sent_train])
    
    if shuffle:
        combined_train = combined_train.shuffle(seed=seed)
    
    # Validation: use all from both
    combined_val = concatenate_datasets([doc_ds['validation'], sent_ds['validation']])
    
    print(f"Mixed ({doc_ratio:.0%} doc): {len(combined_train)} train, {len(combined_val)} val")
    
    return DatasetDict({
        'train': combined_train,
        'validation': combined_val
    })


if __name__ == "__main__":
    # Test all loading functions
    print("=" * 50)
    print("Testing dataset loaders")
    print("=" * 50)
    
    try:
        print("\n1. Standard dataset:")
        ds = get_akkadian_dataset()
        print(f"   Train: {len(ds['train'])}, Val: {len(ds['validation'])}")
        
        print("\n2. Document-level dataset:")
        doc_ds = get_document_level_dataset()
        
        print("\n3. Sentence-level dataset:")
        sent_ds = get_sentence_level_dataset()
        
        print("\n4. Curriculum dataset (by length):")
        curr_ds = get_curriculum_dataset(sort_by='length')
        
        print("\n5. Mixed dataset (50% docs):")
        mixed_ds = get_mixed_dataset(doc_ratio=0.5)
        
        print("\n✓ All dataset loaders working!")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
