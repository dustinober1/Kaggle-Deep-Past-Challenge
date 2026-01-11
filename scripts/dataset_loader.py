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
    
    # Define file paths
    train_path = os.path.join(data_dir, "train_final.csv")
    val_path = os.path.join(data_dir, "val_final.csv")
    
    # Check if files exist
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Training file not found at {train_path}")
    if not os.path.exists(val_path):
        raise FileNotFoundError(f"Validation file not found at {val_path}")
        
    # Load CSVs
    print(f"Loading training data from {train_path}...")
    train_df = pd.read_csv(train_path)
    print(f"Loaded {len(train_df)} training examples.")
    
    print(f"Loading validation data from {val_path}...")
    val_df = pd.read_csv(val_path)
    print(f"Loaded {len(val_df)} validation examples.")
    
    # Ensure necessary columns exist. 
    # Based on previous phases, we expect 'transliteration' (source) and 'translation' (target)
    # However, merge scripts might have named them differently or kept original names.
    # Let's verify and standardize.
    
    required_cols = ['transliteration', 'translation']
    for col in required_cols:
        if col not in train_df.columns:
             # Fallback check for potentially normalized columns if main ones missing
             # But train_final.csv should have them. 
             raise ValueError(f"Training data missing required column: {col}")
        if col not in val_df.columns:
             raise ValueError(f"Validation data missing required column: {col}")

    # Create HuggingFace Datasets
    # We want a format convenient for seq2seq: {'translation': {'ak': ..., 'en': ...}} 
    # OR just keep columns and let the map function handle it.
    # Standard transformers approach often expects 'translation' column which is a dict, 
    # OR specific source/target columns. Let's keep it simple: keep columns.
    
    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)
    
    dataset = DatasetDict({
        'train': train_dataset,
        'validation': val_dataset
    })
    
    return dataset

if __name__ == "__main__":
    # Simple test
    try:
        ds = get_akkadian_dataset()
        print("\nDataset structure:")
        print(ds)
        print("\nFirst training example:")
        print(ds['train'][0])
    except Exception as e:
        print(f"Error loading dataset: {e}")
