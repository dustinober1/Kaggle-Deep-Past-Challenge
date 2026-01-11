import json
import os
import argparse
import matplotlib.pyplot as plt
from glob import glob
import pandas as pd

def load_metrics(metrics_file):
    try:
        with open(metrics_file, 'r') as f:
            data = json.load(f)
        return pd.DataFrame(data)
    except Exception as e:
        print(f"Error reading {metrics_file}: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Compare Training Runs")
    parser.add_argument("--models_dir", type=str, default="models", help="Root models directory")
    args = parser.parse_args()

    # Find all metrics.json files
    metrics_files = glob(os.path.join(args.models_dir, "**", "metrics.json"), recursive=True)
    
    print(f"Found {len(metrics_files)} metrics files.")
    
    results = {}
    
    for mf in metrics_files:
        model_name = os.path.basename(os.path.dirname(mf))
        df = load_metrics(mf)
        if df is not None and not df.empty:
            results[model_name] = df
            
    if not results:
        print("No valid metrics found.")
        return

    # Print Summary Table
    print("\n--- Model Comparison Summary ---")
    summary_data = []
    for name, df in results.items():
        # Get best val loss
        if 'eval_loss' in df.columns:
            best_val_loss = df['eval_loss'].min()
            loss_col = 'eval_loss'
        elif 'val_loss' in df.columns:
            best_val_loss = df['val_loss'].min()
            loss_col = 'val_loss'
        else:
            best_val_loss = float('nan')
            loss_col = None
            
        # Get best BLEU if available
        if 'eval_bleu' in df.columns:
            best_bleu = df['eval_bleu'].max()
        elif 'bleu' in df.columns:
            best_bleu = df['bleu'].max()
        else:
            best_bleu = float('nan')
            
        summary_data.append({
            "Model": name,
            "Best Val Loss": best_val_loss,
            "Best BLEU": best_bleu,
            "Epochs": len(df)
        })
        
    summary_df = pd.DataFrame(summary_data)
    print(summary_df.sort_values("Best Val Loss", ascending=True).to_markdown(index=False, floatfmt=".4f"))
    
    # Simple Plotting (if available environment supports it, effectively mostly for local check or artifact generation)
    # Since we are in an agentic environment, we might not see the plot, but we can save it.
    
    plt.figure(figsize=(10, 6))
    for name, df in results.items():
        if 'eval_loss' in df.columns:
            plt.plot(df['epoch'], df['eval_loss'], label=f"{name} (val_loss)")
        elif 'val_loss' in df.columns:
            plt.plot(df['epoch'], df['val_loss'], label=f"{name} (val_loss)")
            
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Validation Loss Comparison")
    plt.legend()
    plt.grid(True)
    
    plot_path = os.path.join(args.models_dir, "comparison_plot.png")
    plt.savefig(plot_path)
    print(f"\nComparison plot saved to {plot_path}")

if __name__ == "__main__":
    main()
