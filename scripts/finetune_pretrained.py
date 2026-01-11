import argparse
import os
import torch
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
)
import numpy as np
import sacrebleu
from dataset_loader import get_akkadian_dataset

# Model Checkpoints
MODELS = {
    "mbart": "facebook/mbart-large-50-many-to-many-mmt",
    "nllb": "facebook/nllb-200-distilled-600M",
    "m2m": "facebook/m2m100_418M"
}

def compute_metrics(eval_preds, tokenizer):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
        
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    
    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    # Simple post-processing
    decoded_preds = [pred.strip() for pred in decoded_preds]
    decoded_labels = [[label.strip()] for label in decoded_labels]
    
    bleu = sacrebleu.corpus_bleu(decoded_preds, decoded_labels)
    
    return {"bleu": bleu.score}

def main():
    parser = argparse.ArgumentParser(description="Fine-tune pretrained translation models")
    parser.add_argument("--model", type=str, required=True, choices=MODELS.keys(), help="Model architecture")
    parser.add_argument("--output_dir", type=str, default="models/pretrained", help="Output directory")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size per device")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--smoke_test", action="store_true", help="Run a quick test with few samples")
    
    args = parser.parse_args()
    
    model_checkpoint = MODELS[args.model]
    print(f"Loading {args.model} from {model_checkpoint}...")
    
    # Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    
    # Setup source and target languages
    # For mBART/NLLB/M2M, we need to set specific lang codes.
    # Since Akkadian is not standard, we map:
    # Source (Akkadian) -> Use a placeholder or nearest neighbor (e.g., 'ar_AR' or generic)
    # Target (English) -> 'en_XX' (mBART), 'eng_Latn' (NLLB), 'en' (M2M)
    
    if args.model == "mbart":
        tokenizer.src_lang = "ar_AR" # Proxy for Akkadian
        tokenizer.tgt_lang = "en_XX"
        src_lang_code = "ar_AR"
        tgt_lang_code = "en_XX"
    elif args.model == "nllb":
        src_lang_code = "eng_Latn" 
        tgt_lang_code = "eng_Latn" 
        tokenizer.src_lang = src_lang_code
        tokenizer.tgt_lang = tgt_lang_code
    elif args.model == "m2m":
        src_lang_code = "en" 
        tgt_lang_code = "en"
        tokenizer.src_lang = src_lang_code
        tokenizer.tgt_lang = tgt_lang_code
        
    # Load Data
    raw_datasets = get_akkadian_dataset()
    if args.smoke_test:
        raw_datasets["train"] = raw_datasets["train"].select(range(10))
        raw_datasets["validation"] = raw_datasets["validation"].select(range(10))
        
    # Preprocessing
    max_input_length = 128
    max_target_length = 128
    
    def preprocess_function(examples):
        inputs = [ex for ex in examples["transliteration_cleaned"]] # Use cleaned if available
        targets = [ex for ex in examples["translation_normalized"]]
        
        # For NLLB/M2M we might need to handle tokenizer languages differently in call
        # But generally:
        if args.model == "mbart":
             tokenizer.src_lang = src_lang_code
        
        model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True)
        
        # Setup target tokenization
        if args.model == "m2m":
            # M2M needs to set target lang context manager usually, or just tokenizer targets
             with tokenizer.as_target_tokenizer():
                labels = tokenizer(targets, max_length=max_target_length, truncation=True)
        else:
            # mBART/NLLB
            with tokenizer.as_target_tokenizer():
                 labels = tokenizer(targets, max_length=max_target_length, truncation=True)

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    print("Tokenizing dataset...")
    tokenized_datasets = raw_datasets.map(preprocess_function, batched=True, remove_columns=raw_datasets["train"].column_names)
    
    # Load Model
    model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
    
    # Set decoder start token id if needed (crucial for multilingual models)
    if args.model == "mbart":
        model.config.decoder_start_token_id = tokenizer.lang_code_to_id[tgt_lang_code]
    elif args.model == "nllb":
        # NLLB usually handles this auto if processed correctly, but explicit is good
         model.config.decoder_start_token_id = tokenizer.convert_tokens_to_ids(tgt_lang_code)
    elif args.model == "m2m":
        model.config.decoder_start_token_id = tokenizer.get_lang_id(tgt_lang_code)

    # Training Arguments
    run_name = f"{args.model}-finetune"
    output_path = os.path.join(args.output_dir, run_name)
    
    # Check for device
    if torch.backends.mps.is_available():
        print("Using MPS device.")
        device = "mps"
    elif torch.cuda.is_available():
        print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
        device = "cuda"
    else:
        print("Using CPU.")
        device = "cpu"
        
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_path,
        eval_strategy="epoch",
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        weight_decay=0.01,
        save_total_limit=2,
        num_train_epochs=args.epochs,
        predict_with_generate=True,
        fp16=False, # MPS doesn't support fp16 mixed precision well usually, standard float32 is safer
        use_mps_device=True if device == "mps" else False,
        push_to_hub=False,
    )
    
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=lambda preds: compute_metrics(preds, tokenizer),
    )
    
    print("Starting training...")
    trainer.train()
    
    print("Evaluating...")
    metrics = trainer.evaluate()
    print(f"Metrics: {metrics}")
    
    trainer.save_model(output_path)
    print(f"Model saved to {output_path}")

if __name__ == "__main__":
    main()
