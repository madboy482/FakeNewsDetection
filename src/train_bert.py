# src/train_bert.py

import os
import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding
)
from datasets import Dataset
import random
import warnings
import json

warnings.filterwarnings("ignore")

# Set seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def main():
    print("🚀 Starting BERT-based fake news detection...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load and clean data
    df = pd.read_csv("data/liar2_meta.csv")
    df['clean_statement'] = df['clean_statement'].fillna("")
    df['clean_justification'] = df['clean_justification'].fillna("")
    df['context'] = df['context'].fillna("")

    labels = sorted(df['label'].unique())
    num_labels = len(labels)
    label_map = {lab: idx for idx, lab in enumerate(labels)}
    inv_label_map = {v: k for k, v in label_map.items()}
    df['labels'] = df['label'].map(label_map)

    train_df, val_df = train_test_split(
        df, test_size=0.2, stratify=df['labels'], random_state=42
    )
    print(f"Train samples: {len(train_df)}, Val samples: {len(val_df)}, Classes: {num_labels}")

    # 2. Tokenizer
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    # 3. Create combined text
    def create_text_input(row):
        text = f"Statement: {row['clean_statement']}"
        if row['clean_justification']:
            text += f" Justification: {row['clean_justification']}"
        return text

    train_df['text_input'] = train_df.apply(create_text_input, axis=1)
    val_df['text_input'] = val_df.apply(create_text_input, axis=1)

    # 4. Tokenization with labels
    def tokenize_function(examples):
        tok = tokenizer(
            examples["text_input"],
            truncation=True,
            max_length=512,
            padding="max_length"
        )
        tok["labels"] = examples["labels"]
        return tok

    train_ds = Dataset.from_pandas(train_df)
    val_ds = Dataset.from_pandas(val_df)
    train_ds = train_ds.map(tokenize_function, batched=True)
    val_ds = val_ds.map(tokenize_function, batched=True)

    columns = ['input_ids', 'attention_mask', 'token_type_ids', 'labels']
    train_ds.set_format(type="torch", columns=columns)
    val_ds.set_format(type="torch", columns=columns)

    # 5. Load model
    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased", num_labels=num_labels
    )

    # 6. Training setup
    training_args = TrainingArguments(
        output_dir="models/bert_fake_news",
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        logging_dir="logs",
        logging_steps=50,
        report_to="none"
    )

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {"accuracy": accuracy_score(labels, preds)}

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics
    )

    # 7. Training
    print("⏱️ Fine-tuning BERT model...")
    trainer.train()

    # 8. Evaluation
    print("📝 Evaluating model...")
    eval_res = trainer.evaluate(eval_dataset=val_ds)
    print(f"Validation Accuracy: {eval_res['eval_accuracy']:.4f}")

    val_preds = trainer.predict(val_ds)
    y_true = val_ds["labels"].numpy()
    y_pred = np.argmax(val_preds.predictions, axis=1)
    print("\n📊 Classification Report:\n", classification_report(y_true, y_pred))

    # 9. Save model and tokenizer
    output_dir = "models/bert_fake_news_final"
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    # 🔧 Fix for JSON error: convert numpy int64 to native int
    def convert_keys_to_int(d):
        return {int(k): int(v) for k, v in d.items()}

    clean_label_map = convert_keys_to_int(label_map)
    clean_inv_label_map = convert_keys_to_int(inv_label_map)

    label_mapping = {
        "label_map": clean_label_map,
        "inv_label_map": clean_inv_label_map
    }

    with open(os.path.join(output_dir, "label_mapping.json"), "w") as f:
        json.dump(label_mapping, f)

    print("✅ Training complete!")
    print(f"Model and tokenizer saved at: {output_dir}")

if __name__ == "__main__":
    main()