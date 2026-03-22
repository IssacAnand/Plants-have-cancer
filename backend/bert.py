import os
import re
import json
import sys
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
import evaluate

from dataset import PlantWildTextDataset


# ──────────────────────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────────────────────

MODEL_NAME     = "recobo/agriculture-bert-uncased"
DATASET_PATH   = "./data/text/plantwild.json"
LABEL_MAP_PATH = "./data/label_map.json"
SAVE_DIR       = "./checkpoints"
DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"

NUM_LABELS     = 89
MAX_LENGTH     = 128
BATCH_SIZE     = 8
NUM_EPOCHS     = 10
LEARNING_RATE  = 2e-5
WEIGHT_DECAY   = 0.01
 

# ──────────────────────────────────────────────────────────────────────────────
# DATA PREPROCESSING
# ──────────────────────────────────────────────────────────────────────────────

def remove_disease_name(row):
    disease     = str(row["disease"]).strip()
    description = str(row["description"]).strip()
 
    if not disease or not description:
        return description
 
    cleaned = re.sub(re.escape(disease), "", description, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" ,.-:;")
    return cleaned
 
 
def preprocess_text_data(path, label_map):
    df = pd.read_json(path)
 
    # reshape wide -> long
    df_long = df.melt(var_name="disease", value_name="description")
 
    # clean
    df_long = df_long.dropna()
    df_long["description"] = df_long["description"].astype(str).str.strip()
    df_long = df_long[df_long["description"] != ""]
 
    # remove disease name from descriptions
    df_long["description"] = df_long.apply(remove_disease_name, axis=1)
 
    # create numeric label from existing map
    df_long["label"] = df_long["disease"].map(label_map)
    df_long = df_long.dropna(subset=["label"])
    df_long["label"] = df_long["label"].astype(int)
 
    return df_long
 
 
def build_datasets(tokenizer, label_map):

    preprocessed_df = preprocess_text_data(DATASET_PATH, label_map)
 
    train_df, test_df = train_test_split(
        preprocessed_df,
        test_size=0.2,
        stratify=preprocessed_df["label"],
        random_state=42,
    )
 
    train_dataset = PlantWildTextDataset(train_df, tokenizer, max_length=MAX_LENGTH)
    test_dataset  = PlantWildTextDataset(test_df,  tokenizer, max_length=MAX_LENGTH)
 
    print(f"Train samples: {len(train_df)}  Test samples: {len(test_df)}")
 
    return train_dataset, test_dataset, train_df, test_df


# ──────────────────────────────────────────────────────────────────────────────
# METRICS
# ──────────────────────────────────────────────────────────────────────────────

metric_f1  = evaluate.load("f1")
metric_acc = evaluate.load("accuracy")
 
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=-1)
    f1  = metric_f1.compute(predictions=predictions, references=labels, average="macro")
    acc = metric_acc.compute(predictions=predictions, references=labels)
    return {**f1, **acc}


# ──────────────────────────────────────────────────────────────────────────────
# TRAINING
# ──────────────────────────────────────────────────────────────────────────────

def train(model, train_dataset, test_dataset):
 
    model_slug = MODEL_NAME.split("/")[-1]
 
    args = TrainingArguments(
        os.path.join(SAVE_DIR, "trainer_tmp"),
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=NUM_EPOCHS,
        weight_decay=WEIGHT_DECAY,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        push_to_hub=False,
        fp16=True,
    )
 
    trainer = Trainer(
        model,
        args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
    )
 
    trainer.train()
    metrics = trainer.evaluate()
    print(metrics)
 
    os.makedirs(SAVE_DIR, exist_ok=True)
    trainer.save_model(os.path.join(SAVE_DIR, "best_text_encoder.pt"))
 
    return trainer


# ──────────────────────────────────────────────────────────────────────────────
# EVALUATION
# ──────────────────────────────────────────────────────────────────────────────

def evaluate_predictions(trainer, test_dataset, label_map, test_df):
    """Run predictions and print F1 / accuracy. Also logs top misclassifications."""
    pred_output = trainer.predict(test_dataset)
    logits      = pred_output.predictions
    y_pred      = np.argmax(logits, axis=-1)
    y_true      = pred_output.label_ids
 
    id_to_label = {v: k for k, v in label_map.items()}
    pred_names  = [id_to_label[int(x)] for x in y_pred]
    true_names  = [id_to_label[int(x)] for x in y_true]
 
    f1  = metric_f1.compute(predictions=y_pred, references=y_true, average="macro")
    acc = metric_acc.compute(predictions=y_pred, references=y_true)
 
    print("F1:",       f1)
    print("Accuracy:", acc)
 
    # Misclassification analysis ------------------------------------------------------------------
    raw_df               = pd.read_json(DATASET_PATH)
    label_descriptions_df = pd.DataFrame([
        {
            "Disease Name":       disease,
            "Sample Description": raw_df[disease].dropna().iloc[0] if not raw_df[disease].dropna().empty else "",
        }
        for disease in raw_df.columns
    ])
 
    results_df = pd.DataFrame({
        "True Label":      true_names,
        "Predicted Label": pred_names,
        "Description":     test_df["description"].reset_index(drop=True),
    })
 
    misclassified_df = results_df[results_df["True Label"] != results_df["Predicted Label"]]
 
    for side in [("True Label", "True Label Description"), ("Predicted Label", "Predicted Label Description")]:
        col, rename = side
        misclassified_df = pd.merge(
            misclassified_df,
            label_descriptions_df[["Disease Name", "Sample Description"]],
            left_on=col, right_on="Disease Name",
            suffixes=("", f"_{col}"),
        )
        misclassified_df = misclassified_df.drop(columns=["Disease Name"])
        misclassified_df = misclassified_df.rename(columns={"Sample Description": rename})
 
    misclassification_counts      = misclassified_df.groupby(["True Label", "Predicted Label"]).size().reset_index(name="Count")
    top_misclassifications_summary = misclassification_counts.sort_values(by="Count", ascending=False).head(10)
 
    print("\nTop 10 misclassified pairs:")
    print(top_misclassifications_summary.to_string(index=False))
 
    print("\nTop 10 individual misclassified samples:")
    pd.set_option("display.max_colwidth", None)
    print(misclassified_df.head(10).to_string(index=False))
    pd.reset_option("display.max_colwidth")


# ──────────────────────────────────────────────────────────────────────────────
# EMBEDDING EXTRACTION
# ──────────────────────────────────────────────────────────────────────────────

class FeatureExtractorModel(torch.nn.Module):
    def __init__(self, original_model):
        super().__init__()
        self.bert = original_model.bert
 
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # [CLS] token embedding (index 0) used as sentence representation
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        return cls_embedding
 
 
def extract_features(dataset, model, device, batch_size=16):
    """Extract [CLS] token embeddings from BERT for all samples in a dataset."""
    feature_model = FeatureExtractorModel(model)
    feature_model.to(device)
    feature_model.eval()
 
    all_features = []
    all_labels   = []
    dataloader   = DataLoader(dataset, batch_size=batch_size, shuffle=False)  # shuffle=False is critical
 
    with torch.no_grad():
        for batch in dataloader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["label"]
            features       = feature_model(input_ids=input_ids, attention_mask=attention_mask)
            all_features.append(features.cpu())
            all_labels.append(labels)
 
    return torch.cat(all_features, dim=0), torch.cat(all_labels, dim=0)
 
 
def save_features(train_dataset, test_dataset, model, tokenizer):
    """Load best model, extract features for train/test, and save to disk."""
    model     = AutoModelForSequenceClassification.from_pretrained(os.path.join(SAVE_DIR, "best_text_encoder.pt"))
    tokenizer = AutoTokenizer.from_pretrained(os.path.join(SAVE_DIR, "best_text_encoder.pt"))
    model.to(DEVICE)
    model.eval()
 
    train_features, train_labels = extract_features(train_dataset, model, DEVICE)
    test_features,  test_labels  = extract_features(test_dataset,  model, DEVICE)
 
    print(f"Train text features: {train_features.shape}")
    print(f"Test  text features: {test_features.shape}")
 
    os.makedirs(SAVE_DIR, exist_ok=True)
 
    torch.save(
        {
            "train_features": train_features,
            "train_labels":   train_labels,
            "test_features":  test_features,
            "test_labels":    test_labels,
        },
        os.path.join(SAVE_DIR, "text_embeddings.pt"),
    )
 
    print(f"\n✓ Features saved to {os.path.join(SAVE_DIR, 'text_embeddings.pt')}")

# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

def main():
    print(f"PyTorch {torch.__version__}")
    print(f"Device: {DEVICE}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
 
    # Tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model     = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=NUM_LABELS)
 
    print(f"Output labels:  {model.config.num_labels}")
    print(f"Hidden size:    {model.config.hidden_size}")
 
    # Data
    with open(LABEL_MAP_PATH, "r") as f:
        label_map = json.load(f)
 
    train_dataset, test_dataset, train_df, test_df = build_datasets(tokenizer, label_map)
 
    # Train
    print(f"\nFine-tuning {MODEL_NAME} for {NUM_EPOCHS} epochs...\n")
    trainer = train(model, train_dataset, test_dataset)
 
    # Save tokenizer alongside model
    tokenizer.save_pretrained(os.path.join(SAVE_DIR, "best_text_encoder.pt"))
 
    # Evaluate
    print("\nRunning final evaluation...")
    evaluate_predictions(trainer, test_dataset, label_map, test_df)
 
    # Extract and save features
    print("\nExtracting BERT features...")
    save_features(train_dataset, test_dataset, model, tokenizer)
 
 
if __name__ == "__main__":
    main()