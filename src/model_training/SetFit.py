# Trains SetFit with "sentence-transformers/all-MiniLM-L6-v2" backbone
# Saves to "models/trained_setfit_model"
# (Note: this was originally run on Google Colab)

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

from datasets import Dataset
from sklearn.metrics import classification_report, f1_score, accuracy_score
from setfit import SetFitModel, Trainer, TrainingArguments
import json

TAG_NAMES = [
    "Bit Manipulation",
    "Data Structures",
    "Dynamic Programming",
    "Geometry",
    "Graphs & Trees",
    "Greedy & Implementation",
    "Math & Combinatorics",
    "Search & Sort",
    "Strings",
    "Two Pointers & Sliding Window"
]

# Pick a high-performing embedding backbone
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

def load_and_process_json(json_file_path):
    with open(json_file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    texts = []
    labels = []

    for entry in data:
        prob_text = entry.get("text", "").strip()

        if "Problem credits" in prob_text:
            prob_text = prob_text.split("Problem credits")[0].strip()


        constraints = entry.get("constraints", "")

        if constraints:
            full_input = f"{prob_text}\n\nConstraints:\n{constraints}"
        else:
            full_input = prob_text

        raw_tags = entry.get("tags", [])
        binary_label = [1.0 if tag in raw_tags else 0.0 for tag in TAG_NAMES]

        texts.append(full_input)
        labels.append(binary_label)

    return Dataset.from_dict({"text": texts, "label": labels})

TRAIN_JSON_PATH = "data/training.json"
TEST_JSON_PATH = "data/testing.json"

print("Loading and formatting datasets from JSON...")
train_ds = load_and_process_json(TRAIN_JSON_PATH)
test_ds = load_and_process_json(TEST_JSON_PATH)

print(f"Loaded {len(train_ds)} training samples and {len(test_ds)} test samples.")

print(f"Loading SetFit backbone: {MODEL_NAME}...")

model = SetFitModel.from_pretrained(
    MODEL_NAME,
    multi_target_strategy="multi-output"
)

training_args = TrainingArguments(
    batch_size=12,
    num_epochs=10,
    num_iterations=10,
    body_learning_rate=2e-5,
    head_learning_rate=2e-5,
    seed=42,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=test_ds,
    metric="f1",
)

print("Starting SetFit contrastive training...")
trainer.train()

# save the trained model to a local directory after training
model_save_path = "models/trained_setfit_model"
model.save_pretrained(model_save_path)
print(f"Trained model saved to: {model_save_path}")


#tune thresholds with validation split
test_indices, val_indices = train_test_split(
    range(len(test_ds)),
    test_size=0.5,
    random_state=42
)

val_ds = test_ds.select(val_indices)
final_test_ds = test_ds.select(test_indices)

print(f"Validation samples: {len(val_ds)}")
print(f"Final Test samples: {len(final_test_ds)}")

#Re-calculate thresholds on the VALIDATION set only
y_val = np.array(val_ds['label'])
val_probs = model.predict_proba(val_ds['text'])
if isinstance(val_probs, torch.Tensor): val_probs = val_probs.cpu().numpy()

new_optimal_thresholds = []
print("\n--- New Thresholds (calculated on Validation Set) ---")
for i, tag in enumerate(TAG_NAMES):
    best_thresh = 0.5
    best_f1 = -1.0
    for thresh in np.arange(0.10, 0.90, 0.02):
        pred_col = (val_probs[:, i, 1] >= thresh).astype(int)
        score = f1_score(y_val[:, i], pred_col, zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_thresh = thresh
    new_optimal_thresholds.append(best_thresh)
    print(f"{tag:<30}: {best_thresh:.2f}")

#inal evaluation on the UNSEEN test set
y_final_test = np.array(final_test_ds['label'])
test_probs = model.predict_proba(final_test_ds['text'])
if isinstance(test_probs, torch.Tensor): test_probs = test_probs.cpu().numpy()

y_pred_final = np.zeros_like(test_probs[..., 0])
for i in range(len(TAG_NAMES)):
    y_pred_final[:, i] = (test_probs[:, i, 1] >= new_optimal_thresholds[i]).astype(int)

print("\n--- UNBIASED FINAL EVALUATION ---")
print(classification_report(y_final_test, y_pred_final, target_names=TAG_NAMES, zero_division=0))

opt_macro_f1 = f1_score(y_final_test, y_pred_final, average="macro", zero_division=0)

accuracy = accuracy_score(y_final_test, y_pred_final)

print(f"Optimized Thresholds Macro F1:    {opt_macro_f1:.4f}\n")
print(f"opt accuracy: {accuracy:.4f}\n")

print("Detailed Classification Report (Optimized Thresholds):")
print(classification_report(y_final_test, y_pred_final, target_names=TAG_NAMES, zero_division=0))

print("=" * 85)
print(f"{'Tag Name':<32} | {'Accuracy':<8} | {'F1-Score':<8} | {'Support':<7}")
print("-" * 85)
for i, tag in enumerate(TAG_NAMES):
    acc = accuracy_score(y_final_test[:, i], y_pred_final[:, i])
    f1 = f1_score(y_final_test[:, i], y_pred_final[:, i], zero_division=0)
    sup = int(y_final_test[:, i].sum())
    print(f"{tag:<32} | {acc:<8.4f} | {f1:<8.4f} | {sup:<7}")
print("=" * 85)