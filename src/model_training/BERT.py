# Trains the BERT models (SciBERT, CodeBERT, DeBERTa)
# To change which BERT model is trained, edit the model_name variable and uncomment the correct training_args
# (Note: this was originally run on Google Colab)


import torch
import json
import numpy as np
import pandas as pd
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments
)
from sklearn.metrics import classification_report, f1_score, accuracy_score
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
from scipy.special import expit

possible_tags = [
    "Bit Manipulation", "Data Structures", "Dynamic Programming", "Geometry",
    "Graphs & Trees", "Greedy & Implementation", "Math & Combinatorics",
    "Search & Sort", "Strings", "Two Pointers & Sliding Window"
]


with open("data/training.json", "r") as file:
    data = json.load(file)

def format_text(item):
    text = item.get("text", "")

    if "Problem credits" in text:
      text = text[:text.index("Problem credits")].strip()

    constraints = item.get("constraints", "")
    if constraints:
        return f"Constraints:\n{constraints}\n\nProblem Description:\n{text}"
    return text

df = pd.DataFrame([{
    'text': format_text(item),
    'labels': [float(possible_tags[i] in item.get("tags", [])) for i in range(len(possible_tags))]
} for item in data])

msss = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
X = df['text'].values
y = np.array(df['labels'].tolist())

# train_dataset = Dataset.from_pandas(df)

train_idx, val_idx = next(msss.split(X, y))
train_df, val_df = df.iloc[train_idx], df.iloc[val_idx]

train_dataset = Dataset.from_pandas(train_df)
val_dataset = Dataset.from_pandas(val_df)

with open("manualTestSet.json","r") as file:
  test_data = json.load(file)

test_df = pd.DataFrame([{
    'text': format_text(item),
    'labels': [float(possible_tags[i] in item.get("tags", [])) for i in range(len(possible_tags))]
} for item in test_data])


test_dataset = Dataset.from_pandas(test_df)

model_name = "microsoft/deberta-v3-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)

train_dataset = train_dataset.map(tokenize_function, batched=True)
val_dataset = val_dataset.map(tokenize_function, batched=True)
test_dataset = test_dataset.map(tokenize_function, batched=True)

# pos_weight = (number of negative samples) / (number of positive samples)
train_labels = np.array(train_df['labels'].tolist())
pos_counts = train_labels.sum(axis=0)
neg_counts = len(train_labels) - pos_counts

raw_weights = (neg_counts + 1e-5) / (pos_counts + 1e-5)

# Dampen the weights using a square root
pos_weights = torch.tensor(np.sqrt(raw_weights), dtype=torch.float32)

print("Dampened pos_weights for tags:")
for tag, w in zip(possible_tags, pos_weights):
    print(f"  {tag:30s}: {w.item():.2f}")

#loads and freezes model for any of SciBERT, CodeBERT, or DeBERTa
def load_and_freeze_model(model_name, num_labels, unfreeze_last_n=3):
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        problem_type="multi_label_classification"
    )

    for param in model.parameters():
        param.requires_grad = False

    if hasattr(model, 'classifier'):
        for param in model.classifier.parameters():
            param.requires_grad = True
    elif hasattr(model, 'score'): # Used by some architectures
        for param in model.score.parameters():
            param.requires_grad = True

    base = getattr(model, model.config.model_type, None)
    if base is None: # Fallback search
        for attr in ['bert', 'roberta', 'deberta']:
            if hasattr(model, attr):
                base = getattr(model, attr)
                break

    if base and hasattr(base, 'encoder'):
        layers = base.encoder.layer
        for layer in layers[-unfreeze_last_n:]:
            for param in layer.parameters():
                param.requires_grad = True

    return model

model = load_and_freeze_model(model_name, num_labels=len(possible_tags), unfreeze_last_n=3)

# Custom Trainer for BCEWithLogitsLoss Weighted Loss
class WeightedBCEWithLogitsTrainer(Trainer):
    def __init__(self, pos_weights=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pos_weights = pos_weights

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")

        if self.pos_weights is not None:
            loss_fct = torch.nn.BCEWithLogitsLoss(pos_weight=self.pos_weights.to(logits.device))
        else:
            loss_fct = torch.nn.BCEWithLogitsLoss()

        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    probs = expit(logits)
    predictions = (probs > 0.5).astype(int)
    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1_macro": f1_score(labels, predictions, average="macro", zero_division=0)
    }

#scibert normal
"""
training_args = TrainingArguments(
    output_dir="./results",
    learning_rate=5e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=12,
    weight_decay=0.1,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1_macro",
    greater_is_better=True,
    logging_steps=5,
)

"""
#deberta
training_args = TrainingArguments(
    output_dir="./results_deberta",
    learning_rate=1e-5,
    warmup_ratio=0.1,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=10,
    weight_decay=0.01,
    fp16=False,
    bf16=torch.cuda.is_bf16_supported(),
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1_macro",
    greater_is_better=True,
    logging_steps=5,
)


trainer = WeightedBCEWithLogitsTrainer(
    pos_weights=pos_weights,
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    processing_class=tokenizer,
    compute_metrics=compute_metrics,
)

print("\nFine-tuning with frozen layers and weighted loss...")
trainer.train()

#Threshold Search (ON VALIDATION SET)
val_predictions = trainer.predict(val_dataset)
val_probs = expit(val_predictions.predictions)
val_true_labels = np.array(val_dataset["labels"])


best_thresholds = []
for i, tag in enumerate(possible_tags):
    best_thresh, best_f1 = 0.5, 0.0
    for thresh in np.arange(0.1, 0.9, 0.05):
        pred_i = (val_probs[:, i] > thresh).astype(int)
        score = f1_score(val_true_labels[:, i], pred_i, zero_division=0)
        if score > best_f1:
            best_f1, best_thresh = score, thresh
    best_thresholds.append(best_thresh)

# Evaluation (ON TEST SET)
predictions = trainer.predict(test_dataset)
test_probs = expit(predictions.predictions)
true_labels = np.array(test_dataset["labels"])

optimal_preds = np.zeros_like(test_probs)
for i in range(len(possible_tags)):
    optimal_preds[:, i] = (test_probs[:, i] > best_thresholds[i]).astype(int)

print("\n================ FINAL EVALUATION ================")
print("Standard (0.5 threshold) Macro F1:", f1_score(true_labels, (test_probs > 0.5).astype(int), average="macro", zero_division=0))
print("Optimized Thresholds Macro F1:   ", f1_score(true_labels, optimal_preds, average="macro", zero_division=0))
print("\nDetailed Report:\n", classification_report(true_labels, optimal_preds, target_names=possible_tags, zero_division=0))
print("Optimized Exact Match Accuracy:", accuracy_score(true_labels, optimal_preds))

print("\n" + "=" * 85)
print(f"{'Tag Name':<32} | {'Accuracy':<8} | {'F1-Score':<8} | {'Support':<7}")
print("-" * 85)

for i, tag in enumerate(possible_tags):
    tag_true = true_labels[:, i]
    tag_pred = optimal_preds[:, i]

    acc = accuracy_score(tag_true, tag_pred)
    f1 = f1_score(tag_true, tag_pred, zero_division=0)
    supp = int(tag_true.sum())

    print(f"{tag:<32} | {acc:<8.4f} | {f1:<8.4f} | {supp:<7}")

print("=" * 85)

save_directory = "models/saved_bert_model"
trainer.save_model(save_directory)
tokenizer.save_pretrained(save_directory)
print(f"Model and tokenizer successfully saved to {save_directory}")