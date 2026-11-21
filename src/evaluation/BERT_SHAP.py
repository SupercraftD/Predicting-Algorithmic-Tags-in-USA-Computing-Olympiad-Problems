#Evaluates the saved BERT model with SHAP, generating plots for each tag and a text plot for a sample problem for each tag.
#(Note: this was originally run on Google Colab)

import json
import torch
import numpy as np
import pandas as pd
import shap
from transformers import AutoTokenizer, AutoModelForSequenceClassification

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "models/saved_bert_model"

possible_tags = [
    "Bit Manipulation", "Data Structures", "Dynamic Programming", "Geometry",
    "Graphs & Trees", "Greedy & Implementation", "Math & Combinatorics",
    "Search & Sort", "Strings", "Two Pointers & Sliding Window"
]

print("Loading saved model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)
model.to(device)
model.eval()

with open("data/testing.json", "r") as f:
    data = json.load(f)

def format_text(item):
    text = item.get("text", "")

    if "Problem credits" in text:
      text = text[:text.index("Problem credits")].strip()

    constraints = item.get("constraints", "")
    if constraints:
        return f"Constraints:\n{constraints}\n\nProblem Description:\n{text}"
    return text

sample_texts = [format_text(item) for item in data]

def predict_pipeline(texts):
    if isinstance(texts, np.ndarray):
        texts = texts.tolist()

    inputs = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)

        # RETURN RAW LOGITS INSTEAD OF EXPIT/SIGMOID
        logits = outputs.logits.cpu().numpy()

    return logits

print("Initializing SHAP explainer...")
explainer = shap.Explainer(predict_pipeline, tokenizer, output_names=possible_tags)

print("Computing SHAP values...")
shap_values = explainer(sample_texts)

for target_tag in possible_tags:
  target_idx = possible_tags.index(target_tag)

  print(f"\nGenerating explanation for sample #0 for tag: {target_tag}")
  shap.plots.text(shap_values[0, :, target_idx])

  print(f"\nPlotting global word importance for: {possible_tags[target_idx]}")
  shap.plots.bar(shap_values[:, :, target_idx])