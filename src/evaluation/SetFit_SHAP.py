# Runs SHAP analysis on saved SetFit model, visualizing text plots and bar plots for each tag
# (Note: this was originally run on Google Colab)

import shap
import torch
import numpy as np
from setfit import SetFitModel
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
model_save_path = "./trained_setfit_model"
model = SetFitModel.from_pretrained(
    model_save_path,
    multi_target_strategy="multi-output"
)

def predict_probs(texts):
    probs = model.predict_proba(texts)
    if isinstance(probs, torch.Tensor):
        probs = probs.cpu().numpy()

    return probs[:, :, 1]

explainer = shap.Explainer(predict_probs, masker=shap.maskers.Text(tokenizer=model.model_body.tokenizer))

with open("data/testing.json", "r", encoding="utf-8") as f:
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


sample_texts = texts

shap_values = explainer(sample_texts)

#Generate Visualizations
for target_tag in TAG_NAMES:
  target_idx = TAG_NAMES.index(target_tag)

  print(f"\nGenerating explanation for sample #0 for tag: {target_tag}")
  shap.plots.text(shap_values[0, :, target_idx])

  print(f"\nPlotting global word importance for: {TAG_NAMES[target_idx]}")
  shap.plots.bar(shap_values[:, :, target_idx])