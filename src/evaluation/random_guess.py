# Evaluates a Dummy Classifier using a Stratified strategy
# Reports the random baseline's performance against the same test set
# Runs 1000 trials

import json
import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.metrics import f1_score, accuracy_score
from sklearn.preprocessing import MultiLabelBinarizer

POSSIBLE_TAGS = [
    "Bit Manipulation", "Data Structures", "Dynamic Programming", "Geometry",
    "Graphs & Trees", "Greedy & Implementation", "Math & Combinatorics",
    "Search & Sort", "Strings", "Two Pointers & Sliding Window",
]

# Load train and test label sets
with open("data/training.json") as f:
    train_data = json.load(f)
with open("data/testing.json") as f:
    test_data = json.load(f)

train_data = [i for i in train_data if len(i["tags"]) > 0]
test_data = [i for i in test_data if len(i["tags"]) > 0]

mlb = MultiLabelBinarizer(classes=POSSIBLE_TAGS)
Y_train = mlb.fit_transform([i["tags"] for i in train_data])
Y_true  = mlb.transform([i["tags"] for i in test_data])

macro_f1s, micro_f1s, accs = [], [], []

for trial in range(1000):
    Y_pred = np.zeros_like(Y_true)
    for j, tag in enumerate(POSSIBLE_TAGS):

        clf = DummyClassifier(strategy="stratified", random_state=trial)
        clf.fit(np.zeros((len(Y_train), 1)), Y_train[:, j])
        Y_pred[:, j] = clf.predict(np.zeros((len(Y_true), 1)))

    macro_f1s.append(f1_score(Y_true, Y_pred, average="macro", zero_division=0))
    micro_f1s.append(f1_score(Y_true, Y_pred, average="micro", zero_division=0))
    accs.append(accuracy_score(Y_true, Y_pred))

print("Stratified random baseline (n=1000 trials):")
print(f"  Macro F1: {np.mean(macro_f1s):.4f} ± {np.std(macro_f1s):.4f}")
print(f"  Micro F1: {np.mean(micro_f1s):.4f} ± {np.std(micro_f1s):.4f}")
print(f"  Acc: {np.mean(accs):.4f} ± {np.std(accs):.4f}")