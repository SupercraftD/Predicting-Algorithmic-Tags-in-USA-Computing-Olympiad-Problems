# Loads the given model and uses it to predict the training set
# Evaluates the performance and reports metrics


import json
import joblib
import numpy as np
import pandas as pd
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.metrics import classification_report, f1_score, accuracy_score
from sklearn.preprocessing import MultiLabelBinarizer

lemmatizer = WordNetLemmatizer()

POSSIBLE_TAGS = [
    "Bit Manipulation",
    "Data Structures",
    "Dynamic Programming",
    "Geometry",
    "Graphs & Trees",
    "Greedy & Implementation",
    "Math & Combinatorics",
    "Search & Sort",
    "Strings",
    "Two Pointers & Sliding Window",
]

def pre_lemmatize_text(raw_text):
    
    def cleanlatex(txt):
        
        txt = txt[:txt.index("Problem credits:")]

        txt = txt.replace("\n"," ")
        txt = txt.replace("$","")

        latexreplace = {
            "\\cdot":" ",
            "\\leq":"<=",
            "\\le":"<",
            "\\times":" ",
            "\\in":"∈",
            "\\dots":"...",
            "\\texttt":"",
            "\\text{":"",
            "\\ldots":"...",
            "\\end{":"",
            "\\left":"",
            "\\right":"",
            "\\implies":"=>",
            "\\equiv":"==",
            "\\geq":">=",
            "\\ge":">",
            "\\oplus":"XOR",
            "\\not":"!",
            "\\lfloor":" floor( ",
            "\\dagger":"†",
            "\\rfloor":")",
            "\\rightarrow": " → ",
            "\\mathtt{":"",
            "\\neq":" ≠ ",
            "\\ne":" ≠ ",
            "\\pmod":" mod",
            "\\to":"→",
            "\\begin{cases}": " ",
            "\\end{cases}": " ",
        }

        for i in latexreplace:
            txt = txt.replace(i, latexreplace[i])
        
        txt = txt.replace("{", " ")
        txt = txt.replace("}", " ") 
        txt = txt.replace("\\","")

        txt = txt.lower()

        return txt

    def get_wordnet_pos(tag):
        if tag.startswith('J'):
            return 'a'
        elif tag.startswith('V'):
            return 'v'
        elif tag.startswith('N'):
            return 'n'
        elif tag.startswith('R'):
            return 'r'
        else:
            return 'n'


    #Clean, tokenize, POS-tag, and lemmatize text ONCE into a single space-separated string
    cleaned = cleanlatex(raw_text)
    tokens = word_tokenize(cleaned)
    tagged = pos_tag(tokens)

    lemmas = [
        lemmatizer.lemmatize(word, get_wordnet_pos(tag))
        for word, tag in tagged
        if word.isalnum() and not word.isdigit() and len(word) > 1
    ]
    return " ".join(lemmas)

def evaluate_dataset(json_filepath, model_path):
    
    with open(json_filepath, "r") as f:
        data = json.load(f)

    data = [i for i in data if len(i["tags"])>0]

    print(f"Loaded {len(data)} test problems from {json_filepath}...")

    #prepare dataframe
    rows = []
    y_true_labels = []

    for item in data:
        text = item.get("text", "")
        constraints = item.get("constraints", [])

        # Process text features
        processed_text = pre_lemmatize_text(text)
        max_c = np.log10(max(constraints) + 1) if len(constraints) > 0 else 0.0
        min_c = np.log10(min(constraints) + 1) if len(constraints) > 0 else 0.0
        count_c = len(constraints)

        rows.append(
            {
                "processed_text": processed_text,
                "CONSTRAINT_MAX": max_c,
                "CONSTRAINT_MIN": min_c,
                "CONSTRAINT_COUNT": count_c,
            }
        )

        # Store ground truth tags
        y_true_labels.append(item.get("tags", []))

    df_test = pd.DataFrame(rows)

    mlb = MultiLabelBinarizer(classes=POSSIBLE_TAGS)
    Y_true = mlb.fit_transform(y_true_labels)

    #load and predict
    pipeline = joblib.load(model_path)
    Y_pred = pipeline.predict(df_test)

    print("                CLASSIFICATION REPORT")
    print(
        classification_report(
            Y_true,
            Y_pred,
            target_names=POSSIBLE_TAGS,
            zero_division=0,
            digits=4,
        )
    )

    report_dict = classification_report(
        Y_true,
        Y_pred,
        target_names=POSSIBLE_TAGS, 
        zero_division=0, 
        output_dict=True
    )
    report_df = pd.DataFrame(report_dict).transpose()
    print(report_df.round(3).to_csv(sep=';'))


    print("   -- ACCURACY -- ")
    print("True:",accuracy_score(Y_true, Y_pred))

    print("                GLOBAL SUMMARY METRICS")
    print(
        f"Micro F1 Score   : {f1_score(Y_true, Y_pred, average='micro'):.4f}"
    )
    print(
        f"Macro F1 Score   : {f1_score(Y_true, Y_pred, average='macro'):.4f}"
    )
    print(
        f"Weighted F1 Score: {f1_score(Y_true, Y_pred, average='weighted'):.4f}"
    )
    print(
        f"Samples F1 Score : {f1_score(Y_true, Y_pred, average='samples'):.4f}"
    )

evaluate_dataset("data/testing.json","models/usaco_onevrest_nocv_pipeline.pkl")