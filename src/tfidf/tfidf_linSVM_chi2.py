import random
import json

text = []
everytag = set()

with open("normalized_problems.json","r") as file:
    data = json.load(file)

    text = [{"text":i["text"],  "tags":i["tags"]} for i in data]
    
    for i in data:
        for t in i["tags"]:
            everytag.add(t)
    #print(text)

#print(text[3])
#print("-----")

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

#print(cleanlatex(text[random.randint(0, len(text)-1)]))

#exit()


import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

corpus = [cleanlatex(t["text"]) for t in text]# if "Graphs & Trees" in t["tags"]]



base_stop_words = list(TfidfVectorizer(stop_words='english').get_stop_words())
""" #run 3
usaco_noise = [
    'cow', 'cows', 'bessie', 'farm', 'barn', 'barns', 'farmer', 'john', 'fj', 'field',
    'input', 'output', 'sample', 'format', 'line', 'file', 'number', 'time',"elsie",
    "000","test","a_i","b_i","brian dean", "benjamin qi", "dhruv rohatgi",
    "gift","breed","breeds","milk","haybales","bucket","buckets","mildred","300","t_i",
    "l_i","p_i","magazine","herd","spotty","10","nhoj","censor","a_","s_i",    
]"""

usaco_noise = [
    'cow', 'cows', 'bessie', 'farm', 'barn', 'barns', 'farmer', 'john', 'fj', 'field',
    'input', 'output', 'sample', 'format', 'line', 'file', 'number', 'time',"elsie",
    "000","test","a_i","b_i","brian dean", "benjamin qi", "dhruv rohatgi",
    "gift","breed","breeds","milk","haybales","bucket","buckets","mildred","300","t_i",
    "l_i","p_i","magazine","herd","spotty","10","nhoj","censor","a_","s_i",
    "lines","contain","space","separated","integers","print","terminal","stdin","stdout","constraints","contains",
    "conveniently","numbered","cases","satisfy","determine","help","scoring","make","sure","100","cowvid","19","12","a_i","r_i",
    "milked","a_1","order","possible","second","1000","additional","inputs","1001","101","10inputs","5s","75","80","8inputs","twice","default",
    "512mb","21","24","250",

    "bella","slingshot","moonies","moonet","buddy","billboard","fence","pool","beam","paint","flowers","flower","gallons","robot","friendship",
    "meal","flavor","diamonds","pie","ball","airport","session","box","hungry","machine","pours","stars","stall","room","mounting","stopped","arrive",
    "units","clockwise","genome","factor"
]

custom_stop_words = base_stop_words + usaco_noise

vectorizer = TfidfVectorizer(stop_words=custom_stop_words, max_df=0.75, min_df=2, ngram_range=(1,3))

tfidf_matrix = vectorizer.fit_transform(corpus)

feature_names = vectorizer.get_feature_names_out()
dense_matrix = tfidf_matrix.todense()

df = pd.DataFrame(dense_matrix, columns=feature_names)
print(df)

mean_scores = df.mean(axis=0)

# 2. Sort the words by their score in descending order
top_words = mean_scores.sort_values(ascending=False)

# 3. Print the top 20 most important words
print("--- Top 20 Words ---")
print(top_words.head(20))

for target_tag in sorted(list(everytag)):
    
    # Get the indices of the problems that contain this tag
    matching_indices = [i for i, t in enumerate(text) if target_tag in t["tags"]]
    
    # Slice only those rows out of our master TF-IDF DataFrame
    tag_specific_matrix = df.iloc[matching_indices]
    
    # Calculate the top words safely!
    mean_scores = tag_specific_matrix.mean(axis=0)
    top_words = mean_scores.sort_values(ascending=False).head(10)
    
    print(f"{target_tag}")
    print(top_words)

from sklearn.metrics.pairwise import cosine_similarity

def calculate_cosine_matrix(master_tfidf_df, tag_indices_dict):
    """
    master_tfidf_df: Your main TF-IDF DataFrame (words as columns)
    tag_indices_dict: A dictionary mapping tag names to their row indices in the DF
    """
    tags = sorted(list(tag_indices_dict.keys()))
    
    # 1. Compute the "mean vector" representation for each tag
    tag_vectors = []
    for tag in tags:
        indices = tag_indices_dict[tag]
        # Average the TF-IDF vectors for all problems matching this tag
        mean_vector = master_tfidf_df.iloc[indices].mean(axis=0)
        tag_vectors.append(mean_vector)
        
    # Convert list of mean vectors into a single 2D NumPy array (shape: num_tags x num_features)
    tag_vectors_matrix = np.array(tag_vectors)
    
    # 2. Compute the pairwise cosine similarity
    # This outputs a matrix of shape (num_tags x num_tags)
    similarity_matrix = cosine_similarity(tag_vectors_matrix)
    
    # 3. Pack it neatly back into a Pandas DataFrame
    cosine_df = pd.DataFrame(similarity_matrix, index=tags, columns=tags)
    return cosine_df

overlap_df = calculate_cosine_matrix(df, {i:[idx for idx,v in enumerate(text) if i in v["tags"]] for i in sorted(list(everytag))})

print(overlap_df.round(3).to_csv(sep=';'))

print(*usaco_noise,sep="\n")

# =====================================================================
# FINAL HONEST BASELINE: MULTI-OUTPUT LINEAR SVM WITH TUNED THRESHOLD
# =====================================================================
from sklearn.svm import LinearSVC
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import classification_report, f1_score
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
import numpy as np
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.pipeline import Pipeline

# 1. Split labels
y_raw = [t["tags"] for t in text]
mlb = MultiLabelBinarizer()
y = mlb.fit_transform(y_raw)
X_dense = tfidf_matrix.tocsr()

# 2. Split data (80% train, 20% test)
msss = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_index, test_index = next(msss.split(X_dense, y))

X_train, X_test = X_dense[train_index], X_dense[test_index]
y_train, y_test = y[train_index], y[test_index]

# 3. Train the Linear SVM
svm_pipeline = Pipeline([
    ('chi2_select', SelectKBest(score_func=chi2, k=2000)),
    ('svm', LinearSVC(class_weight='balanced', C=1, dual="auto", random_state=42))
])

model = MultiOutputClassifier(svm_pipeline)
model.fit(X_train, y_train)

# =====================================================================
# 4. ADAPTIVE THRESHOLD OPTIMIZATION (WITH CONSERVATIVE BOUNDS)
# =====================================================================
print("--- Optimizing Adaptive Thresholds on Train Split ---")
X_train_scores = np.column_stack([estimator.decision_function(X_train) for estimator in model.estimators_])
best_thresholds = []

for i, class_name in enumerate(mlb.classes_):
    best_f1 = -1
    best_t = 0.0
    
    # Restrict the range so it cannot pick an excessively punishing positive threshold
    # Shifting the window to (-1.0 to 0.2) stops it from setting a 0.5 or 0.6 threshold
    for t in np.arange(-1.0, 0.25, 0.05):
        preds = (X_train_scores[:, i] > t).astype(int)
        score = f1_score(y_train[:, i], preds, average='binary', zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_t = t
            
    # Fallback: If the class has extremely low support, force it to be slightly permissive (e.g., -0.2)
    # to guarantee it actually attempts predictions on the test set.
    if y_train[:, i].sum() < 15: 
        best_t = min(best_t, -0.15)
        
    best_thresholds.append(best_t)
    print(f"Optimal Threshold for [{class_name:28}]: {best_t:5.2f}")

# =====================================================================
# 5. TESTING & METRIC EVALUATION
# =====================================================================
decision_scores = np.column_stack([estimator.decision_function(X_test) for estimator in model.estimators_])

predictions_tuned = np.zeros_like(decision_scores, dtype=int)
for i in range(len(best_thresholds)):
    predictions_tuned[:, i] = (decision_scores[:, i] > best_thresholds[i]).astype(int)

# 6. Print final report
print(f"Vocabulary Size: {X_dense.shape[1]} features")
print(f"Training Samples: {X_train.shape[0]} | Testing Samples: {X_test.shape[0]}")
print(f"\n=== Final Baseline Report (Threshold = {best_thresholds}) ===")
# 1. Generate the report as a DICTIONARY instead of a string
report_dict = classification_report(
    y_test, 
    predictions_tuned, 
    target_names=mlb.classes_, 
    zero_division=0, 
    output_dict=True
)

# 2. Convert the dictionary into a Pandas DataFrame
report_df = pd.DataFrame(report_dict).transpose()

# 3. Print it as a CSV string using semicolons (;) as the separator
print("\n=== Copy and Paste into Google Sheets ===")
print(report_df.round(3).to_csv(sep=';'))
