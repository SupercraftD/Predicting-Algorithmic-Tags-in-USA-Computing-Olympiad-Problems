# Trains separate binary classifiers for each of the 10 tags.
# Trains each model on the entire N=401 dataset, and evaluates them using the N=40 held-out test set
# Reports each model's metrics (does not save the models anywhere to file)


import json
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer
import numpy as np
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

usaco_noise = [
    'cow', 'cows', 'bessie', 'farm', 'barn', 'barns', 'farmer', 'john', 'fj', 'field',
    'input', 'output', 'sample', 'format', 'line', 'file', 'number', 'time',"elsie",
    "000","test","a_i","b_i","brian","dean", "benjamin","qi", "dhruv","rohatgi",
    "gift","breed","breeds","milk","haybales","bucket","buckets","mildred","300","t_i",
    "l_i","p_i","magazine","herd","spotty","10","nhoj","censor","a_","s_i",
    "lines","contain","space","separated","integers","print","terminal","stdin","stdout","constraints","contains",
    "conveniently","numbered","cases","satisfy","determine","help","scoring","make","sure","100","cowvid","19","12","a_i","r_i",
    "milked","a_1","order","possible","second","1000","additional","inputs","1001","101","10inputs","5s","75","80","8inputs","twice","default",
    "512mb","21","24","250",

    "bella","slingshot","moonies","moonet","buddy","billboard","fence","pool","beam","paint","flowers","flower","gallons","robot","friendship",
    "meal","flavor","diamonds","pie","ball","airport","session","box","hungry","machine","pours","stars","stall","room","mounting","stopped","arrive",
    "units","clockwise","genome","factor","w_i","subscription","xx","s_j","a_j","b_j","a_5",

    "integer","wants","game","ith","case","favorite","example","single","fields","grass","pasture","pastures","farms",
    "conditions","satisfied","a_n","a_2","33","23","27","18","11","bovine","cowcow","dairies","clever","apothecowry","problem","answer","exists","16","31","15",
    "moo","manure","friendly","unfriendly","mowing","moos","mailboxes","id","team","fullness","snakes","teams","city","d_i",
    "videos","friend","x_i","friends","video","canvas","photo","pour","stalls","milking","cowbasic","b_1","program","return","b_n","17","20",
    "spottiness","th","does","animal","tommy","amy","standingout","like","cook","hooveskeeping","hoof","scissors","coats","painted","fort",
    "want","relevance",
    "cowphabet","air","conditioner", "ice","cream","moonie","buttercup","corn","sprinkler","particle"
]
lemmatizer = WordNetLemmatizer()

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


possible_tags = ["Bit Manipulation", "Data Structures", "Dynamic Programming", "Geometry", "Graphs & Trees", "Greedy & Implementation", "Math & Combinatorics",
                 "Search & Sort", "Strings", "Two Pointers & Sliding Window"]

data = []

with open(f"data/training.json","r") as file:
    data = json.load(file)

print("Pre-lemmatizing problem texts...")
for item in data:
    item["processed_text"] = pre_lemmatize_text(item["text"])
print("Done lemmatizing")

y_cols = []
for current_tag in range(10):
    y_cols.append([int(possible_tags[current_tag] in d["tags"]) for d in data])


Y = np.column_stack(y_cols)

base_stop_words = list(TfidfVectorizer(stop_words='english').get_stop_words())

df = pd.DataFrame([{
    'processed_text': item.get("processed_text", ""),
    'CONSTRAINT_MAX': np.log10(max(item["constraints"]) + 1) if len(item.get("constraints",[]))>0 else 0.0,
    'CONSTRAINT_MIN': np.log10(min(item["constraints"]) + 1) if len(item.get("constraints",[]))>0 else 0.0,
    'CONSTRAINT_COUNT': len(item.get("constraints", []))
} for item in data])


with open("data/testing.json", "r") as f:
    testdata = json.load(f)

testdata = [i for i in testdata if len(i["tags"])>0]

print(f"Loaded {len(testdata)} test problems...")

rows = []
y_true_labels = []

for item in testdata:
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

df_test = pd.DataFrame(rows)
y_test_cols = []
for current_tag in range(10):
    y_test_cols.append([int(possible_tags[current_tag] in d["tags"]) for d in testdata])


Y_true = np.column_stack(y_test_cols)



#Training
custom_stop_words = base_stop_words + usaco_noise

preprocessor = ColumnTransformer(
    transformers=[
        ('text', TfidfVectorizer(
            stop_words=custom_stop_words,
            ngram_range=(1, 2),
            min_df=2,
            sublinear_tf=True
        ), "processed_text"),
        
        ('num', StandardScaler(), ["CONSTRAINT_MAX", "CONSTRAINT_MIN", "CONSTRAINT_COUNT"])
    ]
)

predictionsForEach = []

for i in range(10):


    clf = LogisticRegression(class_weight="balanced", solver="liblinear", random_state=42)
    #clf = LinearSVC(class_weight="balanced", dual="auto", random_state=42,max_iter=5000)

    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('clf', clf)
    ])

    pipeline.fit(df, Y[:,i])


    Y_pred = pipeline.predict(df_test)
    predictionsForEach.append(Y_pred)

    # 5. Generate and Print Metrics Report
    print(f"{possible_tags[i]}")
    print(
        classification_report(
            Y_true[:,i],
            Y_pred,
            zero_division=0,
            digits=4,
        )
    )

    print("Accuracy:",accuracy_score(Y_true[:,i], Y_pred))

    print()