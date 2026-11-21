# This file processes TF-IDF, without lemmatization. It outputs the top words overall and per-tag
# This file does NOT train a model

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
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


corpus = [cleanlatex(t["text"]) for t in text]

base_stop_words = list(TfidfVectorizer(stop_words='english').get_stop_words())

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
    "units","clockwise","genome","factor","w_i","subscription","xx","s_j","a_j","b_j","a_5"
]

custom_stop_words = base_stop_words + usaco_noise

vectorizer = TfidfVectorizer(stop_words=custom_stop_words, max_df=0.75, min_df=2, ngram_range=(1,3))

tfidf_matrix = vectorizer.fit_transform(corpus)

feature_names = vectorizer.get_feature_names_out()
dense_matrix = tfidf_matrix.todense()

df = pd.DataFrame(dense_matrix, columns=feature_names)
print(df)

mean_scores = df.mean(axis=0)

top_words = mean_scores.sort_values(ascending=False)

print("--- Top 20 Words ---")
print(top_words.head(20))

for target_tag in sorted(list(everytag)):
    
    matching_indices = [i for i, t in enumerate(text) if target_tag in t["tags"]]
    
    tag_specific_matrix = df.iloc[matching_indices]
    
    mean_scores = tag_specific_matrix.mean(axis=0)
    top_words = mean_scores.sort_values(ascending=False).head(10)
    
    print(f"{target_tag}")
    print(top_words)