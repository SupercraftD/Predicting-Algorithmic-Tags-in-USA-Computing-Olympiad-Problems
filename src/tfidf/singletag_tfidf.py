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
    "units","clockwise","genome","factor","w_i","subscription","xx","s_j","a_j","b_j","a_5",

    "integer","wants","game","ith","case","favorite","example","single","fields","grass","pasture","pastures","farms",
    "conditions","satisfied","a_n","a_2","33","23","27","18","11","bovine","cowcow","dairies","clever","apothecowry","problem","answer","exists","16","31","15",
    "moo","manure","friendly","unfriendly","mowing","moos","mailboxes","id","team","fullness","snakes","teams","city","d_i",
    "videos","friend","x_i","friends","video","canvas","photo","pour","stalls","milking","cowbasic","b_1","program","return","b_n","17","20",
    "spottiness","th","does","animal","tommy","amy","standingout","like","cook","hooveskeeping","hoof","scissors","coats","painted","fort"
]



corpus = [cleanlatex(t["text"]) for t in text if "Graphs & Trees" in t["tags"] and len(t["tags"])==1]
print(len(corpus))
custom_stop_words = base_stop_words + usaco_noise

vectorizer = TfidfVectorizer(stop_words=custom_stop_words, ngram_range=(1,1))

tfidf_matrix = vectorizer.fit_transform(corpus)

feature_names = vectorizer.get_feature_names_out()
dense_matrix = tfidf_matrix.todense()

df = pd.DataFrame(dense_matrix, columns=feature_names)
print(df)

mean_scores = df.mean(axis=0)

top_words = mean_scores.sort_values(ascending=False)

print("--- Top 20 Words ---")
print(top_words.head(20))
