#Takes data from everyProblem.json and deduplicates problems, merging tags
#gets rid of problems with 0 tags

import json

data = {}

with open("data/everyProblem.json", "r") as file:
    data = json.load(file)

uniqueIds = set()
for prob in data:
    uniqueIds.add(prob["uniqueId"])

newdata = []

tagfreq = {}

for id in uniqueIds:

    probinfo = {}
    uniontags = []

    for prob in data:
        if prob["uniqueId"] == id:
            uniontags += prob["tags"]
            probinfo = prob
    
    probinfo["tags"] = uniontags

    if len(uniontags)==0:
        continue

    for tag in uniontags:
        if tag in tagfreq:
            tagfreq[tag]+=1
        else:
            tagfreq[tag]=1

    newdata.append(probinfo)

print(len(newdata))
print(tagfreq)

with open("data/everyProblem.json", "w") as file:
    json.dump(newdata, file, indent=4)