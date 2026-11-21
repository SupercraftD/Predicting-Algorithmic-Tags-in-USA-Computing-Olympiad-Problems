#Searches every subdirectory in raw_USACO_Guide/ to find the problem lists.
#Filters for problems sourced from the official usaco.org, and saves all problem data into every_problem.json

from pathlib import Path
import json

everyproblem = []

for path in Path("data/raw_USACO_Guide").rglob("*.problems.json"):
    if path.is_file():
        
        with open(path, 'r') as file:
            data = json.load(file)

            for i in data:
                if i=="MODULE_ID":continue
                else:

                    for prob in data[i]:
                        if prob["source"] in ["Bronze", "Silver", "Gold", "Platinum"]:
                            everyproblem.append(prob)

#also get the extra file of problems
with open("data/raw_USACO_Guide/extraProblems.json", 'r') as file:
    data = json.load(file)

    for prob in data:
        if prob["source"] in ["Bronze", "Silver", "Gold", "Platinum"]:
            everyproblem.append(prob)

print(everyproblem)
print(len(everyproblem))

tags = {}

for i in everyproblem:
    for t in i['tags']:
        if t in tags:
            tags[t]+=1
        else:
            tags[t]=1

print(tags)

with open("data/everyProblem.json", "w") as file:
    json.dump(everyproblem, file, indent=4)