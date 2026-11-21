import json
import copy

data=[]

with open("normalized_problems_with_constraints.json","r") as file:
    data = json.load(file)

for tag in ["Bit Manipulation","Data Structures","Dynamic Programming","Geometry","Graphs & Trees","Greedy & Implementation","Math & Combinatorics","Search & Sort","Strings","Two Pointers & Sliding Window"]:
    binaried = copy.deepcopy(data)

    for prob in binaried:
        if tag in prob["tags"]:
            prob["is_target_tag"] = True
        else:
            prob["is_target_tag"] = False

    with open(f"separate_datasets/{tag}.json","w") as file:
        json.dump(binaried, file, indent=4)
