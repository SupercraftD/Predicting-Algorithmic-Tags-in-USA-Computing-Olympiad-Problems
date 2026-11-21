#Current all unique tags and their frequencies:

# {'Stack': 7, 'Binary Search': 32, 'Greedy': 35, 'DSU': 13, 'Sorting': 41, 'Range DP': 5, 'Deque': 3, 'Complete Search': 29, 
# 'Prefix Sums': 29, 'DP': 61, 'BFS': 13, 'TopoSort': 2, 'Strings': 2, 'Set': 7, 'Binary Jumping': 1, '2D Prefix Sums': 5, 'Priority Queue': 7, 
# 'Connected Components': 11, 'Meet in the Middle': 1, '2P': 8, 'Geometry': 4, 'Cactus': 1, 'Small to Large': 1, 'Euler Tour': 5, 'PURS': 10, 
# 'Merging': 4, 'Indexed Set': 1, 'Combinatorics': 6, 'Tree': 26, 'LCA': 6, 'HLD': 4, 'Binary Lifting': 3, 'BBST': 1, 'Simulation': 27, 
# 'PIE': 5, 'MST': 5, 'Flood Fill': 13, 'Recursion': 3, 'Permutation': 3, 'Graph': 1, 'Amortized Analysis': 1, 'Sorted Set': 11, 'Knapsack': 6, 
# 'Prime Factorization': 2, 'Game': 2, 'SP': 6, 'Trie': 1, 'SegTree': 4, 'Rectangle': 4, 'Bitset': 5, '1DRQ': 4, 'Convex': 7, 'Bitmasks': 3, 
# 'Exponentiation': 3, 'Bitmask DP': 1, 'Probability': 1, 'Math': 2, 'Inversions': 4, 'Suffix Array': 1, 'Two Pointers': 3, 'Prefix Sum': 1, 
# 'Sliding Window': 4, 'D&C': 3, 'Graphs': 4, 'Spanning Tree': 2, 'Functional Graph': 3, 'DFS': 3, 'Coloring': 2, 'Pigeonhole': 1, 'Constructive': 2, 
# 'APSP': 2, 'Linked List': 4, 'Matrix': 4, 'Bitmask': 1, 'Sqrt': 6, 'Hashing': 3, 'Game Theory': 1, 'NT': 2, 'Bitwise': 5, 'Combo': 1, 'Map': 4, 
# 'Modular Arithmetic': 1, 'Casework': 5, "Euler's Formula": 3, 'Broken Profile': 1, 'Lazy SegTree': 4, 'Determinant': 1, 'Max Subarray Sum': 1, 
# 'Shortest Path': 2, 'Diameter': 1, 'Centroid': 2, 'BCC': 1, '2DRQ': 2, 'BIT': 2, 'Cycle': 1, 'Slope Trick': 1, 'Toposort': 1, 'PURQ': 6, 'Ad-Hoc': 1,
#  'Bipartite': 2, 'Sweep Line': 1, 'Maps': 1, 'Subsets': 1, 'SCC': 1, 'Cycles': 1, 'Generating Functions': 1, 'Ternary Search': 1}

#this file tries to merge related tags together so we have enough for the ML model to train off
#overwrites everyProblem.json

import json

mapsTo = {

    'DP': 'Dynamic Programming',
    'Range DP': 'Dynamic Programming',
    'Knapsack': 'Dynamic Programming',
    'Bitmask DP': 'Dynamic Programming',
    'Broken Profile': 'Dynamic Programming',
    'Slope Trick': 'Dynamic Programming',
    'Max Subarray Sum': 'Dynamic Programming',

    'Tree': 'Graphs & Trees',
    'MST': 'Graphs & Trees',
    'Spanning Tree': 'Graphs & Trees',
    'Bipartite': 'Graphs & Trees',
    'Connected Components': 'Graphs & Trees',
    'DSU': 'Graphs & Trees',
    'SP': 'Graphs & Trees',
    'APSP': 'Graphs & Trees',
    'TopoSort': 'Graphs & Trees',
    'Diameter': 'Graphs & Trees',
    'Centroid': 'Graphs & Trees',
    'Functional Graph': 'Graphs & Trees',
    'LCA': 'Graphs & Trees',
    'Flood Fill': 'Graphs & Trees',
    'HLD': 'Graphs & Trees',
    'Euler Tour': 'Graphs & Trees',
    'Cycles': 'Graphs & Trees',
    'Cycle': 'Graphs & Trees',
    'BFS': 'Graphs & Trees',
    'DFS': 'Graphs & Trees',
    'SCC': 'Graphs & Trees',
    'BCC': 'Graphs & Trees',
    'Graph': 'Graphs & Trees',
    'Graphs': 'Graphs & Trees',
    'Coloring': 'Graphs & Trees',
    'Binary Jumping': 'Graphs & Trees',
    'Small to Large': 'Graphs & Trees',
    'Binary Lifting': "Graphs & Trees",
    'Shortest Path': 'Graphs & Trees',
    'Toposort': 'Graphs & Trees',
    'Cactus':'Graphs & Trees',

    'PURQ': 'Data Structures',
    'PURS': 'Data Structures',
    '2DRQ': 'Data Structures',
    'BIT': 'Data Structures',
    'Lazy SegTree': 'Data Structures',
    'Trie': 'Data Structures',
    'Priority Queue': 'Data Structures',
    'Deque': 'Data Structures',
    'Stack': 'Data Structures',
    'Linked List': 'Data Structures',
    'Set': 'Data Structures',
    'Sorted Set': 'Data Structures',
    'Indexed Set': 'Data Structures',
    'Map': 'Data Structures',
    'Maps': 'Data Structures',
    'SegTree': 'Data Structures',
    'BBST': 'Data Structures',
    '1DRQ': 'Data Structures',

    'Math': 'Math & Combinatorics',
    'Combinatorics': 'Math & Combinatorics',
    'Combo': 'Math & Combinatorics',
    'Probability': 'Math & Combinatorics',
    'Prime Factorization': 'Math & Combinatorics',
    'Exponentiation': 'Math & Combinatorics',
    'NT': 'Math & Combinatorics',
    'Modular Arithmetic': 'Math & Combinatorics',
    'PIE': 'Math & Combinatorics',
    'Generating Functions': 'Math & Combinatorics',
    'Matrix': 'Math & Combinatorics',
    'Permutation': 'Math & Combinatorics',
    'Determinant': 'Math & Combinatorics',
    'Pigeonhole': 'Math & Combinatorics',


    'Sorting': 'Search & Sort',
    'Binary Search': 'Search & Sort',
    'Ternary Search': 'Search & Sort',
    'Complete Search': 'Search & Sort',
    'Sqrt': 'Search & Sort',
    'Meet in the Middle': 'Search & Sort',
    'D&C': 'Search & Sort',
    'Inversions': 'Search & Sort',

    'Two Pointers': 'Two Pointers & Sliding Window',
    '2P': 'Two Pointers & Sliding Window',
    'Sliding Window': 'Two Pointers & Sliding Window',
    'Prefix Sums': 'Two Pointers & Sliding Window',
    'Prefix Sum': 'Two Pointers & Sliding Window',
    '2D Prefix Sums': 'Two Pointers & Sliding Window',
    'Merging': 'Two Pointers & Sliding Window',

    'Geometry': 'Geometry',
    'Convex': 'Geometry',
    'Sweep Line': 'Geometry',
    "Euler's Formula": 'Geometry',
    'Rectangle': 'Geometry',

    'Strings': 'Strings',
    'Suffix Array': 'Strings',
    'Hashing': 'Strings',

    'Bitwise': 'Bit Manipulation',
    'Bitmasks': 'Bit Manipulation',
    'Bitset': 'Bit Manipulation',
    'Subsets': 'Bit Manipulation',
    'Bitmask': 'Bit Manipulation',

    'Greedy': 'Greedy & Implementation',
    'Simulation': 'Greedy & Implementation',
    'Ad-Hoc': 'Greedy & Implementation',
    'Constructive': 'Greedy & Implementation',
    'Casework': 'Greedy & Implementation',
    'Recursion': 'Greedy & Implementation',
    'Game Theory': 'Greedy & Implementation',
    'Game': 'Greedy & Implementation',
}

source = "data/everyProblem.json"
dest = "data/everyProblem.json"


data = []
with open(source,"r") as file:
    data = json.load(file)

for prob in data:
    realtags = set()
    for t in prob["tags"]:
        if t in mapsTo:
            realtags.add(mapsTo[t])
    prob["tags"] = list(realtags)

with open(dest, "w") as file:
    json.dump(data, file, indent=4)