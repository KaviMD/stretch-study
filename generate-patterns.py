import simplejson as json
from tqdm import tqdm
from copy import copy
from itertools import combinations
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import fsolve
import re
import pandas as pd

# %%
def count_tuple(lst, item):
    cnt = 0
    for i, j in lst:
        if j == item:
            cnt += 1
    return cnt

def graph_equation(formula, x_range, **kwargs):
    x = np.array(x_range)
    y = formula(x)
    plt.plot(x,y, **kwargs)
# %%
# Brute force pattern identification

d = {}

MINLEN = 2
MAXLEN = 100
MINCNT = 1

substrings = [s[x:y] for x, y in combinations(range(len(s) + 1), r = 2)]

print(f"Generated {len(substrings)} substrings")

for i in tqdm(range(len(substrings)), desc="Searching for patterns"):
    sub = substrings[i]
    l_sub = len(sub)
    if l_sub >= MINLEN and l_sub <= MAXLEN:
        if sub not in d:
            cnt = s.count(sub)
            if cnt >= MINCNT:
                d[sub] = cnt

with open('data/brute-force.json', 'w') as f:
    json.dump(d, f)
# %%
with open('data/brute-force.json', 'r') as f:
    patterns = json.load(f)

with open('data/mapping.json', 'r') as f:
    mapping = json.load(f)

inverted_char_mapping = {v: float(k) for k, v in mapping.items()}

# Sort patterns by how many times they were repeated
patterns_sorted = sorted(patterns.items(), key=lambda kv: kv[1])

print(f"Loaded {len(patterns_sorted)} patterns total")
#%%
# This code solves for a rational function that intersects 3 points
# It is used to remove patterns that don't appear often enough for their pattern length
# points is formated: [x,y]
points = [[2,150],[4,30],[9,2]]
def e(i):
    a,b,c = i
    r = []
    for p in points:
        r.append((a/(p[1]-b))-c-p[0])
    return r

a,b,c = fsolve(e, (1, 1, 1))
f = lambda x: (a/(x+c))+b


x = [len(k) for k, v in patterns_sorted]
y = [v for k,v in patterns_sorted]
col = []
for i in range(len(y)):
    if y[i] < f(x[i]):
        col.append((0.5,0.5,1,0.05))
    else:
        col.append((0,0.5,0,0.5))

plt.rcParams["figure.figsize"] = (20, 15)
plt.scatter(x, y, s=200, color=col)


extend_line = 0.1
graph_equation(f, np.arange(2-extend_line,12+extend_line,0.05), c="red", linewidth=2)

plt.xlabel('Pattern Length', fontsize=18)
plt.ylabel('Pattern Frequency', fontsize=18)
# %%
patterns_filtered = np.array([k for k,v in patterns_sorted])[y >= f(x)]
# %%
user_action_count = []
user_list = []
for user in data['users'].keys():
    user_action_count.append(len(data['users'][user]) + (user_action_count[-1] if user_action_count else 0))
    user_list.append(user)

action_char_list = "".join(simplified)

def locate_char_pattern(action_pattern):
    # Add a "+" after each character to make it a valid regex pattern
    re_pattern = ""
    to_escape = "[](){}*+?|^%.\\"
    for c in action_pattern:
        if c in to_escape:
            re_pattern += "\\"
        re_pattern += c + "+"
    
    return [(m.start(0), m.end(0)) for m in re.finditer(re_pattern, action_char_list)]

def find_pattern_user(pattern_location):
    pattern_start = pattern_location[0]
    user_index = 0
    for i in range(len(user_action_count)):
        if pattern_start < user_action_count[i]:
            user_index = i
            break
    if user_index > 0:
        pattern_start -= user_action_count[user_index-1]
    
    return (user_list[user_index], list(data['users'][user_list[user_index]].keys())[pattern_start])

final_patterns = []

for pattern in patterns_filtered:
    translated_pattern = ",".join([inverted_event_mapping[inverted_char_mapping[c]] for c in pattern])
    pattern_locations = locate_char_pattern(pattern)

    pattern_info = ""

    for loc in pattern_locations:
        pattern_info += ":".join(find_pattern_user(loc)) + ";"

    final_patterns.append([translated_pattern, pattern_info, len(pattern), s.count(pattern)])

df = pd.DataFrame(data=final_patterns, columns=['pattern','user-timestamp','pattern-length', 'pattern-frequency'])
df.to_csv("data/patterns.csv")
# %%