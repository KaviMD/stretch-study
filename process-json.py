#%%
import simplejson as json
from tqdm import tqdm
from copy import copy
from itertools import combinations
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import fsolve
import re
import pandas as pd
#%%
def make_list_chars(arr):
    unique = list(set(arr))
    d = {}
    for i in range(len(unique)):     
        d[unique[i]] = chr(97+i)
    arr_modified = []
    for a in arr:
        arr_modified.append(d[a])
    return (d,arr_modified)

# From: https://www.geeksforgeeks.org/remove-consecutive-duplicates-string/
def removeDuplicates(str_input): 
    S = copy(str_input)
    n = len(S)  
      
    # We don't need to do anything for  
    # empty or single character string.  
    if (n < 2) : 
        return
          
    # j is used to store index is result  
    # string (or index of current distinct  
    # character)  
    j = 0
      
    # Traversing string  
    for i in range(n):  
          
        # If current character S[i]  
        # is different from S[j]  
        if (S[j] != S[i]): 
            j += 1
            S[j] = S[i]  
      
    # Putting string termination  
    # character.  
    j += 1
    S = S[:j] 
    return S

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
#%%
with open("data/stretch-teleop.json") as f:
    data = json.load(f)
#%%
'''
-1: setting changes
 0: session/tasks start/end
 1: look
 2: turn
 3: move
 4: lift
 5: arm
 6: wrist
 7: gripper
'''

event_mapping = {
    'SessionStarted': 0.0,
    'Task1Started': 0.1,
    'Task1Ended': 0.2,
    'Task2Started': 0.3,
    'Task2Ended': 0.4,
    'Task3Started': 0.5,
    'Task3Ended': 0.6,
    'Task4Started': 0.7,
    'Task4Ended': 0.8,
    'LookUp': 1.0,
    'LookRight': 1.1,
    'LookDown': 1.2,
    'LookLeft': 1.3,
    'TurnRight': 2.0,
    'TurnLeft': 2.1,
    'MoveForward': 3.0,
    'MoveBackward': 3.1,
    'LiftUp': 4.0,
    'LiftDown': 4.1,
    'ArmRetract': 5.0,
    'ArmExtend': 5.1,
    'WristIn': 6.0,
    'WristOut': 6.1,
    'GripperClose': 7.0,
    'GripperOpen': 7.1,
    'ModeChange': -1.0,
    'SetCameraView': -1.1,
    'SpeedChange': -1.2
}

inverted_event_mapping = {v: k for k, v in event_mapping.items()}
#%%

eventInfo = set()
eventName = set()

event_list = []

for user in data['users']:
    for event in data['users'][user]:
        event_data = data['users'][user][event]
        eventInfo.add(event_data['eventInfo'])
        eventName.add(event_data['eventName'])
        event_list.append(event_mapping[event_data['eventName']])
# %%
mapping, simplified = make_list_chars(event_list)

s = "".join(removeDuplicates(simplified))

with open('data/simplified.txt', 'w') as f:
    f.write(s)


with open('data/mapping.json', 'w') as f:
    json.dump(mapping, f)

print(eventInfo)
print(eventName)
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