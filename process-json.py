#%%
import simplejson as json
from tqdm import tqdm
from copy import copy
from itertools import combinations 
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
    'Task1Started': 0.0,
    'Task1Ended': 0.0,
    'Task2Started': 0.0,
    'Task2Ended': 0.0,
    'Task3Started': 0.0,
    'Task3Ended': 0.0,
    'Task4Started': 0.0,
    'Task4Ended': 0.0,
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
    'SetCameraView': -1.0,
    'SpeedChange': -1.0
}
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
MAXLEN = 10
MINCNT = 2

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

inverted_mapping = {v: k for k, v in mapping.items()}

print(patterns.keys())

# %%
