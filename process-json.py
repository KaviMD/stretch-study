#%%
import simplejson as json
from copy import copy, deepcopy
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

event_list_grouped = {1:{},2:{},3:{},4:{}}

for user in data['users']:
    print(user)
    current_task = 0
    for event in data['users'][user]:
        event_data = data['users'][user][event]
        eventInfo.add(event_data['eventInfo'])
        eventName.add(event_data['eventName'])
        event_list.append(event_mapping[event_data['eventName']])

        if event_data['eventName'] in ['Task1Started', 'Task2Started', 'Task3Started', 'Task4Started']:
            current_task = int(event_data['eventName'][4])
        
            if user in event_list_grouped[current_task]:
                event_list_grouped[current_task][user].append([])
            else:
                event_list_grouped[current_task][user] = [[]]
        
        if current_task > 0:
            event_list_grouped[current_task][user][-1].append(event_mapping[event_data['eventName']])
        
        if event_data['eventName'] in ['Task1Ended', 'Task2Ended', 'Task3Ended', 'Task4Ended']:
            current_task = 0
# %%
# Save action mappings and full user action set
mapping, simplified_all = make_list_chars(event_list)

s_all = "".join(removeDuplicates(simplified_all))
s_all_duplicates = "".join(simplified_all)

with open('data/simplified_all.txt', 'w') as f:
    f.write(s_all)

with open('data/simplified_all_duplicates.txt', 'w') as f:
    f.write(s_all_duplicates)

with open('data/mapping.json', 'w') as f:
    json.dump(mapping, f)

print(eventInfo)
print(eventName)
# %%
# Save grouped user actions

# Map the numbers to letters using the previously generated mapping
event_list_grouped_duplicates = deepcopy(event_list_grouped)

for task in event_list_grouped:
    for user in event_list_grouped[task]:
        for i in range(len(event_list_grouped[task][user])):
            no_duplicates = removeDuplicates([mapping[a] for a in event_list_grouped[task][user][i]])
            event_list_grouped[task][user][i] = no_duplicates

            mapped = [mapping[a] for a in event_list_grouped_duplicates[task][user][i]]
            event_list_grouped_duplicates[task][user][i] = mapped

with open('data/grouped.json', 'w') as f:
    json.dump(event_list_grouped, f)

with open('data/grouped_duplicates.json', 'w') as f:
    json.dump(event_list_grouped_duplicates, f)
# %%
