# %%
from nltk.lm import MLE
from nltk.util import ngrams
from nltk.lm.preprocessing import pad_both_ends, padded_everygram_pipeline

from sklearn.model_selection import KFold
import numpy as np
import pandas as pd

import simplejson as json
# %%
def split_chars(arr):
    return [c for c in arr]

# %%

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

with open('data/mapping.json', 'r') as f:
    char_mapping = json.load(f)

mode_change_char = char_mapping[str(event_mapping['ModeChange'])]

with open('data/grouped.json', 'r') as f:
    events_grouped = json.load(f)

with open('data/simplified_all.txt', 'r') as f:
    events_all = f.read().replace(mode_change_char, "")


# %%

# Each task has a different number of completions
#  Task 1: 16
#  Task 2: 10
#  Task 3: 9
#  Task 4: 7


# Get all of the completions for task 1
event_list = []
for user in events_grouped['4']:
    for completion in events_grouped['4'][user]:
        event_list.append(list(filter(lambda a: a != mode_change_char, completion)))


kf = KFold(n_splits = 5, shuffle=False) # 5 fold cross validation is used so that the test/train split is 20%

# %%
split = next(kf.split(event_list))

event_list = np.array(event_list)

train_data = event_list[split[0]]
test_data = event_list[split[1]]

n = 3
train_processced, train_vocab = padded_everygram_pipeline(n, train_data)

# %%
model = MLE(n)
model.fit(train_processced, train_vocab)

# %%
test_processced, test_vocab = padded_everygram_pipeline(n, test_data)

test_vocab_unique = set(list(test_vocab))

total_checked = 0
total_correct = 0

confidence_threshold = 0.5
threshold_checked = 0
threshold_correct = 0

for everygram in test_processced:
    # I think there is one everygram generated per input seqence
    for gram in everygram:
        # We only want grams of length n
        if len(gram) == n:
            history = list(gram[:n-1])
            answer = gram[-1]

            max_probability = -1
            max_gram = "failed"

            for g in test_vocab_unique:
                s = model.score(g, history)
                #print(gram, g, history, s)
                if s > max_probability:
                    max_probability = s
                    max_gram = g
            
            total_checked += 1

            if max_probability >= confidence_threshold:
                threshold_checked += 1

            if max_gram == answer:
                total_correct += 1
                if max_probability >= confidence_threshold:
                    threshold_correct += 1
            
            print(gram, max_gram, max_probability)

# %%
# Predict next actions based on a starting sequence using RNG
model.generate(2, text_seed=split_chars('mwm'))

# %%
# Calculate the probability of a specific gram, could be used for a lot more control
# https://stackoverflow.com/a/54979617

print(model.counts['m']) # i.e. Count('m')
print(model.counts[['m']]['w']) # i.e. Count('w'|'m')
print(model.counts[split_chars('mw')]['m']) # i.e. Count('m'|'mw')

print(model.score('w', 'm'))  # P('w'|'m')
print(model.score('m', split_chars('mw')))  # P('m'|'mw')

# %%
