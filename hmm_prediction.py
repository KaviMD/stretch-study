# %%
from hmmlearn import hmm

from sklearn.model_selection import KFold
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import simplejson as json

sns.set()
# %%
def split_chars(arr):
    return [c for c in arr]

def prepare_data(raw_data):
    processed_data = []
    for e_list in raw_data:
        for e in e_list:
            processed_data.append([ord(e)])

    processed_data_lengths = [len(e_list) for e_list in raw_data]

    return (processed_data, processed_data_lengths)

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

# Ngram hyperparameters
min_n_components = 2
max_n_components = 5 # 40
# Confidence Threshold hyperparameters
min_ct = -15
max_ct = 0
ct_step = 5
# Gram Length
# %%
ngram_results = []

for task in events_grouped:
    # Get data for this task
    event_list = []
    for user in events_grouped[task]:
        for completion in events_grouped[task][user]:
            event_list.append(list(filter(lambda a: a != mode_change_char, completion)))

    # Set hyperparameters
    for n_components in range(min_n_components, max_n_components):
        for confidence_threshold in np.arange(min_ct, max_ct, ct_step):
            # Run cross validation with teh given hyperparameters
            k = 5
            kf = KFold(n_splits = k, shuffle=True) # 5 fold cross validation is used so that the test/train split is 20%

            total_accuracy = 0
            total_threshold_accuracy = 0

            total_grams = 0
            total_above_threshold = 0

            for i, split in enumerate(kf.split(event_list)):
                event_list = np.array(event_list)

                train_data_raw = event_list[split[0]]
                test_data_raw = event_list[split[1]]

                train_data, train_data_lengths = prepare_data(train_data_raw)
                test_data, test_data_lengths = prepare_data(test_data_raw)

                vocab = set([v[0] for v in test_data])

                # Create and train model
                np.random.seed(42)
                model = hmm.GaussianHMM(n_components=n_components, covariance_type="full", n_iter=10000)
                model.fit(train_data, lengths=train_data_lengths)

                total_checked = 0
                total_correct = 0

                threshold_checked = 0
                threshold_correct = 0

                past_n = 3
                for gram in [test_data[i : i + past_n] for i in range(0, len(test_data))]:
                    # We only want grams of length past_n
                    if len(gram) == past_n:
                        history = list(gram[:past_n-1])
                        answer = gram[-1]

                        max_probability = -9999999999999999999999
                        max_gram = "failed"

                        for v in vocab:
                            s = model.score(history + [[v]])
                            if s > max_probability:
                                max_probability = s
                                max_gram = v

                        total_checked += 1

                        if max_probability >= confidence_threshold:
                            threshold_checked += 1

                        if max_gram == answer:
                            total_correct += 1
                            if max_probability >= confidence_threshold:
                                threshold_correct += 1
                        
                        #print(max_gram, max_probability, total_checked, threshold_correct)
                
                total_grams += total_checked
                total_above_threshold += threshold_checked

                try:
                    total_accuracy += total_correct / total_checked
                except ZeroDivisionError:
                    pass
                
                try:
                    total_threshold_accuracy += threshold_correct / threshold_checked
                except ZeroDivisionError:
                    pass
            ngram_results.append([task, n_components, confidence_threshold, total_accuracy/k, total_threshold_accuracy/k, total_grams, total_above_threshold])

# %%
split = next(kf.split(event_list))

event_list = np.array(event_list)

train_data_raw = event_list[split[0]]
test_data_raw = event_list[split[1]]

train_data, train_data_lengths = prepare_data(train_data_raw)
test_data, test_data_lengths = prepare_data(test_data_raw)

vocab = set([v[0] for v in test_data])

# %%
np.random.seed(42)

model = hmm.GaussianHMM(n_components=3, covariance_type="full", n_iter=10000)

# %%
model.fit(train_data, lengths=train_data_lengths)
# %%

history = [[118]]
answer = [[100]]

max_probability = -9999999999999999999999
max_gram = "failed"

for v in vocab:
    s = model.score(history + [[v]])
    print(v, s)
    if s > max_probability:
        max_probability = s
        max_gram = v
# %%
