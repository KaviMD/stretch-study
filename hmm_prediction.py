# %%
from hmmlearn import hmm

from sklearn.model_selection import KFold
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LogNorm

import simplejson as json
import math

sns.set()
# %%
def split_chars(arr):
    return [c for c in arr]

def prepare_data(raw_data):
    processed_data = []
    for e_list in raw_data:
        for e in e_list:
            processed_data.append([ord(e)-97])

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

# HMM hyperparameters
n_components_range = [7, 10, 15, 20, 25, 28]#range(3,30)#
# Confidence Threshold hyperparameters
ct_range = [-15, -10, -6, 3] #np.arange(-10, -2, 0.25) #
# Gram Length
# %%
hmm_results = []

for task in events_grouped:
    # Get data for this task
    event_list = []
    for user in events_grouped[task]:
        for completion in events_grouped[task][user]:
            event_list.append(list(filter(lambda a: a != mode_change_char, completion)))

    # Set hyperparameters
    for n_components in n_components_range:
        for confidence_threshold in ct_range:
            # Run cross validation with teh given hyperparameters
            k = 5
            kf = KFold(n_splits = k, shuffle=True) # 5 fold cross validation is used so that the test/train split is 20%

            total_accuracy = 0
            total_threshold_accuracy = 0

            total_grams = 0
            total_above_threshold = 0
            total_threshold_correct = 0

            for i, split in enumerate(kf.split(event_list)):
                event_list = np.array(event_list)

                train_data_raw = event_list[split[0]]
                test_data_raw = event_list[split[1]]

                train_data, train_data_lengths = prepare_data(train_data_raw)
                test_data, test_data_lengths = prepare_data(test_data_raw)


                train_vocab = set([v[0] for v in train_data])
                test_vocab = set([v[0] for v in test_data])

                # Create and train model
                np.random.seed(42)
                model = hmm.MultinomialHMM(n_components=n_components, n_iter=10000, tol=0.0001, init_params='e')

                pi = np.random.rand(n_components)
                pi /= pi.sum()
                model.startprob_ = pi
                
                A = np.random.rand(n_components, n_components)
                A /= A.sum(axis=1,keepdims=1)
                model.transmat_ = A
                
                model.fit(train_data, lengths=train_data_lengths)

                #print(model.startprob_, model.transmat_, model.emissionprob_)

                total_checked = 0
                total_correct = 0

                threshold_checked = 0
                threshold_correct = 0

                past_n = 2
                for gram in [test_data[i : i + past_n] for i in range(0, len(test_data))]:
                    # We only want grams of length past_n
                    if len(gram) == past_n:
                        history = list(gram[:past_n-1])
                        answer = gram[-1]

                        max_probability = -9999999999999999999999
                        max_gram = "failed"

                        for v in test_vocab:
                            s = model.score(history + [[v]])
                            #print(gram, history + [[v]], s)
                            if s > max_probability:
                                max_probability = s
                                max_gram = v

                        total_checked += 1

                        if max_probability >= confidence_threshold:
                            threshold_checked += 1

                        if max_gram == answer[0]:
                            total_correct += 1
                            if max_probability >= confidence_threshold:
                                threshold_correct += 1
                        
                        #print(answer[0], max_gram, max_probability, total_checked, total_correct)
                
                total_grams += total_checked
                total_above_threshold += threshold_checked

                total_accuracy += total_correct / total_checked
                try:
                    total_threshold_accuracy += threshold_correct / threshold_checked
                except ZeroDivisionError:
                    pass

                total_threshold_correct += threshold_correct
            
            hmm_results.append([task, n_components, confidence_threshold, total_accuracy/k, total_threshold_accuracy/k, total_grams/k, total_above_threshold/k, total_threshold_correct/k])

 # %%
df = pd.DataFrame(hmm_results, columns=['task_number', 'n_components', 'confidence_threshold', 'total_accuracy', 'total_threshold_accuracy', 'total_grams', 'total_above_threshold', 'total_threshold_correct'])
df.to_csv('data/hmm_results.csv', index=False)
df.head(100)

#%%
df = pd.read_csv("data/hmm_results.csv")
df['total_threshold_incorrect'] = df['total_above_threshold'] - df['total_threshold_correct']
df['normalized_accuracy'] = df['total_threshold_correct'] / df['total_threshold_incorrect']

df.to_csv('data/hmm_results.csv', index=False)
# %%
r = 4
c = 4
fig, big_axes = plt.subplots(nrows=r, ncols=1, figsize=(30, 25))
plt.subplots_adjust(hspace=0.4)

for row, big_ax in enumerate(big_axes, start=1):
    big_ax.set_title(f"Task {row}", fontsize=16,  y=1.08)

    # Turn off axis lines and ticks of the big subplot 
    # obs alpha is 0 in RGBA string!
    big_ax.tick_params(labelcolor=(1.,1.,1., 0.0), top='off', bottom='off', left='off', right='off')
    # removes the white frame
    big_ax._frameon = False

for i in range(0, 4):
    filtered_data = df[df.task_number == i+1].round({'confidence_threshold': 2})

    n_confidence_accuracy = filtered_data.pivot(index='n_components', columns='confidence_threshold', values='total_threshold_accuracy')
    n_confidence_removed = filtered_data.pivot(index='n_components', columns='confidence_threshold', values='total_above_threshold')

    #log_norm = LogNorm(vmin=n_confidence_removed.min().min(), vmax=n_confidence_removed.max().max())
    #cbar_ticks = [math.pow(10, i) for i in range(math.floor(math.log10(n_confidence_removed.min().min())), 1+math.ceil(math.log10(n_confidence_removed.max().max())))]


    g = sns.heatmap(n_confidence_accuracy, ax=fig.add_subplot(r,c,i*c+1))
    g.set_title('# of Hidden States & Confidence Threshold vs. Accuracy')
    g.set_xticklabels(g.get_xticklabels(), rotation=45)
    g.set_ylabel('# of Hidden States')
    g.set_xlabel('Confidence Threshold')

    h = sns.heatmap(n_confidence_removed, ax=fig.add_subplot(r,c,i*c+2)) #, norm=log_norm, cbar_kws={"ticks": cbar_ticks})
    h.set_title('# of Hidden States & Confidence Threshold vs. # of Predictions Made')
    h.set_xticklabels(h.get_xticklabels(), rotation=45)
    h.set_ylabel('# of Hidden States')
    h.set_xlabel('Confidence Threshold')

    
    j = sns.scatterplot(data=filtered_data, x='total_threshold_accuracy', y='total_threshold_correct', hue='n_components', ax=fig.add_subplot(r,c,i*c+3))
    j.set_title('Accuracy vs. # of Correct Predictions Made')
    j.set_xlabel('Accuracy')
    j.set_ylabel('# of Correct Predictions Made')
    j.set_xlim((0, 1))

    
    k = sns.scatterplot(data=filtered_data, x='total_threshold_accuracy', y='total_threshold_incorrect', hue='n_components', ax=fig.add_subplot(r,c,i*c+4))
    k.set_title('Accuracy vs. # of Incorrect Predictions Made')
    k.set_xlabel('Accuracy')
    k.set_ylabel('# of Incorrect Predictions Made')
    k.set_xlim((0, 1))

plt.show()
fig.savefig("data/hmm_results.png")
# %%
