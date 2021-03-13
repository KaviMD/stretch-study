# %%
from nltk.lm import MLE
from nltk.util import ngrams
from nltk.lm.preprocessing import pad_both_ends, padded_everygram_pipeline

import simplejson as json
# %%
def split_chars(arr):
    return [c for c in arr]

# %%

with open('data/simplified_all.txt', 'r') as f:
    events_all = f.read()

with open('data/grouped.json', 'r') as f:
    events_grouped = json.load(f)

# %%
# Get all of the completions for task 1
event_list = []
for user in events_grouped['1']:
    for completion in events_grouped['1'][user]:
        event_list.append(completion)
# %%
n = 3
train, vocab = padded_everygram_pipeline(n, event_list)
# %%
model = MLE(n)
model.fit(train, vocab)
# %%
# Predict next actions based on a starting sequence using RNG
model.generate(2, text_seed=split_chars('mwm'))

# %%
# Calculate the probability of a specific gram, could be used for a lot more control
# https://stackoverflow.com/a/54979617

print(model.counts['m']) # i.e. Count('m')
print(model.counts[['m']]['w']) # i.e. Count('w'|'m')
print(model.counts[split_chars('mw')]['m']) # i.e. Count('w'|'mw')

print(model.score('w', 'm'))  # P('w'|'m')
print(model.score('m', split_chars('mw')))  # P('m'|'mw')

# %%
