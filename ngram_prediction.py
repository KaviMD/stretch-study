# %%
from nltk.lm import MLE
from nltk.util import ngrams
from nltk.lm.preprocessing import pad_both_ends, padded_everygram_pipeline

import simplejson as json
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
lm = MLE(n)
len(lm.vocab)
lm.fit(train, vocab)
# %%
lm.generate(2, text_seed=['m', 'w', 'm'])