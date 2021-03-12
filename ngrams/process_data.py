from ngrams import generate_ngrams
import simplejson as json

corpus = ''

with open('data/movies_text.txt') as f:
    corpus = f.read().lower()

print("loaded file")

ngrams = {}

for i in range(2,16):
    print(f"generating ngrams of length {i}")
    ngrams[i] = generate_ngrams(i, corpus)

with open('data/movies_processed.json', 'w') as f:
    json.dump(ngrams, f)