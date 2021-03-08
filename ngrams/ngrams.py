import random

# Based on the techniques described here: https://web.archive.org/web/20150908034624/https://lagunita.stanford.edu/c4x/Engineering/CS-224N/asset/slp4.pdf

def count_conscutive(search, lst):
    total_count = 0
    found_indices = []

    # Find all occurnces of the first search item in the list
    start_indices = [i for i, x in enumerate(lst) if x == search[0]]

    for i in start_indices:
        found_repeat = True
        for j in range(len(search)):
            if ((i+j >= len(lst)) or (lst[i+j] != search[j])):
                found_repeat = False
        if found_repeat:
            total_count += 1
            found_indices.append(i)
    return float(total_count), found_indices

def mle(wn: str, wn1: list, corpus: str, separator:str = " "):
    gram_list = corpus.split(separator)
    
    p_wn_wn1 = count_conscutive(wn1+[wn], gram_list)[0]
    p_wn1 = count_conscutive(wn1, gram_list)[0]
    return p_wn_wn1/p_wn1

def generate_ngrams(n, corpus, separator = ' '):
    corpus = corpus.split(separator)

    ngrams = {}

    for i in range(len(corpus) - n):
        ngram = corpus[i: i + n]
        ngrame_name = ' '.join(ngram)

        if ngrame_name not in ngrams:
            ngrams[ngrame_name] = count_conscutive(ngram, corpus)[0]
    
    return ngrams

def cdf(ngrams):
    total_count = sum(ngrams.values())

    current_probability = 0
    probabilities = []

    for ngram in ngrams:
        current_probability += ngrams[ngram] / total_count
        probabilities.append((current_probability, ngram))
    
    return probabilities

corpus = ''

with open('data/movies_text.txt') as f:
    corpus = f.read()[1:100_000]

### Simple text generation
ngrams = generate_ngrams(10, corpus)

probabilities = cdf(ngrams)

for i in range(10):
    r = random.uniform(0, 1)

    for i in range(len(probabilities)):
        if probabilities[i][0] > r:
            print(probabilities[i][1])
            break

# basically ngrams help you pick the next n words with the highest probility given the previous n words
# We could have the previous n be an anction list and the next n be on of our patterns
# Could have everything be patterns or everything be action list
# What should n be? it doesn't seem like we have enough data for it to be that large
# Should I get rid of all of the view switches?