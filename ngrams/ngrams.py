from os import sep
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

def filter_dict_keys(dictionary, to_include, starting_index=0,separator=' ', inplace=False):
    to_filter = dictionary
    if not inplace:
        to_filter = dictionary.copy()
    
    keys = to_filter.keys()

    for key in list(keys):
        if not to_include == key.split(separator)[starting_index:len(to_include)+starting_index]:
            del to_filter[key]
    
    return to_filter

def pick_ngram(probabilities, separator = ' '):
    for i in range(len(probabilities)):
        if probabilities[i][0] > random.uniform(0, 1):
            return probabilities[i][1].split(separator)