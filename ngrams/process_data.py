from ngrams import generate_ngrams
import simplejson as json
import multiprocessing
from functools import partial

def generate_ngrams_threaded(n, c):
    print("starting ngrams of length", n)
    ngrams = generate_ngrams(n, c)
    print("generated ngrams of length", n)
    return ngrams


corpus = ''

if __name__ == "__main__":
    multiprocessing.freeze_support()

    with open('data/movies_text.txt') as f:
        corpus = f.read().lower()
        print(len(corpus))

    print("loaded file")

    p = multiprocessing.Pool()

    generate_ngrams_threaded_partial = partial(generate_ngrams_threaded, c=corpus)

    ngrams = p.map(generate_ngrams_threaded_partial, range(2,33))

    with open('data/movies_processed.json', 'w') as f:
        json.dump(ngrams, f)