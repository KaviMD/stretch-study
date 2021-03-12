from ngrams import generate_ngrams, cdf, pick_ngram
### Simple text generation

ngrams = generate_ngrams(10, corpus)

probabilities = cdf(ngrams)

for i in range(10):
    print(pick_ngram(probabilities))