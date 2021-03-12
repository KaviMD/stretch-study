from ngrams import generate_ngrams, cdf, pick_ngram, filter_dict_keys

### Complex text generation (predict next n using last m grams)
    # basically ngrams help you pick the next n words with the highest probility given the previous n words
    # We could have the previous n be an anction list and the next n be on of our patterns
    # Could have everything be patterns or everything be action list
    # What should n be? it doesn't seem like we have enough data for it to be that large
    # Should I get rid of all of the view switches?

past_n_target = 6 # how far to try and look back
past_n_min = 2 # min amount to look back. if a matching ngram of this length is not found, the program will exit
forward_n = 1 # how many new grams to add each iteration
min_ngrams_needed = 2 # how many ngrams need to be found

all_ngrams = generate_ngrams(past_n_target+forward_n, corpus)

generated = ['the']

for i in range(0, 20):
    filtered_ngrams = {}
    temp_past_n = min(past_n_target, len(generated))
    while not filtered_ngrams:
        filtered_ngrams = filter_dict_keys(all_ngrams, generated[-temp_past_n:], starting_index=past_n_target-temp_past_n)
        print(generated[-temp_past_n:], filtered_ngrams, len(filtered_ngrams))

        temp_past_n -= 1
        if (temp_past_n < past_n_min) or (len(filtered_ngrams) < min_ngrams_needed):
            break

    if len(filtered_ngrams) >= min_ngrams_needed:
        probabilities = cdf(filtered_ngrams)

        chosen_gram = pick_ngram(probabilities)[-forward_n:]

        generated += chosen_gram
    else:
        if len(filtered_ngrams) == 0 : print(f"couldn't find any matching ngram of length {temp_past_n}. exiting after {i} iteration(s)")
        else: print(f"couldn't find {min_ngrams_needed} matching ngrams of length {temp_past_n}. exiting after {i} iteration(s)")
        break

print("\ngenerated text:")
print(" ".join(generated))