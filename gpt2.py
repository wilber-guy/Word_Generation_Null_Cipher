# -*- coding: utf-8 -*-
"""
Created on Mon Nov 16 16:15:26 2020

@author: 17013
"""

from transformers import AutoModelForCausalLM, AutoTokenizer, top_k_top_p_filtering
import torch
from torch.nn import functional as F
from nltk.corpus import wordnet, words
import spacy

def run_me(secret='wilberscheid'):

    nlp = spacy.load("en_core_web_md")

    def get_syn(word):
        synonyms = []

        for syn in wordnet.synsets(str(word)):
            for l in syn.lemmas():
                synonyms.append(l.name())

        return list(set(synonyms))

    # to improve ability of model to write secrets within text, try and reduce letters
    # that do not appear as the first letter of a word

    # frequencies of letter as the first letter in a word
    # found at https://en.wikipedia.org/wiki/Letter_frequency
    # frequencies add up to .9026, but this was the only source I could find for these values
    # and for our use case it will be good enough
    letter_frequ = { 'a':.017, 'b':.044, 'c': .052, 'd': .032, 'e':.028, 'f':.04, 'g':.016, 'h':.042,
                     'i':.073, 'j':.0051, 'k':.0086, 'l':.024, 'm':.038, 'n':.023, 'o':.076, 'p':.043,
                     'q':.0022, 'r':.028, 's':.067, 't':.16, 'u':.012, 'v':.0082, 'w':.055, 'x':.00045,
                     'y':.0076, 'z':.00045}

    secret_word = str(secret)

    secret_word = secret_word.replace(' ', '')

    secret_score = 1

    for letter in secret_word:
        secret_score *= letter_frequ.get(letter)

    print('secret score (where higher score is better): ', secret_score * 10000000000000)


    tokenizer = AutoTokenizer.from_pretrained("gpt2-large")
    model = AutoModelForCausalLM.from_pretrained("gpt2-large", return_dict=True)

    sequence = f"Once upon a time"

    not_found = 0

    possible = 1

    for letter in secret_word:

        input_ids = tokenizer.encode(sequence, return_tensors="pt")

        # get logits of last hidden state
        next_token_logits = model(input_ids).logits[:, -1, :]

        # filter
        filtered_next_token_logits = top_k_top_p_filtering(next_token_logits, top_k=300)

        # sample
        probs = F.softmax(filtered_next_token_logits, dim=-1)

        next_token = torch.multinomial(probs, num_samples=300)

        list_of_words = tokenizer.decode(next_token.tolist()[0]).split()

        for x in list_of_words:
            if len(x) < 2 and x.lower() not in ['a', 'i']:
                list_of_words.remove(x)

        # check to see what words have the correct first letter and make sure these words are in a vocab
        list_of_words_with_letter = [w.lower() for w in list_of_words if (w[0] == letter and w in words.words())]

        # take every word and find synanoms then add them to a list if they start with the right letter
        for word in list_of_words:
            syns = get_syn(word)
            for syn in syns:
                if syn[0] == letter:
                    list_of_words_with_letter.append(syn)

        print()
        print(list_of_words_with_letter)

        possible *= len(list_of_words_with_letter)


        found = 0
        for word in list_of_words:
            # if the word genreated by the model fits our secret, add it
            if word[0].lower() == letter:
                print('word: ', word)
                sequence += (' ' + word)
                found = 1
                print('sequence: ', sequence)
                break

        #  use word net to look for synonyms for each word in word list
        if found == 0:
            for word in list_of_words:
                syn_words = get_syn(word)
                for syn in syn_words:
                    if syn[0].lower() == letter:
                        print('syn: ', syn, ' word: ', word)
                        sequence += (' ' + syn)
                        found = 1
                        print('sequence: ', sequence)
                        break
                if found == 1:
                    break

        if found == 0:
            print('did not find word that fit secret')
            print('#'*90)
            not_found += 1


    print(sequence)

    print('#'*90)

    print('not found: ', not_found)

    return sequence

run_me('wilber')