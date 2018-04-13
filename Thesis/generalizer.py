# Chester Palen-Michel
# Thesis
# 4/11/2018
from collections import Counter, defaultdict

import os

import pickle
from nltk.corpus import wordnet as wn


class Generalizer:
    def __init__(self):
        self.wordpair_words = []
        self.chosen_ones = Counter()
        self.syn_word_counts = defaultdict(Counter)
        self.words = []
        self.general_map = {}
        self.candidates = []

    def load_wordpairs(self, filepath):
        print('opening wordpairs')
        wordpairs_f = open(filepath, 'r')
        lines = wordpairs_f.readlines()
        for line in lines:
            words = line.split()
            for w in words:
                self.wordpair_words.append(w.strip().split('_')[-1])

    def load_words(self, filepath):
        print('loading words')
        word_file = open(filepath, 'r')
        for line in word_file.readlines():
            if not line.isspace():
                self.words.append(line.strip().split('_')[-1])

    def initialize_dict(self):
        for word in self.words:
            synsets = wn.synsets(word, pos=wn.NOUN)
            if synsets:
                syn = synsets[0]
                self.syn_word_counts[syn.name()][word] += 1
                self.propagate(syn, word)

    def propagate(self, syn, word):
        paths = syn.hypernym_paths()
        for path in paths:
            for synset in path[-2::-1]:
                self.syn_word_counts[synset.name()][word] += 1

    def create_mapping(self):
        for word in set(self.words):
            self.general_map[word] = self.top_match(word)

    def top_match(self, word):
        # for chosen ones if word in chosen ones: chosen_one[word] / sum([chosen_one[key] for key in chosen_one])
        # sort
        scored_chosen_ones = [(chosen_one, self.score(chosen_one, word)) for chosen_one in self.candidates
                              if word in self.syn_word_counts[chosen_one]]
        # scored_chosen_ones = []
        # for chosen_one in self.candidates:
        #     if word in self.syn_word_counts[chosen_one]:
        #         scored_chosen_ones.append((chosen_one, self.score(chosen_one, word)))
        scored_chosen_ones.sort(key=lambda x : x[1], reverse=True)
        return scored_chosen_ones[0] if scored_chosen_ones else None # take highest value

    def generalize(self, word):
        # return a semantic category
        if word in self.general_map:
            if self.general_map[word] is not None:
                return self.general_map[word][0]
            else:
                return None
        else:
            return None

    def make_candidates(self):
        hyper = lambda s: s.hypernyms()
        for head in self.wordpair_words:
            synset_list = wn.synsets(head, pos=wn.NOUN)
            if synset_list:
                syn = synset_list[0]
                self.chosen_ones.update([s.name() for s in list(syn.closure(hyper))[-7:]])
        self.candidates = [co[0] for co in self.chosen_ones.most_common(150)]


    def score(self, chosen_one, word):
        return float(self.syn_word_counts[chosen_one][word]) / \
               sum([self.syn_word_counts[chosen_one][key] for key in self.syn_word_counts[chosen_one]])

    def pickle(self):
        pic_file =open('../Resources_Thesis/general_map_single.pkl', 'wb')
        pickle.dump(self.general_map, pic_file)

    def load_pickle(self, path):
        pickle_file = open(path, 'rb')
        self.general_map = pickle.load(pickle_file)

    def generalize_wordpairs(self, filepath, outfile):
        wordpairs = open(filepath, 'r')
        gen_wordpairs = open(outfile, 'w')
        lines = wordpairs.readlines()
        for line in lines:
            words = line.split()
            new_worpair = []
            for word in words:
                generalized = self.generalize(word.strip().split('_')[-1])
                if generalized:
                    new_worpair.append(generalized)
                else:
                    new_worpair.append('null')
            print(new_worpair)
            gen_wordpairs.write('\t'.join(new_worpair) + '\n')

if __name__ == '__main__':
    generalizer = Generalizer()
    if os.path.isfile('../Resources_Thesis/general_map_single.pkl'):
        generalizer.load_pickle('../Resources_Thesis/general_map_single.pkl')
        generalizer.generalize_wordpairs('../Resources_Thesis/concattenated/wordpairs.txt', '../Resources_Thesis/concattenated/gen_wordpairs.txt' )
    else:
        generalizer.load_wordpairs('../Resources_Thesis/onto_results_50/wordpairs.txt')
        generalizer.load_words('../Resources_Thesis/dumped_extraction_words.txt')
        print('making candidates...')
        generalizer.make_candidates()
        print('initializing dict...')
        generalizer.initialize_dict()
        print('creating mapping')
        generalizer.create_mapping()
        print('pickling...')
        generalizer.pickle()

    user_in = 'placeholder'
    while user_in != 'exit':
        user_in = input('enter a word, or exit to quit:')
        print(generalizer.generalize(user_in))


