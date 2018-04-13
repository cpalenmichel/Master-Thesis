# Chester Palen-Michel
# Thesis
# Sanity Check to see what coverage is in my resources and to check sanity of all future resources
import codecs
import sys

import os

from collections import defaultdict


class Sanity:
    def __init__(self):
        self.anaph_ante_goldtups = []
        self.wordpairs = defaultdict(list)
        self.wordpair_antecedents = set()
        self.anaphors =  []
        self.antecedents = []
        self.correct_anaphors = 0
        self.total_anaphors = 0
        self.correct_antecedents = 0
        self.total_antecedents = 0
        self.correct_wordpairs = 0
        self.total_wordpairs = 0

    def read(self, bridge_gold_dir):
        for dir, sub_dir, files in os.walk(bridge_gold_dir):
            for f in files:
                # print('file:', f)
                gold_f = open(os.path.join(dir, f), 'r')
                self._count_gold(gold_f)

    def read_wordpairs(self, filepath):
        with codecs.open(filepath, 'r') as wordpairfile:
            lines = wordpairfile.readlines()
            for line in lines:
                pair = line.split('\t')
                self.wordpairs[pair[0].strip().split('_')[-1]].append(pair[1].strip().split('_')[-1])

    def _count_gold(self, gold):
        self.anaph_ante_goldtups.extend([(line.split('\t')[0].strip().split('_')[-1], line.split('\t')[2].strip().split('_')[-1])
                                         for line in gold.readlines()])

    def make_lists(self):
        for ana , ante in self.anaph_ante_goldtups:
            self.anaphors.append(ana)
            self.antecedents.append(ante)

        for l in self.wordpairs.values():
            print(l)
            self.wordpair_antecedents.update(l)
    def check_anaphors(self):

        for anaphor in set(self.anaphors):
            if anaphor in set(self.wordpairs.keys()) :
                self.correct_anaphors +=1
            self.total_anaphors +=1
        return self.correct_anaphors / float(self.total_anaphors)

    def check_antecedents(self):
        for antecedent in set(self.antecedents):
            if antecedent in self.wordpair_antecedents:
                self.correct_antecedents += 1
            self.total_antecedents += 1
        return self.correct_antecedents / float(self.total_antecedents)

    def check_wordpairs(self):
        print(self.anaph_ante_goldtups)
        for word_pair in self.anaph_ante_goldtups:
            if word_pair[0] in self.wordpairs:
                if word_pair[1] in self.wordpairs[word_pair[0]]:
                    self.correct_wordpairs += 1
            self.total_wordpairs += 1
        return self.correct_wordpairs / float(self.total_wordpairs)


if __name__ == '__main__':
    keys_dir = sys.argv[1]
    sanity = Sanity()
    sanity.read(keys_dir)
    sanity.read_wordpairs('../Resources_Thesis/concattenated/wordpairs.txt')
    sanity.make_lists()
    print(sanity.wordpairs.values())
    print('anaphors', sanity.check_anaphors())
    print('antecedents', sanity.check_antecedents())
    print('wordpairs', sanity.check_wordpairs())