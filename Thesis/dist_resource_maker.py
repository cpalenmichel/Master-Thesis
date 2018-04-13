# Chester Palen-Michel
# Thesis
# 4/10/18
import pickle
from ANCBootstrapper import NpatN
from collections import defaultdict, Counter
from ANCBootstrapper import WordPair

class DistResourceMaker:
    def __init__(self):
        self.patterns_extractions = []
        self.final_dict = defaultdict(Counter)
        self.patterns = None

    def unpickle(self, filepath):
        print('loading patterns and extractions from pickle! ')
        pattern_file = open('patterns_anc.pkl', 'rb')
        self.patterns_extractions = pickle.load(pattern_file)
        pattern_file.close()
        print('patterns loaded')

    def load_patterns(self, filepath):
        pat_file = open(filepath, 'r')
        self.patterns = pat_file.readlines()
        self.patterns = [p.strip() for p in self.patterns]

    def make_useful_extractiondict(self):
        good_patterns = [pat for pat in self.patterns_extractions if pat.pattern in self.patterns]
        for gp in good_patterns:
            for ex in gp.extractions:
                self.final_dict[ex.anaphor][ex.antecedent] += 1

    def dump_extraction_words(self, filepath):
        file = open(filepath, 'w')
        for k, v in self.final_dict.items():
            for thing in v:
                file.write(k + '\n')
                file.write(thing + '\n')

    def whatever(self):
        # TODO cool, can we get a ratio that is informative? Dunning log likelihood from Hou?
        # TODO can we generalize it a bit more so that it's the semantic category from wordnet?
        #TODO Just do simple wordnet distance: Poesio did it
        print(self.final_dict['chairman'])
        print(self.final_dict['president'])
        print(self.final_dict['symptoms'])
        print(self.final_dict['car'])
        print(self.final_dict['building'])
        print(self.final_dict['school'])

if __name__ == '__main__':
    maker = DistResourceMaker()
    maker.unpickle('patterns_anc.pkl')
    maker.load_patterns('anc_results_100/patterns.txt')
    maker.make_useful_extractiondict()
    maker.whatever()
    #maker.dump_extraction_words('../Resources_Thesis/dumped_extraction_words.txt')