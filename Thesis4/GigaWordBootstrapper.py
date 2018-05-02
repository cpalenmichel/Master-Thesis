# Chester Palen-Michel
# Final Project (Part of Master's Thesis)
# 3/16/18
import codecs
import json
from collections import namedtuple, defaultdict

import sys
import os

from math import log

import pickle
from nltk.corpus import TaggedCorpusReader

WordPair = namedtuple('WordPair', ['anaphor', 'antecedent']) #Should this store pairs of mentions or pairs of words?


class GigaBootstrapper:
    """
    Bootstrapper for ANC tagged corpus.
    To change the iterations you can do so in the constructor either explicitly or in the constructor call.
    Arguments to the program are seedword list and output directory.
    Running this script from command line:
    python3 ANCBootstrapper.py <seedwords> <outputdir>
    The outputdir is any directory you want to write the output files to. see anc_results_50_5wpi as an example
    Seedwords is just a list of wordpairs. See seedwords2.txt as an example.
    I have set the default iterations to 5 so it runs faster for a demo.
    """
    def __init__(self, iterations=50, extractions=10):
        self.ITERATIONS = iterations
        self.extractions_per_iterations = extractions
        self.seedwords = None
        self.perm_lex = None
        self.json_giga = None
        self.temp_lex = {}
        self.ep_list = []
        self.candidate_eps = []
        self.pattern_extractions = defaultdict(list)
        self.uselesswords = ['lot', 'sort', 'kind', 'um', 'huh', 'yeah', ']', '[', '/', 'i', 'things', 'right', 'yep',
                             'something', 'thing', 'anything', 'stuff', 'CD', 'hm', 'hmm', 'p','P', 'ras', '<', '>',
                             'C', 'c', 'tm', 'Trp', 'th', '']
        self.pattern_scores = {}

    def read_seedwords(self, filename):
        seed_f = codecs.open(filename, 'r', encoding='utf-8', errors='ignore')
        lines = seed_f.readlines()
        ret = []
        for line in lines:
            entries = line.split('\t')
            wp = WordPair(anaphor=entries[0].strip(), antecedent=entries[1].strip())
            ret.append(wp)
        self.seedwords = ret

    def preprocess_seeds(self):
        seedhead_pairs = []
        for wordpair in self.seedwords:
            anaphor = wordpair.anaphor.split('_')[-1]
            antecedent = wordpair.antecedent.split('_')[-1]
            seedhead_pairs.append(WordPair(anaphor=anaphor, antecedent=antecedent))
        self.seedwords = seedhead_pairs
        print('seeds processed')

    def generate_candidate_patterns(self, filename):
        self.json_giga =  json.load(open(filename, 'r'))
        print('patterns loaded from gigaword json object!')
        # for pat in json_giga:
        #     for w1 in json_giga[pat]:
        #         for w2 in json_giga[pat][w1]:
        #             self.pattern_extractions[pat].extend([WordPair(w1, w2)] * json_giga[pat][w1][w2])
        # print("converted into appropriate lists ")
        # # TODO what if this was cut out? Could we still bootstrap over just the dict? Close the file by now? release json_giga
        # for pat in self.pattern_extractions:
        #     self.candidate_eps.append(NpatN(pat, self.pattern_extractions[pat]))
        # print('patterns loaded')

    # def candidate_patterns_from_sentence(self, sentence):
    #     # sentence is now a list of tuples [1] is pos
    #     i = 0
    #     index_end = len(sentence) - 1
    #     temp_pat = '<Y> '
    #     extraction1 = []
    #     extraction2 = []
    #     # skip everything until noun
    #     while not sentence[i][1].startswith('N') and i < index_end:
    #         i += 1
    #     # take everything until no longer noun
    #     while sentence[i][1].startswith('N') and i < index_end:
    #         extraction1.append(sentence[i][0])
    #         i += 1
    #     # take all non noun and add to pattern
    #     while i < index_end:
    #         while not sentence[i][1].startswith('N') and i < index_end:
    #             if not (sentence[i][1] == 'DT' or sentence[i][1].startswith('J') or sentence[i][1] == 'CD'
    #                     or sentence[i][1].startswith('PRP')):
    #                 temp_pat = temp_pat + sentence[i][0] + ' '
    #             i += 1
    #         # find second noun mention and add <X>
    #         while sentence[i][1].startswith('N')  and i < index_end:
    #             extraction2.append(sentence[i][0])
    #             i += 1
    #         # create pattern and store, reset
    #         if len(temp_pat.split(' ')) < 6:
    #             temp_pat = temp_pat + '<X>'
    #             extraction1 = [e for e in extraction1 if e not in self.uselesswords]
    #             extraction2 = [e for e in extraction2 if e not in self.uselesswords]
    #             if extraction1 and extraction2 and not extraction1[-1] == extraction2[-1]:
    #                 self.pattern_extractions[temp_pat].append(
    #                     WordPair(anaphor=extraction1[-1], antecedent=extraction2[-1]))
    #                 self.pattern_extractions['<X>' + temp_pat[3:-3] + '<Y>'].append(WordPair(anaphor=extraction2[-1], antecedent=extraction1[-1]))
    #         temp_pat = '<Y> '
    #         extraction1 = extraction2[:]
    #         extraction2.clear()

    # def pickle_patterns(self):
    #     patterns_file = open('patterns_anc2.pkl', 'wb')
    #     pickle.dump(self.candidate_eps, patterns_file)
    #     print('dumped pickled file!')

    def write_results(self, outdir):
        pattern_f = codecs.open(os.path.join(outdir, 'patterns.txt'), 'w', encoding='utf-8', errors='ignore')
        pattern_f.writelines([pat + '\n' for pat in self.ep_list])
        pattern_f.close()

        wp_f = codecs.open(os.path.join(outdir, 'wordpairs.txt'), 'w', encoding='utf-8', errors='ignore')
        anaphor_f = codecs.open(os.path.join(outdir, 'anaphors.txt'), 'w', encoding='utf-8', errors='ignore')
        antecedents_f = codecs.open(os.path.join(outdir, 'antecedents.txt'), 'w', encoding='utf-8', errors='ignore')

        for wp in self.perm_lex:
            # write wordpairs to file
            wp_f.write(wp.anaphor + '\t' + wp.antecedent + '\n')
            # write anaphors
            anaphor_f.write(wp.anaphor + '\n')
            # write antecedents
            antecedents_f.write(wp.anaphor + '\n')

    def load_patterns(self):
        print('loading patterns and extractions from pickle! ')
        pattern_file = open('patterns_anc2.pkl', 'rb')
        self.candidate_eps = pickle.load(pattern_file)
        pattern_file.close()
        print('patterns loaded')

    def run(self):
        # Run Iterations
        self.temp_lex = self.perm_lex[:]
        for x in range(self.ITERATIONS):
            print('Iteration #', x)
            # Find matches
            print('tempLex: ', self.temp_lex[:50])
            # print(self.candidate_eps)
            # for strict match, make comparable early
            # comp_templex = [self.make_comparable(wp) for wp in self.perm_lex]

            best_score = 0
            best_pattern = None
            for pattern in self.json_giga:
                score = self.score_pattern(pattern, self.perm_lex)  # TODO replace with anaphor and antecedent when needed for relaxed
                if score > best_score and pattern not in self.ep_list:
                    best_score = score
                    best_pattern = pattern
            # print(self.candidate_eps)
            # candidate_ep_and_scores = [(ep, ep.score) for ep in self.candidate_eps if ep.score > 0 ]
            # candidate_ep_and_scores.sort(key=lambda x: x[1], reverse=True)
            # print(candidate_ep_and_scores[:20])
            # best_pat = self.best_pattern([ep for ep, score in candidate_ep_and_scores if score > 0])
            if best_pattern is not None and best_pattern not in self.ep_list:
                self.ep_list.append(best_pattern)
                self.temp_lex.extend(self.get_extractions(best_pattern))
                self.pattern_scores[best_pattern] = best_score
            # add all extractions from ALL! patterns in ep_list to temp_lex.
            print('ep list: ', self.ep_list[:25])
            # for pattern in self.ep_list:
            #     self.temp_lex.extend(self.get_extractions(pattern))
            # add the 5 best extractions from whole temp_lex
            # add 5 best extractions in best_pattern's extractions
            extractions_scores = [(extraction, self.score_extraction(extraction)) for extraction in set(self.temp_lex)]
            extractions_scores.sort(key=lambda x: x[1], reverse=True)
            # print('extraction_scores: ', extractions_scores)
            self.perm_lex.extend(self.highest_n(extractions_scores, self.extractions_per_iterations))  # add the top 10 or 5
            # Note: for first few rounds this is probably kind of arbitrary which extractions are added. More informative
            # once there are more patterns to compare against.
            print('permanent_lexicon: ', self.perm_lex)
            # print('ep list: ', self.ep_list)

            self.reset_pattern_counts(self.candidate_eps)
        print('FINISHED!')
        print(self.perm_lex)
        print('ep list: ', self.ep_list)


    def make_comparable(self, wp):
        anap = wp.anaphor.split('_')[-1]
        ante = wp.antecedent.split('_')[-1]
        return WordPair(anaphor=anap, antecedent=ante)

    def best_pattern(self, sorted_candidates):
        # print('sorted candidates', sorted_candidates)
        i = 0
        while i < len(sorted_candidates):
            if sorted_candidates[i] not in self.ep_list:
                return sorted_candidates[i]
            i += 1
        return None

    def score_extraction(self, extraction):
        ret = 0
        for pat in self.ep_list:
            ret += 1 + (.01 * self.pattern_scores[pat]) if extraction in self.get_extractions(pat) else 0 # TODO PROBS SlOW ?
        return ret

    def highest_n(self, extraction_scores, n):
        # take the n highest extraction scores not already in permanent lexicon
        ret = []
        i = 0
        while len(ret) < n and i < len(extraction_scores):
            if extraction_scores[i][0] not in self.perm_lex:
                ret.append(extraction_scores[i][0])
            i += 1
        return ret

    def reset_pattern_counts(self, candidate_eps):
        for pattern in candidate_eps:
            pattern.extractions_in_lex = 0
            pattern.total_extractions = 0

    def score_pattern(self, pattern, lexicon):
        extractions_in_lex = 0
        for w1 in self.json_giga[pattern]:
            for w2 in self.json_giga[pattern][w1]:
                extractions_in_lex += 1 * self.json_giga[pattern][w1][w2]\
                    if self.strict_in_lexicon(self.make_wordpair(pattern, w1, w2), lexicon) else 0
        total_extractions = self.num_of_extractions(self.json_giga[pattern])
        return self.calculate_score(extractions_in_lex, total_extractions)

    def calculate_score(self, extractions_in_lex, total_extractions):
        if extractions_in_lex == 0 or total_extractions == 0:
            self.score = 0
        else:
            self.score = (float(extractions_in_lex) / total_extractions) * log(extractions_in_lex)
        return self.score

    def make_wordpair(self, pat, w1 , w2):
        if pat.startswith('<X'):
            return WordPair(anaphor=w2, antecedent=w1)
        else:
            return WordPair(anaphor=w1, antecedent=w2)

    def num_of_extractions(self, extractions_dict):
        return sum([sum(extractions_dict[k].values()) for k in extractions_dict])

    def strict_in_lexicon(self, wordpair, lexicon):
        return wordpair in lexicon

    def get_extractions(self, pattern):  # TODO PROBS SLOW
        return [self.make_wordpair(pattern, w1, w2) for w1 in self.json_giga[pattern] for w2 in self.json_giga[pattern][w1]]

class Pattern:
    def __init__(self, pattern):
        self.pattern = pattern #string form
        self.extractions_in_lex = 0
        self.total_extractions = 0
        self.extractions = []
        self.score = 0

    def __repr__(self):
        return self.pattern

    def strict_in_lexicon(self, wordpair, lexicon):
        return wordpair in lexicon

    def relaxed_in_lexicon(self, wordpair, anaphors, antecedents):
        return wordpair.anaphor.split('_')[-1].lower() in anaphors and wordpair.antecedent.split('_')[-1].lower() in antecedents

    def score_pattern(self, lexicon):
        for wp in self.extractions:
            self.extractions_in_lex += 1 if self.strict_in_lexicon(wp, lexicon) else 0
        self.total_extractions = len(self.extractions)
        return self.calculate_score()

    def calculate_score(self):
        if self.extractions_in_lex == 0 or self.total_extractions == 0:
            self.score = 0
        else:
            self.score = (float(self.extractions_in_lex) / self.total_extractions) * log(self.extractions_in_lex)
        return self.score

    # def make_comparable(self, word_pair):
    #     anap = word_pair.anaphor.split('_')[-1]
    #     ante = word_pair.antecedent.split('_')[-1]
    #     return WordPair(anaphor=anap, antecedent=ante)

class NpatN(Pattern):
    # My idea, and like literally everyone else.
    # take pattern, and create the regex to go with it.
    def __init__(self, pattern, extractions):
        Pattern.__init__(self, pattern)
        self.is_anaphor_first = pattern.startswith('<Y>') # Track whether anaphor is first
        #self.regex_pattern = re.compile(self.pattern.replace('<X>', '(\S+)').replace('<Y>', '(\S+)'))
        self.tokens_of_pattern = self.pattern.replace('<X>', '').replace('<Y>', '').strip().split(' ')
        self.extractions =  extractions
        # if self.is_anaphor_first:
        #     self.extractions = [WordPair(anaphor=ext1, antecedent=ext2) for ext1, ext2 in extractions]
        # else:
        #     self.extractions = [WordPair(antecedent=ext1, anaphor=ext2) for ext1, ext2 in extractions]
        self.extraction_heads = set([self.make_comparable(e) for e in self.extractions])

if __name__ == '__main__':
    input_file = sys.argv[1]
    seedwords_filename = sys.argv[2]
    outdir = sys.argv[3]

    # For different iterations or number of words extracted per iteration:
    #  add the optional fields to the boostrapper constructor or change them explicitly in the constructor.
    bootstrapper = GigaBootstrapper()
    bootstrapper.read_seedwords(seedwords_filename)

    # first steps
    bootstrapper.preprocess_seeds()
    bootstrapper.perm_lex = bootstrapper.seedwords
    # iterate through corpus to extract candidate patterns if needed
    # if not os.path.isfile('patterns_giga.pkl'):
    bootstrapper.generate_candidate_patterns(input_file)
    #     bootstrapper.pickle_patterns()
    # else:
    #     bootstrapper.load_patterns()

    bootstrapper.run()

    # do bootstrapping loop
    if not os.path.isdir(outdir):
        os.makedirs(outdir)
    bootstrapper.write_results(outdir)