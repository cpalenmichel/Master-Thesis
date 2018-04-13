# Chester Palen-Michel
# Master's Thesis
# 3/16/18

from collections import namedtuple, defaultdict, Counter

import os
import codecs
import pickle

from nltk import sys
from nltk.corpus import TaggedCorpusReader
import numpy as np
from scipy import spatial
from math import log


WordPair = namedtuple('WordPair', ['anaphor', 'antecedent']) #Should this store pairs of mentions or pairs of words?


class WordPairBootstrapper:
    """ To temporarily avoid having to refactor a thousand things and because this just needs to work, I'll change the
    two lines manually for switching between vector and normal matching.
    """
    # TODO MAKE WORK WITH ANC formatting. Can't use so many stupid list comprehensions... Need to get extractions with patterns to make faster and easier. pickle then
    # (4) TODO Once have enough decent patterns (esp. if the patterns expected: 'of', 's, 'have', 'yx', other preps, etc.  get all potential in ontonotes
    # (5) TODO for anaphor-antecedent patterns and matching, let's use regex?

    def __init__(self):
        self.ITERATIONS = 50
        # Probably will need the following:
        # -- seedwords
        # -- permanent_lexicon (list of WordPairs)
        # -- temp_lexicon  (list of WordPairs)
        #self.seedwords_np_head = self.get_seedwords(np_only=True, head_only=True)
        self.seedwords = None # self.get_seedwords()
        self.perm_lex = None
        self.temp_lex = {}
        # -- extraction_pattern_list (just list of Pattern objects)
        # -- candidate_eps  (list of Pattern objects)
        self.ep_list = []
        self.candidate_eps = []
        self.pattern_extractions = defaultdict(list)

    def run(self, corpus):
        # TODO DO THIS ONLY IF NO PICKLE FILE.
        # Set-up candidate patterns and their extractions.
        if not os.path.isfile('patterns_anc.pkl'):
            self.extract_patterns(corpus)
        else:
            print('loading patterns and extractions from pickle! ')
            pattern_file = open('patterns_anc.pkl', 'rb')
            self.pattern_extractions = pickle.load(pattern_file)
            pattern_file.close()
            print('patterns loaded')

        # Run Iterations
        for x in range(self.ITERATIONS):
            print('Iteration #', x)
            # Find matches
            self.temp_lex = set(self.perm_lex[:])
            # anaphors = set([wp.anaphor.lower().split('_')[-1]  for wp in self.temp_lex])
            # antecedents = set([wp.anaphor.lower().split('_')[-1]  for wp in self.temp_lex])
            print('tempLex: ', self.temp_lex)
            # for strict match, make comparable early
            comp_templex = [self.make_comparable(wp) for wp in self.perm_lex] # TODO or templex? hmm

            # score patterns
            # for pattern in self.pattern_extractions:
            candidate_ep_and_scores = [(pattern, self.score_pattern(self.pattern_extractions[pattern], comp_templex))
                                    for pattern in self.pattern_extractions
                                       if self.score_pattern(self.pattern_extractions[pattern], comp_templex) != 0]

            # print(self.candidate_eps)
            # candidate_ep_and_scores = [(ep, ep.score) for ep in self.candidate_eps]
            candidate_ep_and_scores.sort(key=lambda x: x[1], reverse=True)
            print(candidate_ep_and_scores)
            best_pat = self.best_pattern([ep for ep, score in candidate_ep_and_scores])
            if best_pat is not None and best_pat not in self.ep_list:
                self.ep_list.append(best_pat)
            print('found best pattern')
            # add all extractions from ALL! patterns in ep_list to temp_lex.
            for pattern in self.ep_list:
                self.temp_lex.update(self.pattern_extractions[pattern])
            print('temp lex extended.', len(self.temp_lex))
            # add the 5 best extractions from whole temp_lex
            # add 5 best extractions in best_pattern's extractions
            extractions_scores = [(extraction, self.score_extraction(extraction, self.perm_lex)) for extraction in self.temp_lex]
            print('extraction scores created')
            extractions_scores.sort(key=lambda x: x[1], reverse=True)
            print('extraction scores sorted')
            #print('extraction_scores: ', extractions_scores)
            self.perm_lex.extend(self.highest_n(extractions_scores, 5)) # add the top 10 or 5
            print('perm lex extended')
            # Note: for first few rounds this is probably kind of arbitrary which extractions are added. More informative
            # once there are more patterns to compare against.
            print('permanent_lexicon: ', self.perm_lex)
            #print('ep list: ', self.ep_list)

            self.reset_pattern_counts(self.candidate_eps)
        print('FINISHED!')
        print(self.perm_lex)
        print('ep list: ', self.ep_list)
        self.write_extracted_wordpairs('extraction_pairs_anc.tsv')

    def extract_patterns(self, corpus):
        # extracts and pickles patterns
        # candidate_ep_size = len(self.pattern_extractions)
        # pattern_progress_counter = 0
        # for pattern in self.pattern_extractions:
        #     self.candidate_eps.append(NpatN(pattern, self.pattern_extractions[pattern]))
        #     # pattern.match(corpus)
        #     pattern_progress_counter += 1
        #     print(pattern_progress_counter)
        #     self.report_progress(pattern_progress_counter, candidate_ep_size)
        patterns_file = open('patterns_anc.pkl', 'wb')
        pickle.dump(self.pattern_extractions, patterns_file)
        print('dumped pickled file!')

    def run_candidate_patterns(self, corpus):
        self.preprocess_seeds()
        self.perm_lex = self.seedwords
        # Create first round of patterns if no cache
        self.generate_candidate_patterns(corpus)
        # self.cache_candidate_eps()
        # self.candidate_ep_strings = [ep for ep in self.candidate_ep_strings]
        # self.string2pattern_set()

    def write_extracted_wordpairs(self, filepath):
        with open(filepath, 'w') as outfile:
            for wordpair in self.perm_lex:
                outfile.write(wordpair.anaphor + '\t' + wordpair.antecedent + '\n')

    def preprocess_seeds(self):
        seedhead_pairs = []
        for wordpair in self.seedwords:
            anaphor = wordpair.anaphor.split('_')[-1]
            antecedent = wordpair.antecedent.split('_')[-1]
            seedhead_pairs.append(WordPair(anaphor= anaphor, antecedent=antecedent))
        self.seedwords = seedhead_pairs

    def best_pattern(self, sorted_candidates):
        #print('sorted candidates', sorted_candidates)
        i = 0
        while i < len(sorted_candidates):
            if sorted_candidates[i] not in self.ep_list:
                return sorted_candidates[i]
            i += 1
        return None

    def highest_n(self, extraction_scores, n):
        # take the n highest extraction scores not already in permanent lexicon
        ret = []
        i = 0
        while len(ret) < n and i < len(extraction_scores):
            if extraction_scores[i][0] not in self.perm_lex:
                ret.append(extraction_scores[i][0])
            i += 1
        return ret

    def score_extraction(self, extraction, lexicon):
        # TODO Need to thoroughly debug this method and see how scored and how assigning values. sanctions/africa is 0 and I don't think it should be...
        ret = 0
        for pat in self.ep_list:
            # print('pattern:', pat)
            # print('compare extraction: ', self.make_comparable(extraction))
            # print('extraction heads: ', pat.extraction_heads)
            ret += 1 +(.01 * self.score_pattern(self.pattern_extractions[pat], lexicon)) if self.make_comparable(extraction) in self.pattern_extractions[pat] else 0
        return ret

    def make_comparable(self, word_pair):
        anap = word_pair.anaphor.split('_')[-1]
        ante = word_pair.antecedent.split('_')[-1]
        return WordPair(anaphor=anap, antecedent=ante)

    def string2pattern_set(self):
        candidate_set = set(self.candidate_ep_strings)
        for ep in candidate_set:
            ep = ep.replace('*', '', 3) #Thesis hack to avoid bad input from is notes.
            if ep == '<X> <X>':
                self.candidate_eps.append(NPNP())
            else:
                self.candidate_eps.append(NpatN(ep))

    def generate_candidate_patterns(self, corpus):
        for sent in corpus.tagged_sents():
            self.candidate_patterns_from_sentence(sent)
        print('patterns loaded!')
        self.pattern_extractions = {key : extract_list for key, extract_list in
                                    self.pattern_extractions.items() if len(extract_list) > 6}
        print('patterns reduced!')

    def candidate_patterns_from_sentence(self, sentence):
        # sentence is now a list of tuples [1] is pos
        i = 0
        index_end = len(sentence)-1
        temp_pat = '<Y> '
        extraction1 = []
        extraction2 = []
        # skip everything until noun
        while not sentence[i][1].startswith('N') and i < index_end:
            i+=1
        # take everything until no longer noun
        while sentence[i][1].startswith('N') and i < index_end:
            extraction1.append(sentence[i][0])
            i+=1
        # take all non noun and add to pattern
        while i < index_end:
            while not sentence[i][1].startswith('N') and i < index_end:
                temp_pat = temp_pat + sentence[i][0] + ' '
                i+=1
            # find second noun mention and add <X>
            while sentence[i][1].startswith('N') and i < index_end:
                extraction2.append(sentence[i][0])
                i += 1
            # create pattern and store, reset
            if len(temp_pat.split(' ')) < 6:
                temp_pat = temp_pat + '<X>'
                self.pattern_extractions[temp_pat].append(WordPair(antecedent='_'.join(extraction1),anaphor=('_'.join(extraction2))))
                self.pattern_extractions['<X>' + temp_pat[3:-3] + '<Y>'].append(WordPair(anaphor='_'.join(extraction1)
                                                                                ,antecedent=('_'.join(extraction2))))
            temp_pat = '<Y> '
            extraction1 = extraction2
            extraction2.clear()

    def cache_candidate_eps(self):
        pass
        # cachefile = codecs.open('wordpair_bootstrapper_candidate_eps_anc2.txt', 'w', encoding='utf-8', errors='ignore')
        # for pat, count in self.candidate_ep_strings.most_common(47397): # this number chosen, each pattern with 3 or more hits
        #     #cachefile.write(pat + '\t' + str(count) + '\n')
        #     cachefile.write(pat +  '\n')
    def read_cache_candidate_eps(self):
        cachefile = codecs.open('wordpair_bootstrapper_candidate_eps_anc.txt', 'r', encoding='utf-8', errors='ignore')
        lines = cachefile.readlines()
        for line in lines:
            self.candidate_ep_strings.add((line.strip()))

    def read_seedwords(self, filename):
        seed_f = codecs.open(filename, 'r', encoding='utf-8', errors='ignore')
        lines = seed_f.readlines()
        ret = []
        for line in lines:
            entries = line.split('\t')
            wp = WordPair(anaphor=entries[0].strip(), antecedent=entries[1].strip())
            ret.append(wp)
        self.seedwords = ret

    def wordpair_mentions2wordpair_str(self, wp):
        """ Convert WordPair of mentions into WordPair
        of strings, so don't have to worry about type checking. """
        anap = '_'.join(wp.anaphor.token_str_list)
        ante = '_'.join(wp.antecedent.token_str_list)
        return WordPair(anaphor=anap, antecedent=ante)

    def write_results(self, outdir):
        pattern_f = codecs.open(os.path.join(outdir,'patterns.txt'), 'w',encoding='utf-8', errors='ignore')
        pattern_f.writelines([pat.pattern + '\n' for pat in self.ep_list])
        pattern_f.close()

        wp_f = codecs.open(os.path.join(outdir, 'wordpairs.txt'), 'w', encoding='utf-8', errors='ignore')
        anaphor_f = codecs.open(os.path.join(outdir,'anaphors.txt'), 'w',encoding='utf-8', errors='ignore')
        antecedents_f = codecs.open(os.path.join(outdir,'antecedents.txt'), 'w',encoding='utf-8', errors='ignore')

        for wp in self.perm_lex:
            # write wordpairs to file
            wp_f.write(wp.anaphor + '\t' + wp.antecedent + '\n')
            # write anaphors
            anaphor_f.write(wp.anaphor + '\n')
            #write antecedents
            antecedents_f.write(wp.anaphor + '\n')

    def reset_pattern_counts(self, candidate_eps):
        for pattern in candidate_eps:
            pattern.extractions_in_lex = 0
            pattern.total_extractions = 0

    def report_progress(self, pattern_progress_counter, candidate_ep_size):
        percent25 = int(candidate_ep_size * 0.25)
        percent50 = int(candidate_ep_size * 0.5)
        percent75 = int(candidate_ep_size * 0.75)
        if pattern_progress_counter == percent25:
            print('25% complete')
        elif pattern_progress_counter == percent50:
            print('50% complete')
        elif pattern_progress_counter == percent75:
            print('75% complete')

    def strict_in_lexicon(self, wordpair, lexicon):
        comp_wp = self.make_comparable(wordpair)
        return comp_wp in lexicon

    def score_pattern(self, extractions, lexicon):
        extractions_in_lex = 0
        for wp in extractions:
            extractions_in_lex += 1 if self.strict_in_lexicon(wp, lexicon) else 0
        total_extractions = len(extractions)
        return self.calculate_score(extractions_in_lex, total_extractions)

    def calculate_score(self, extractions_in_lex, total_extractions):
        if extractions_in_lex == 0 or total_extractions == 0:
           score = 0
        else:
           score = (float(extractions_in_lex) / total_extractions) * log(extractions_in_lex)
        return score

#Vector similarity stuff. Should be part of a class somewhere, but not clear where to put it.
def load_embeddings(filepath):
    print('Loading embeddings....')
    embeddings_index = {}
    with open(filepath, encoding='utf8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
    print("Finished loading embeddings!")
    return embeddings_index

def vector_sim(embeddings_index, word1, word2):
    if word1 in embeddings_index and word2 in embeddings_index:
        result = 1 - spatial.distance.cosine(embeddings_index[word1], embeddings_index[word2])
        return result >= 0.9
    else:
        return False

# Testing word vector similarity stuff
#embeddings = load_embeddings("/home/chester/Documents/Thesis/WordVectors/glove.6B/glove.6B.100d.txt")

class Pattern:
    # Use this to make a easily readable pattern notation that gets turned into either regex or something behind scenes
    # and new finds that aren't already in the lexicon of pair words.
    def __init__(self, pattern):
        self.pattern = pattern #string form
        self.extractions_in_lex = 0
        self.total_extractions = 0
        self.extractions = []
        self.score = 0

    def __repr__(self):
        return self.pattern

    def match(self, corpus):
        # searches data for matches with the pattern. if found extracts the <X>, <Y> and adds them to a list of founds
        pass

    def strict_in_lexicon(self, wordpair, lexicon):
        comp_wp = self.make_comparable(wordpair)
        return comp_wp in lexicon


    def relaxed_in_lexicon(self, wordpair, anaphors, antecedents):
        return wordpair.anaphor.split('_')[-1].lower() in anaphors and wordpair.antecedent.split('_')[-1].lower() in antecedents

    def vec_relaxed_in_lexicon(self, wordpair, anaphors, antecedents, embeddings):
        return self.vec_in(wordpair.anaphor.split('_')[-1].lower(), anaphors, embeddings) and self.vec_in(wordpair.antecedent.split('_')[
                                                                           -1].lower(), antecedents, embeddings)

    def vec_strict_in_lexicon(self, wordpair, lexicon):
        comp_wp = self.make_comparable(wordpair)
        for lex in lexicon:
            if vector_sim(embeddings, comp_wp.antecedent, lex.antecedent) and vector_sim(embeddings, comp_wp.anaphor, lex.anaphor):
                return True
        return False

    def vec_in(self, word, list, embeddings):
        for w in list:
            if vector_sim(embeddings, w, word):
                return True
        return False

    def score_pattern(self, lexicon):
        for wp in self.extractions:
            self.extractions_in_lex += 1 if self.strict_in_lexicon(wp, lexicon) else 0
        self.total_extractions = len(self.extractions)
        return self.calculate_score()

    def calculate_score(self):
        # TODO Log of 1 = 0 is preventing sanctions against africa from being scored in strict_in_lexicon
        if self.extractions_in_lex == 0 or self.total_extractions == 0:
            self.score = 0
        else:
            self.score = (float(self.extractions_in_lex) / self.total_extractions) * log(self.extractions_in_lex)
        return self.score

    def make_comparable(self, word_pair):
        anap = word_pair.anaphor.split('_')[-1]
        ante = word_pair.antecedent.split('_')[-1]
        return WordPair(anaphor=anap, antecedent=ante)

class NpatN(Pattern):
    # My idea, and like literally everyone else.
    # take pattern, and create the regex to go with it.
    def __init__(self, pattern, extractions):
        Pattern.__init__(self, pattern)
        self.is_anaphor_first = pattern.startswith('<X>') # Track whether anaphor is first
        #self.regex_pattern = re.compile(self.pattern.replace('<X>', '(\S+)').replace('<Y>', '(\S+)'))
        self.tokens_of_pattern = self.pattern.replace('<X>', '').replace('<Y>', '').strip().split(' ')
        if self.is_anaphor_first:
            self.extractions = [WordPair(anaphor=ext1, antecedent=ext2) for ext1, ext2 in extractions]
        else:
            self.extractions = [WordPair(antecedent=ext1, anaphor=ext2) for ext1, ext2 in extractions]

    # def match(self, corpus):
    #     for sent in corpus.tagged_sents():
    #         wordpair_list = self.extract(self.tokens_of_pattern, sent)
    #         self.extractions.extend(wordpair_list)
    #     self.extraction_heads = set([self.make_comparable(e) for e in self.extractions])
    #
    # def extract(self, pattern, sent):
    #     extractions = [] # list of wordpairs
    #     if pattern[0] in [w[0] for w in sent]:
    #         starts = [i for i in range(len([w[0] for w in sent])) if [w[0] for w in sent][i] == pattern[0]]
    #         # list of start end tuples
    #         matches = [(s, s + len(pattern)-1) for s in starts if self.is_match(pattern, sent, s)]
    #         for match in matches:
    #             s = match[0] -1
    #             e = match[1] + 1
    #             extraction1 = []
    #             extraction2 = []
    #             while s >= 0 and [w[1] for w in sent][s].startswith('N'):
    #                 extraction1.insert(0, [w[0] for w in sent][s])
    #                 s -= 1
    #             while e < len(sent) and [w[1] for w in sent][e].startswith('N'):
    #                 extraction2.append([w[0] for w in sent][e])
    #                 e += 1
    #             if extraction1 and extraction2:
    #                 if self.is_anaphor_first:
    #                     wp_extraction = WordPair(anaphor='_'.join(extraction1), antecedent='_'.join(extraction2))
    #                     extractions.append(wp_extraction)
    #                     # self.extractions_in_lex += 1 if self.relaxed_in_lexicon(wp_extraction, anaphors, antecedents) else 0
    #                     # self.total_extractions +=1
    #                 else:
    #                     wp_extraction = WordPair(antecedent='_'.join(extraction1), anaphor='_'.join(extraction2))
    #                     extractions.append(wp_extraction)
    #                     # self.extractions_in_lex +=1 if self.relaxed_in_lexicon(wp_extraction, anaphors, antecedents) else 0
    #                     # self.total_extractions +=1
    #     return extractions
    #
    # def is_match(self, pattern, sent, s):
    #     i = 1
    #     while s + i < len(sent) and i < len(pattern):
    #         if [w[0] for w in sent][s + i] != pattern[i]:
    #             return False
    #         i +=1
    #     return True

# class NPNP(Pattern):
#     # This is a special pattern just for capturing Noun-Noun compounds which appear to be indicative of
#     # antecedent-anaphor pairs. ie 'terrorism expert'
#
#     def __init__(self):
#         Pattern.__init__(self, '<Y> <X>')
#         self.is_anaphor_first = False # Track whether anaphor is first
#
#     def match(self, corpus):
#         for sent in corpus.tagged_sents:
#             for i in range(len(sent)-1):
#                 tok1 = sent[i]
#                 tok2 = sent[i+1]
#                 if tok1[1].startswith('N') and tok2[1].startswith('N') and tok1[0][0].islower() and tok2[0][0].islower():
#                     extracted_wordpair = WordPair(anaphor=tok2[0], antecedent=tok1[0])
#                     self.extractions.append(extracted_wordpair)
#                 # self.extractions_in_lex += 1 if self.relaxed_in_lexicon(extracted_wordpair, anaphors, antecedents) else 0
#                 # self.total_extractions += 1
#         self.extraction_heads = set([self.make_comparable(e) for e in self.extractions])

if __name__ == '__main__':
    input_dir = sys.argv[1]
    #coref_reader = CoreReader()

    wp_bootstrapper = WordPairBootstrapper()
    seedwords_filename = sys.argv[2]

    # make this a directory and output the wordpairs, the anaphors, the antecedents and patterns
    outdir = sys.argv[3]

    # read in seedwords as specified
    wp_bootstrapper.read_seedwords(seedwords_filename)

    # read in entire corpus
    corpus = TaggedCorpusReader('../../prepANC', '.*', '_', encoding='utf-8')
    print('corpus is loaded!')

    # iterate through corpus to extract candidate patterns if needed
    if not os.path.isfile('patterns_anc.pkl'):
        wp_bootstrapper.run_candidate_patterns(corpus)
    else:
        wp_bootstrapper.preprocess_seeds()
        wp_bootstrapper.perm_lex = wp_bootstrapper.seedwords  #TODO Clean this up later it's clunky looking
        # print('Read patterns from cache')
        # wp_bootstrapper.read_cache_candidate_eps()
        # wp_bootstrapper.string2pattern_set()
    # iterate through corpus again for as many iterations as needed and do the extraction process
    # wp_bootstrapper.run_candidate_patterns(corpus)
    wp_bootstrapper.run(corpus)

    # do bootstrapping loop
    if not os.path.isdir(outdir):
        os.makedirs(outdir)
    wp_bootstrapper.write_results(outdir)



        # Reading in files.
        # Seedwords file input:
        # input_dir = 'Coref_conll_IS_files_only/dev'
        #
        #
        # # Full IS Notes files
        # conll_dir = 'Coref_conll_IS_files_only/'
        #
        #
        # wp_bootstrapper = WordPairBootstrapper(corpus)
        # wp_bootstrapper.run()

        # Playing around with seedwords. Could use this to output lists of word pairs as a tsv or csv to include as
        # Appendix or something.
        #
        # seeds = bootstrapper.get_seedwords(np_only=True, head_only=True)
        # print(seeds)
        # print(len(seeds))
        # seeds2 = bootstrapper.get_seedwords(head_only=True)
        # print(seeds2)
        # print(len(seeds2))