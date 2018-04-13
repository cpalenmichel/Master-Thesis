# Chester Palen-Michel
# Master's Thesis
# 3/16/18

from collections import namedtuple

import os
import codecs

from nltk import Tree, re, sys
from CoreReader import CoreReader
from XML_ISNotesReader import ISFile
import numpy as np
from scipy import spatial
from math import log


WordPair = namedtuple('WordPair', ['anaphor', 'antecedent']) #Should this store pairs of mentions or pairs of words?


class WordPairBootstrapper:
    """ To temporarily avoid having to refactor a thousand things and because this just needs to work, I'll change the
    two lines manually for switching between vector and normal matching.
    """
    # TODO use all patterns so new in_lexicon method works. Sanctions against south africa should be a match.
    # TODO what would happen if we just initialized all the exractions to 1. log 1 will 0 out anything unneeded?
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
        self.temp_lex = None
        # -- extraction_pattern_list (just list of Pattern objects)
        # -- candidate_eps  (list of Pattern objects)
        self.ep_list = []
        self.candidate_ep_strings = set()
        self.candidate_eps = []

    def run(self, input_dir):
        for x in range(self.ITERATIONS):
            print('Iteration #', x)
            # Find matches
            self.temp_lex = self.perm_lex[:]
            print('tempLex: ', self.temp_lex)
            for dir, sub_dir, files in os.walk(input_dir):
                for f in files:
                    print('working on ', f)
                    with codecs.open(os.path.join(dir, f), 'r', encoding='utf-8', errors='ignore') as input_f:
                        documents = coref_reader.read_file(input_f)
                        for doc in documents:
                            #doc.clusters_to_tokens() #why was this even here?
                            for pattern in self.candidate_eps:
                                pattern.match(doc, self.temp_lex)
            for pattern in self.candidate_eps:
                pattern.score_pattern()


            print(self.candidate_eps)
            candidate_ep_and_scores = [(ep, ep.score) for ep in self.candidate_eps]
            candidate_ep_and_scores.sort(key=lambda x: x[1], reverse=True)
            print(candidate_ep_and_scores)
            best_pat = self.best_pattern([ep for ep, score in candidate_ep_and_scores if score > 0])
            if best_pat is not None and best_pat not in self.ep_list:
                self.ep_list.append(best_pat)
            # add all extractions from ALL! patterns in ep_list to temp_lex.
            for pattern in self.ep_list:
                self.temp_lex.extend(pattern.extractions)
            # add the 5 best extractions from whole temp_lex
            # add 5 best extractions in best_pattern's extractions
            extractions_scores = [(extraction, self.score_extraction(extraction)) for extraction in self.temp_lex]
            extractions_scores.sort(key=lambda x: x[1], reverse=True)

            if len(extractions_scores) > 5:
                self.perm_lex.extend(self.highest_n(extractions_scores, 10)) # add the top 10
            # Note: for first few rounds this is probably kind of arbitrary which extractions are added. More informative
            # once there are more patterns to compare against.
            print(self.perm_lex)
            print('ep list: ', self.ep_list)

            self.reset_pattern_counts(self.candidate_eps)
        print('FINISHED!')
        print(self.perm_lex)
        print('ep list: ', self.ep_list)
        self.write_extracted_wordpairs('extraction_pairs.tsv')

    def run_candidate_patterns(self, doc):
        self.preprocess_seeds()
        self.perm_lex = self.seedwords
        # Create first round of patterns if no cache
        self.generate_candidate_patterns(doc)
        # THIS IS JUST TO CHECK WHAT KIND OF OUTPUT WE'RE GETTING.
        # cntr_eps = Counter(self.candidate_ep_strings)
        # # print(len(cntr_eps))
        # self.candidate_ep_strings = [ep for ep, cnt in cntr_eps.most_common(800)]

        self.cache_candidate_eps()
        # self.candidate_ep_strings = [ep for ep in self.candidate_ep_strings]
        self.string2pattern_set()

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

    def score_extraction(self, extraction):
        # TODO Need to thoroughly debug this method and see how scored and how assigning values. sanctions/africa is 0 and I don't think it should be...
        ret = 0
        for pat in self.ep_list:
            # print('pattern:', pat)
            # print('compare extraction: ', self.make_comparable(extraction))
            # print('extraction heads: ', pat.extraction_heads)
            ret += 1 +(.01 * pat.score) if self.make_comparable(extraction) in pat.extraction_heads else 0
        return ret

    def make_comparable(self, word_pair):
        anap = word_pair.anaphor.split('_')[-1]
        ante = word_pair.antecedent.split('_')[-1]
        return WordPair(anaphor=anap, antecedent=ante)

    def string2pattern_set(self):
        candidate_set = set(self.candidate_ep_strings)
        for ep in candidate_set:
            if ep == '<X> <X>':
                self.candidate_eps.append(NPNP())
            else:
                self.candidate_eps.append(NpatN(ep))

    def generate_candidate_patterns(self, document):
        for sent in document.sentences:
            self.candidate_ep_strings.update(self.candidate_patterns_from_sentence(sent))

    def candidate_patterns_from_sentence(self, sentence):
        ret = []
        i = 0
        index_end = len(sentence.words)-1
        temp_pat = '<Y> '
        # skip everything until noun
        while not sentence.words[i].pos.startswith('N') and i < index_end:
            i+=1
        # take everything until no longer noun
        while sentence.words[i].pos.startswith('N') and i < index_end:
            i+=1
        # take all non noun and add to pattern
        while i < index_end:
            while not sentence.words[i].pos.startswith('N') and i < index_end:
                temp_pat = temp_pat + sentence.words[i].token + ' '
                i+=1
            # find second noun mention and add <X>
            while sentence.words[i].pos.startswith('N') and i < index_end:
                temp_pat = temp_pat + '<X>'
                ret.append(temp_pat)
                temp_pat = '<Y> '
                i+=1
        reversed = []
        for pattern in ret:
            reversed.append('<X>' + pattern[3:-3] + '<Y>')
        ret.extend(reversed)
        return ret

    def cache_candidate_eps(self):
        cachefile = codecs.open('wordpair_bootstrapper_candidate_eps.txt', 'w', encoding='utf-8', errors='ignore')
        for pat in self.candidate_ep_strings:
            cachefile.write(pat + '\n')

    def read_cache_candidate_eps(self):
        cachefile = codecs.open('wordpair_bootstrapper_candidate_eps.txt', 'r', encoding='utf-8', errors='ignore')
        lines = cachefile.readlines()
        for line in lines:
            self.candidate_ep_strings.add((line.strip()))

    def get_seedwords(self, doc, np_only=False, head_only=False):
        # Get all bridging pairs in the corpus
        # Option to take only NPs and/or only head words
        # TODO should double check this still works
        ret = []
        bridge_pairs = doc.get_bridgepairs()
        if np_only and head_only:
            bridge_pairs = [ WordPair(wp.anaphor.head.token, wp.antecedent.head.token) for wp in bridge_pairs
            if wp.antecedent.isNP() and wp.anaphor.isNP() and wp.anaphor.head and wp.antecedent.head]
        elif np_only:
            #filter out the bridgepairs that aren't both NPs
            bridge_pairs = [self.wordpair_mentions2wordpair_str(wp) for wp in bridge_pairs if wp.antecedent.isNP() and wp.anaphor.isNP()]
        elif head_only:
            # make each bridgepair heads
            bridge_pairs = [WordPair(anap.head.token, ante.head.token) for anap, ante in bridge_pairs
            if anap.head and ante.head]
        else:
            bridge_pairs = [self.wordpair_mentions2wordpair_str(wp) for wp in bridge_pairs]
            # TODO may need to make this get only 'N' pos
        ret.extend(bridge_pairs)
        return ret

    def write_seedwords(self):
        seeds = self.get_seedwords(np_only=True, head_only=False)
        seed_f = codecs.open('seedwords.txt', 'w', encoding='utf-8', errors='ignore')
        for wp in seeds:
            seed_f.write(wp.anaphor + '\t' + wp.antecedent +'\n')

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
            pattern.extractions.clear()

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
        return result >= 0.7
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
        self.extractions = set()
        self.score = 0

    def __repr__(self):
        return self.pattern

    def match(self, document, lexicon):
        # searches data for matches with the pattern. if found extracts the <X>, <Y> and adds them to a list of founds
        pass

    def strict_in_lexicon(self, wordpair, lexicon):
        for lex in lexicon:
            comp_lex = self.make_comparable(lex)
            comp_wp = self.make_comparable(wordpair)
            is_match = comp_wp.anaphor.lower() == comp_lex.anaphor.lower() and comp_wp.antecedent.lower() == comp_lex.antecedent.lower()
            if is_match is True:
                return True
        return False

    def relaxed_in_lexicon(self, wordpair, lexicon):
        anaphors = set([wp.anaphor.lower().split('_')[-1]  for wp in lexicon])
        antecedents = set([wp.antecedent.lower().split('_')[-1]  for wp in lexicon])
        return wordpair.anaphor.split('_')[-1].lower() in anaphors and wordpair.antecedent.split('_')[-1].lower() in antecedents

    def vec_relaxed_in_lexicon(self, wordpair, lexicon):
        anaphors = set([wp.anaphor.lower().split('_')[-1] for wp in lexicon])
        antecedents = set([wp.antecedent.lower().split('_')[-1] for wp in lexicon])
        return self.vec_in(wordpair.anaphor.split('_')[-1].lower(), anaphors) and self.vec_in(wordpair.antecedent.split('_')[
                                                                           -1].lower(), antecedents)

    def vec_in(self, word, list):
        for w in list:
            if vector_sim(embeddings, w, word):
                return True
        return False

    def score_pattern(self):
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
    def __init__(self, pattern):
        Pattern.__init__(self, pattern)
        self.is_anaphor_first = pattern.startswith('<X>') # Track whether anaphor is first
        # self.regex_pattern = re.compile(self.pattern.replace('<X>', '(\S+)').replace('<Y>', '(\S+)'))
        self.tokens_of_pattern = self.pattern.replace('<X>', '').replace('<Y>', '').strip().split(' ')

    def match(self, document, lexicon):
        # for each sentence in the data, do regex_pattern.search
        # with each match pair, check if they are already in perm lexicon.
        # adjust counts for pattern appropriately, add match pair to temp lexicon. add to patterns extraction set

        for sent in document.sentences:
            wordpair_list = self.extract(self.tokens_of_pattern, sent)
            # match = self.regex_pattern.search(sent.string_form)
            # if match:
            #     if self.is_anaphor_first:
            #         extracted_wordpair = WordPair(anaphor=match.group(1), antecedent=match.group(2))
            #     else:
            #         extracted_wordpair = WordPair(antecedent=match.group(1), anaphor=match.group(2))
            self.extractions.update(wordpair_list)
            for wp in wordpair_list:
                self.extractions_in_lex += 1 if self.relaxed_in_lexicon(wp, lexicon) else 0
                self.total_extractions += 1
        self.extraction_heads = set([self.make_comparable(e) for e in self.extractions])

    def extract(self, pattern, sent):
        extractions = [] # list of wordpairs
        if pattern[0] in sent.string_tokens:
            starts = [i for i in range(len(sent.words)) if sent.string_tokens[i] == pattern[0]]
            matches = [] # list of (start, end) tuples
            for s in starts:
                if self.is_match(pattern, sent, s):
                    matches.append((s, s + len(pattern)-1))
            for match in matches:
                s = match[0] -1
                e = match[1] + 1
                extraction1 = []
                extraction2 = []
                while s >= 0 and sent.pos_tokens[s].startswith('N'):
                    extraction1.insert(0, sent.string_tokens[s])
                    s -= 1
                while e < len(sent.pos_tokens) and sent.pos_tokens[e].startswith('N'):
                    extraction2.append(sent.string_tokens[e])
                    e += 1
                if extraction1 and extraction2:
                    if self.is_anaphor_first:
                        extractions.append(WordPair(anaphor = '_'.join(extraction1), antecedent='_'.join(extraction2)))
                    else:
                        extractions.append(WordPair(antecedent='_'.join(extraction1), anaphor='_'.join(extraction2)))
        return extractions

    def is_match(self, pattern, sent, s):
        i = 1
        while s + i < len(sent.words) and i < len(pattern):
            if sent.string_tokens[s + i] != pattern[i]:
                return False
            i +=1
        return True

class NPNP(Pattern):
    # This is a special pattern just for capturing Noun-Noun compounds which appear to be indicative of
    # antecedent-anaphor pairs. ie 'terrorism expert'

    def __init__(self):
        Pattern.__init__(self, '<Y> <X>')
        self.is_anaphor_first = False # Track whether anaphor is first

    def match(self, document, lexicon):
        # for each sentence in the data, do search
        # with each match pair, check if they are already in perm lexicon.
        # adjust counts for pattern appropriately, add match pair to temp lexicon. add to patterns extraction set

        for sent in document.sentences:
            # check if sentence contains any of pattern
            for i in range(len(sent.words)-1):
                tok1 = sent.words[i]
                tok2 = sent.words[i+1]
                if tok1.pos.startswith('N') and tok2.pos.startswith('N') and tok1.token[0].islower() and tok2.token[0].islower():

                    extracted_wordpair = WordPair(anaphor=tok2.token, antecedent=tok1.token)
                    self.extractions.add(extracted_wordpair)
                    self.extractions_in_lex += 1 if self.relaxed_in_lexicon(extracted_wordpair, lexicon) else 0
                    self.total_extractions += 1
        self.extraction_heads = set([self.make_comparable(e) for e in self.extractions])

# TODO FASTER?
# def find_head(np):
#     noun_tags = ['NN', 'NNS', 'NNP', 'NNPS', 'PRP$', 'PRP']
#     if np == None:
#         ret = None
#     elif np.label() == 'NP' or np.label() == 'PRP$' or np.label() == 'PRP':
#         top_level_trees = [np[i] for i in range(len(np)) if type(np[i]) is Tree]
#         # search for a top-level noun
#         top_level_nouns = [t for t in top_level_trees if t.label() in noun_tags]
#         if len(top_level_nouns) > 0:
#             # if you find some, pick the rightmost one
#             ret = top_level_nouns[-1][0]
#         else:
#             # search for a top-level np
#             top_level_nps = [t for t in top_level_trees if t.label() == 'NP']
#             if len(top_level_nps) > 0:
#                 # if you find some, pick the head of the rightmost one
#                 ret = find_head(top_level_nps[-1])
#             else:
#                 # search for any noun
#                 nouns = [p[0] for p in np.pos() if p[1] in noun_tags]
#                 if len(nouns) > 0:
#                     # Choose right most
#                     ret = nouns[-1]
#                 else:
#                     # return the rightmost word
#                     ret = np.leaves()[-1]
#     else:
#         ret = None
#     return ret

if __name__ == '__main__':
    # TODO INPUT THE SEED WORDS FROM FILE AND RUN ON LARGER CORPUS. ALSO MORE ITERATIONS!
    # System args
    mode = sys.argv[1]
    input_dir = sys.argv[2]

    coref_reader = CoreReader()
    wp_bootstrapper = WordPairBootstrapper()

    if mode == 'seeds':
        # use input_dir
        # build is notes corpus
        for dir, sub_dir, files in os.walk(input_dir):
            for f in files:
                input_f = codecs.open(input_dir + '/' + f, 'r', encoding='utf-8', errors='ignore')  # coref input file
                is_input = ISFile('ISAnnotationWithoutTrace/dev/' + f.replace('.v4_auto_conll',
                                                                              '_entity_level.xml'))  # isfile needed to get markables, make from coref filename
                bridge_gold = open('bridging_gold/dev/' + f.replace('.v4_auto_conll', '.bridge_gold'),
                                   'w')  # write to this, make from coref file name or is_input. .bridge_gold
                documents = coref_reader.read_file(input_f)
                for doc in documents:
                    doc.markable2mention(is_input)
                    doc.clusters_to_tokens()
                    wp_bootstrapper.get_seedwords(doc)
        # create seedwords
        # write seeds to file
        wp_bootstrapper.write_seedwords()

    elif mode.startswith('wordpairs'):
        seedwords_filename = sys.argv[3]
        # make this a directory and output the wordpairs, the anaphors, the antecedents and patterns
        outdir = sys.argv[4]

        # read in seedwords as specified
        wp_bootstrapper.read_seedwords(seedwords_filename)

        # iterate through corpus to extract candidate patterns if needed
        if not os.path.isfile('wordpair_bootstrapper_candidate_eps.txt'):
            for dir, sub_dir, files in os.walk(input_dir):
                for f in files:
                    input_f = codecs.open(os.path.join(dir, f), 'r', encoding='utf-8', errors='ignore')  # coref input file
                    documents = coref_reader.read_file(input_f)
                    for doc in documents:
                        doc.clusters_to_tokens()
                        wp_bootstrapper.run_candidate_patterns(doc)
        else:
            wp_bootstrapper.preprocess_seeds()
            wp_bootstrapper.perm_lex = wp_bootstrapper.seedwords  #TODO Clean this up later it's clunky looking
            # print('Read patterns from cache')
            wp_bootstrapper.read_cache_candidate_eps()
            wp_bootstrapper.string2pattern_set()
        # iterate through corpus again for as many iterations as needed and do the extraction process
        wp_bootstrapper.run(input_dir)

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