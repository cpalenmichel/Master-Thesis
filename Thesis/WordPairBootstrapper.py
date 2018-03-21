# Chester Palen-Michel
# Master's Thesis
# 3/16/18

from collections import namedtuple

import os
import codecs

from nltk import Tree, re, sys
from CoreReader import CoreReader
from XML_ISNotesReader import ISFile
from collections import Counter
from math import log
from Word import Word

WordPair = namedtuple('WordPair', ['anaphor', 'antecedent']) #Should this store pairs of mentions or pairs of words?

from collections import Mapping, Container
from sys import getsizeof


class WordPairBootstrapper:
    # TODO Do Candidate Extraction Patterns correctly. >.>
    # TODO try more robust bert-chester hybrid idea with lexicon and pattern --> heuristic for pattern: word prep <X> lex(<Y>)
    # TODO oooo what if I use 'of' and 'of the' to find more pair words that can then allow finding more lexicalized extraction patterns.


    # (2) TODO ensure saving full extraction and not just heads. separate list of 'heads' for scoring. simple rightmost head, for new ones?
    # (4) TODO Once have enough decent patterns (esp. if the patterns expected: 'of', 's, 'have', 'yx', other preps, etc.  get all potential in ontonotes
    # (5) TODO for anaphor-antecedent patterns and matching, let's use regex?
    def __init__(self, corpus):
        self.ITERATIONS = 100
        self.corpus = corpus # list of Documents
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
        self.candidate_ep_strings = []
        self.candidate_eps = []

    def run(self):
        # Use temp_lex pairs and dir of data
        # walk dir structure of ontonotes
        # for each file and each word pair, do search for eps add to candidate list
        # Make and use scorer to score eps and choose best one (not already in?) for ep list.
        # use EP list and get all word pairs into temp lexicon. take the top N-best new ones and add to permanent lex
        # loop through the above

        self.perm_lex = self.seedwords
        # Create first round of patterns if no cache
        if not os.path.isfile('wordpair_bootstrapper_candidate_eps.txt'):
            self.generate_candidate_patterns(corpus)

            # THIS IS JUST TO CHECK WHAT KIND OF OUTPUT WE'RE GETTING.
            cntr_eps = Counter(self.candidate_ep_strings)
            # print(len(cntr_eps))
            self.candidate_ep_strings = [ep for ep,cnt in cntr_eps.most_common(1000)]

            self.cache_candidate_eps()
           # self.candidate_ep_strings = [ep for ep in self.candidate_ep_strings]
            self.string2pattern_set()
        # if there is a cache, just read patterns from file.
        else:
            print('Read patterns from cache')
            self.read_cache_candidate_eps()
            self.string2pattern_set()

        # docs = []
        # for dir, sub_dir, files in os.walk(data_dir):
        #     print(dir)
        #     for f in files:
        #         print(f)
        #         file = codecs.open(os.path.join(dir, f), 'r', encoding='utf-8', errors='ignore')
        #         corefreader = CoreReader()
        #         doc = corefreader.read_file(file)
        #        docs.extend(doc)

        for x in range(self.ITERATIONS):
            print('Iteration #', x)
            # Find matches
            self.temp_lex = set(self.perm_lex[:])
            print('tempLex: ', self.temp_lex)
            for pattern in self.candidate_eps:
                #print('working with :', pattern)

                # for dir, sub_dir, files in os.walk(data_dir):
                #     for f in files:
                #         file = codecs.open(os.path.join(dir, f), 'r', encoding='utf-8', errors='ignore')
                #         corefreader = CoreReader()
                #         docs = corefreader.read_file(file)
                pattern.match(corpus, self.temp_lex) # was 2 tabs in

                pattern.score_pattern()

            print(self.candidate_eps)
            candidate_ep_and_scores = [(ep, ep.score) for ep in self.candidate_eps]
            candidate_ep_and_scores.sort(key=lambda x: x[1], reverse=True)
            print(candidate_ep_and_scores)
            best_pat = self.best_pattern([ep for ep, score in candidate_ep_and_scores])
            if best_pat is not None:
                self.ep_list.append(best_pat)
            # add all extractions from ALL! patterns in ep_list to temp_lex.
            for pattern in self.ep_list:
                self.temp_lex.update(pattern.extractions)
            # add the 5 best extractions from whole temp_lex
            # add 5 best extractions in best_pattern's extractions
            extractions_scores = [(extraction, self.score_extraction(extraction)) for extraction in self.temp_lex]
            extractions_scores.sort(key=lambda x: x[1], reverse=True)
            if len(extractions_scores) > 5:
                self.perm_lex.extend(self.highest_n(extractions_scores, 10)) # add the top 5
            # Note: for first few rounds this is probably kind of arbitrary which extractions are added. More informative
            # once there are more patterns to compare against.
            print(self.perm_lex)
            print(self.ep_list)
        print('FINISHED!')
        print(self.perm_lex)
        print(self.ep_list)

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
        ret = 0
        for pat in self.ep_list:
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

    def generate_candidate_patterns(self, corpus):
        # for dir, sub_dir, files in os.walk(data_dir):
        #     for f in files:
        #         file = codecs.open(os.path.join(dir, f), 'r', encoding='utf-8', errors='ignore')
        #         corefreader = CoreReader()
        #         docs = corefreader.read_file(file)
        for doc in corpus:
            for sent in doc.sentences:
                self.candidate_ep_strings.extend(self.candidate_patterns_from_sentence(sent))
        self.candidate_ep_strings = self.candidate_ep_strings

    def candidate_patterns_from_sentence(self, sentence):
        ret = []
        i = 0
        index_end = len(sentence.words)-1
        temp_pat = '<X> '
        while not sentence.words[i].pos.startswith('N') and i < index_end:
            i+=1
        while sentence.words[i].pos.startswith('N') and i < index_end:
            i+=1
        while i < index_end:
            while not sentence.words[i].pos.startswith('N') and i < index_end:
                temp_pat = temp_pat + sentence.words[i].token + ' '
                i+=1
            while sentence.words[i].pos.startswith('N') and i < index_end:
                temp_pat = temp_pat + '<X>'
                ret.append(temp_pat)
                temp_pat = '<X> '
                i+=1
        return ret

    def cache_candidate_eps(self):
        cachefile = codecs.open('wordpair_bootstrapper_candidate_eps.txt', 'w', encoding='utf-8', errors='ignore')
        for pat in self.candidate_ep_strings:
            cachefile.write(pat + '\n')

    def read_cache_candidate_eps(self):
        cachefile = codecs.open('wordpair_bootstrapper_candidate_eps.txt', 'r', encoding='utf-8', errors='ignore')
        lines = cachefile.readlines()
        for line in lines:
            self.candidate_ep_strings.append((line.strip()))

    def get_seedwords(self, np_only=False, head_only=False):
        # Get all bridging pairs in the corpus
        # Option to take only NPs and/or only head words
        ret = []
        for doc in corpus:
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

    def match(self, data, lexicon):
        # searches data for matches with the pattern. if found extracts the <X>, <Y> and adds them to a list of founds
        pass

    def in_lexicon(self, wordpair, lexicon):

        anaphors = set([wp.anaphor.token.lower() if isinstance(wp.anaphor, Word) else wp.anaphor.lower() for wp in lexicon])
        antecedents = set([wp.antecedent.token.lower() if isinstance(wp.antecedent, Word) else wp.antecedent.lower() for wp in lexicon])
        #print('anaphor: ', wordpair.anaphor.split('_')[-1])
        #print('antecedent: ', wordpair.antecedent.split('_')[-1])
        return wordpair.anaphor.split('_')[-1].lower() in anaphors and wordpair.antecedent.split('_')[-1].lower() in antecedents

    def score_pattern(self):
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
        self.is_anaphor_first = pattern.startswith('<X>') # Track whether anaphor is first TODO need new way to try to capture this...
        # self.regex_pattern = re.compile(self.pattern.replace('<X>', '(\S+)').replace('<Y>', '(\S+)'))

        self.tokens_of_pattern = self.pattern.replace('<X>', '').strip().split(' ')

    def match(self, data, lexicon):
        # for each sentence in the data, do regex_pattern.search
        # with each match pair, check if they are already in perm lexicon.
        # adjust counts for pattern appropriately, add match pair to temp lexicon. add to patterns extraction set
        for doc in data:
            for sent in doc.sentences:
                wordpair_list = self.extract(self.tokens_of_pattern, sent)
                # match = self.regex_pattern.search(sent.string_form)
                # if match:
                #     if self.is_anaphor_first:
                #         extracted_wordpair = WordPair(anaphor=match.group(1), antecedent=match.group(2))
                #     else:
                #         extracted_wordpair = WordPair(antecedent=match.group(1), anaphor=match.group(2))
                self.extractions.update(wordpair_list)
                for wp in wordpair_list:
                    self.extractions_in_lex += 1 if self.in_lexicon(wp, lexicon) else 0
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
                    extractions.append(WordPair(anaphor = '_'.join(extraction1), antecedent='_'.join(extraction2)))
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

    def match(self, data, lexicon):
        # for each sentence in the data, do search
        # with each match pair, check if they are already in perm lexicon.
        # adjust counts for pattern appropriately, add match pair to temp lexicon. add to patterns extraction set
        for doc in data:
            for sent in doc.sentences:
                # check if sentence contains any of pattern
                for i in range(len(sent.words)-1):
                    tok1 = sent.words[i]
                    tok2 = sent.words[i+1]
                    if tok1.pos.startswith('N') and tok2.pos.startswith('N') and tok1.token[0].islower() and tok2.token[0].islower():
                        extracted_wordpair = WordPair(anaphor=tok2.token, antecedent=tok1.token)
                        self.extractions.add(extracted_wordpair)
                        self.extractions_in_lex += 1 if self.in_lexicon(extracted_wordpair, lexicon) else 0
                        self.total_extractions += 1
        self.extraction_heads = set([self.make_comparable(e) for e in self.extractions])

def find_head(np):
    noun_tags = ['NN', 'NNS', 'NNP', 'NNPS', 'PRP$', 'PRP']
    if np == None:
        ret = None
    elif np.label() == 'NP' or np.label() == 'PRP$' or np.label() == 'PRP':
        top_level_trees = [np[i] for i in range(len(np)) if type(np[i]) is Tree]
        # search for a top-level noun
        top_level_nouns = [t for t in top_level_trees if t.label() in noun_tags]
        if len(top_level_nouns) > 0:
            # if you find some, pick the rightmost one
            ret = top_level_nouns[-1][0]
        else:
            # search for a top-level np
            top_level_nps = [t for t in top_level_trees if t.label() == 'NP']
            if len(top_level_nps) > 0:
                # if you find some, pick the head of the rightmost one
                ret = find_head(top_level_nps[-1])
            else:
                # search for any noun
                nouns = [p[0] for p in np.pos() if p[1] in noun_tags]
                if len(nouns) > 0:
                    # Choose right most
                    ret = nouns[-1]
                else:
                    # return the rightmost word
                    ret = np.leaves()[-1]
    else:
        ret = None
    return ret

if __name__ == '__main__':
    # TODO THIS CURRENTLY ONLY USES THE DEV CORPUS---should run this corpus to get seed words, write to file, then
    # TODO INPUT THE SEED WORDS FROM FILE AND RUN ON LARGER CORPUS. ALSO MORE ITERATIONS!
    # System args
    mode = sys.argv[1]
    input_dir = sys.argv[2]


    coref_reader = CoreReader()
    corpus = []

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
                    corpus.append(doc)
            # create seedwords

        # write seeds to file
        wp_bootstrapper = WordPairBootstrapper(corpus)
        wp_bootstrapper.write_seedwords()
    elif mode.startswith('wordpairs'):
        seedwords_filename = sys.argv[3]
        # read in corpus
        for dir, sub_dir, files in os.walk(input_dir):
            for f in files:
                input_f = codecs.open(os.path.join(dir, f), 'r', encoding='utf-8', errors='ignore')  # coref input file
                documents = coref_reader.read_file(input_f)
                for doc in documents:
                    doc.clusters_to_tokens()
                    corpus.append(doc)

        # read in seedwords as specified
        wp_bootstrapper = WordPairBootstrapper(corpus)
        wp_bootstrapper.read_seedwords(seedwords_filename)
        # do bootstrapping loop
        # TODO ? if mode.ends in c --> cosine metric for whether there is match in lexicon, mode.ends in n --> normal head-like match
        wp_bootstrapper.run()



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