# Chester Palen-Michel
# Master's Thesis
# 3/16/18

from collections import namedtuple

import os
import codecs
from nltk import Tree, re
from CoreReader import CoreReader
from XML_ISNotesReader import ISFile

WordPair = namedtuple('WordPair', ['anaphor', 'antecedent']) #Should this store pairs of mentions or pairs of words?

class Bootstrapper:
    #TODO Do Candidate Extraction Patterns correctly.
    # TODO try more robust bert-chester hybrid idea with lexicon and pattern -->heuristic for pattern: word prep <X> lex(<Y>)
    #TODO oooo what if I use 'of' and 'of the' to find more pair words that can then allow finding more lexicalized extraction patterns.
    def __init__(self, corpus):
        self.corpus = corpus # list of Documents
        # Probably will need the following:
        # -- seedwords
        # -- permanent_lexicon (list of WordPairs)
        # -- temp_lexicon  (list of WordPairs)
        self.perm_lex = None
        self.temp_lex = None
        # -- extraction_pattern_list (just list of Pattern objects)
        # -- candidate_eps  (list of Pattern objects)
        self.ep_list = []
        self.candidate_ep = []

    def run(self, data_dir):
        # Use temp_lex pairs and dir of data
        # walk dir structure of ontonotes
        # for each file and each word pair, do search for eps add to candidate list
        # Make and use scorer to score eps and choose best one (not already in?) for ep list.
        # use EP list and get all word pairs into temp lexicon. take the top N-best new ones and add to permanent lex
        # loop through the above

        self.perm_lex = self.get_seedwords(np_only=True, head_only=True) # for svsv patterns
        self.temp_lex = self.perm_lex[:]
        # Create first round of patterns if no cache
        if not os.path.isfile('candidate_ep_cache.txt'):
            for wordpair in self.temp_lex:
                print('working on wordpair: ', wordpair)
                for dir, sub_dir, files in os.walk(data_dir):
                    for f in files:
                        file = codecs.open(os.path.join(dir,f), 'r', encoding='utf-8', errors='ignore')
                        corefreader = CoreReader()
                        docs = corefreader.read_file(file)
                        for doc in docs:
                            #self.candidate_ep.extend(self.find_svsv_patterns(wordpair, doc)) #TODO revist bert inspired patterns
                            self.candidate_ep.extend(self.find_npnp_patterns(wordpair, doc))
            self.cache_candidate_eps()
        # if there is a cache, just read patterns from file.
        else:
            print('Read patterns from cache')
            self.read_cache_candidate_eps()

        # Find matches
        for pattern in self.candidate_ep:
            print('working with :', pattern)
            for dir, sub_dir, files in os.walk(data_dir):
                for f in files:
                    file = codecs.open(os.path.join(dir, f), 'r', encoding='utf-8', errors='ignore')
                    corefreader = CoreReader()
                    docs = corefreader.read_file(file)
                    pattern.match(docs)

    def cache_candidate_eps(self):
        cachefile = codecs.open('candidate_ep_cache.txt', 'w', encoding='utf-8', errors='ignore')
        for pat in self.candidate_ep:
            cachefile.write(pat.pattern + '\n')

    def read_cache_candidate_eps(self):
        cachefile = codecs.open('candidate_ep_cache.txt', 'r', encoding='utf-8', errors='ignore')
        lines = cachefile.readlines()
        for line in lines:
            self.candidate_ep.append(NPNP(line.strip()))

    def find_npnp_patterns(self, wordpair, doc):
        # TODO Make an easier switch from head and full NP?
        # check each sentence and find both words in word pair.
        # if they are both in the sentence, then create a pattern.
        # Creating pattern: if adjacent, <X><Y> or <Y><X>, if not then <X>blahblah<Y> where X and Y are NPs.
        # X and Y are NPs but we want to extract the heads of X and Y.
        # X = anaphor, y = antecedent
        ret = []
        for s in doc.sentences:
            tokens = [tok.token for tok in s.words]
            if wordpair.anaphor.token in tokens and wordpair.antecedent.token in tokens:
                pattern_str1 = '(' + wordpair.anaphor.token + ')(.*)('+ wordpair.antecedent.token + ')'
                pattern_str2 = '(' + wordpair.antecedent.token + ')(.*)(' + wordpair.anaphor.token + ')'
                pattern1 = re.compile(pattern_str1)
                pattern2 = re.compile(pattern_str2)
                sent_str = ' '.join([w.token for w in s.words])
                match1 = re.search(pattern1, sent_str)
                match2 = re.search(pattern2, sent_str)
                if match1:
                    print(match1.group(1,2,3))
                    ret.append(NPNP('<X>(' + match1.group(2) + ')<Y>'))
                if match2:
                    print(match2.group(1,2,3))
                    ret.append(NPNP('<Y>(' + match2.group(2) + ')<X>'))
        return ret

    @DeprecationWarning
    def find_svsv_patterns(self, wordpair, doc):
        # look each 2 sentences if can find word pair as subjects in both sentences make new svsv_pattern
        # make new svsv pattern will require storing verbhead1 and verbhead2
        # return all svsv_patterns found
        # then if so check they are in subject position, if they are NP daughter of S in tree of sent,
        # then get the verb head and create svsvpattern.
        # THIS FUNCTION IS UNFINISHED AFTER IT WAS DETERMINED IT DOESN'T FIND ANY MATCHES IN DATA BESIDES SEED PAIRS
        ret = []
        for i in range(len(doc.sentences)-1):
            s1_sv_pairs= self.get_sv_pairs(doc.sentences[i])
            s2_sv_pairs = self.get_sv_pairs(doc.sentences[i+1])

            # for each sv pairs  in sentence check if mention matches add each match to list of matches
            anaphor_matches = [v for s,v in s2_sv_pairs if wordpair.anaphor.token.lower() == s.lower()]
            antecedent_matches = [v for s,v in s1_sv_pairs if wordpair.antecedent.token.lower() == s.lower()]
            # if they both are non-empty, create a pattern!
            if anaphor_matches and antecedent_matches:
                print('found a match')
            # for tree in strees check child NP and child VP
        return ret

    def get_sv_pairs(self, sentence):
        # Find subject and verb pairings in sentence and subordinate clauses
        s_trees = [subtree for subtree in sentence.tree.subtrees() if subtree.label() == 'S']
        sv_pairs = []
        for st in s_trees:
            np = None
            vhead = None
            for child in st:
                if child.label() == 'NP':
                    np = find_head(child)
                if child.label() == 'VP':
                    for c in child:
                        if c.label().startswith('VB'):
                            vhead = c.leaves()
                        else:
                            for further_child in c:
                                if isinstance(further_child, Tree):
                                    if further_child.label().startswith('VB'):
                                        vhead = further_child.leaves()
            if np and vhead:
                sv_pair = (np, ' '.join(vhead))
                sv_pairs.append(sv_pair)
        return sv_pairs

    def get_seedwords(self, np_only=False, head_only=False):
        # Get all bridging pairs in the corpus
        # Option to take only NPs and/or only head words
        ret = []
        for doc in bridge_dev_corpus:
            bridge_pairs = doc.get_bridgepairs()
            if np_only:
                #filter out the bridgepairs that aren't both NPs
                bridge_pairs = [wp for wp in bridge_pairs if wp.antecedent.isNP() and wp.anaphor.isNP()]
            if head_only:
                # make each bridgepair heads
                bridge_pairs = [WordPair(anap.head, ante.head) for anap, ante in bridge_pairs
                                if anap.head and ante.head]
            ret.extend(bridge_pairs)
        return ret



class Pattern:
    # Use this to make a easily readable pattern notation that gets turned into either regex or something behind scenes
    # Should probably include some counting metrics for how many things it finds, both in the already existing lexicon
    # and new finds that aren't already in the lexicon of pair words.
    def __init__(self, pattern):
        self.pattern = pattern #string form
        self.extractions_old = 0
        self.extractions_new = 0

    def __repr__(self):
        return self.pattern

    def match(self, data):
        # searches data for matches with the pattern. if found extracts the <X>, <Y> and adds them to a list of founds
        pass

class SVSV(Pattern):
  # Bert's idea
    def match(self, data):
        pass

class NPNP(Pattern):
    # My idea, and like literally everyone else.
    # take pattern, and create the regex to go with it.
    def __init__(self, pattern):
        Pattern.__init__(self, pattern)
        self.is_anaphor_first = pattern.startswith('<X>') # Track whether anaphor is first
        self.regex_pattern = re.compile(self.pattern.replace('<X>', '(\S+)').replace('<Y>', '(\S+)'))

    def match(self, data):
        # for each sentence in the data, do regex_pattern.search
        # with each match pair, check if they are already in perm lexicon.
        # adjust counts for pattern appropriately, add match pair to temp lexicon. add to patterns extraction set
        for doc in data:
            for sent in doc.sentences:
                match = self.regex_pattern.search(sent.string_form)
                if match:
                    if self.is_anaphor_first:
                        matchpair = WordPair(anaphor=match.group(1), antecedent=match.group(3))
                    else:
                        matchpair = WordPair(antecedent=match.group(1), anaphor=match.group(3))
                    # TODO if matchpair in perm_lexicon: extractions_old +=1
                        #TODO else: extractions_new +=1


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
    # ~~~~~NOTES~~~~~
    # then can run the documents through Bootstrapper
    # Will need to pass to non-bridging directory to find more items.
    # Bootstrap on all of OntoNotes corpus

    # Basic reading in of files. Could call this a bridge_corpus reader or something?
    coref_reader = CoreReader()
    bridge_dev_corpus = []
    input_dir = 'Coref_conll_IS_files_only/dev'
    for dir, sub_dir, files in os.walk(input_dir):
        for f in files:
            # do things to the files.
            input_f =  codecs.open(input_dir + '/' + f  , 'r', encoding='utf-8', errors='ignore') # coref input file
            is_input = ISFile('ISAnnotationWithoutTrace/dev/' + f.replace('.v4_auto_conll',
                                    '_entity_level.xml'))  # isfile needed to get markables, make from coref filename
            bridge_gold = open('bridging_gold/dev/' + f.replace('.v4_auto_conll', '.bridge_gold'),
                               'w')  # write to this, make from coref file name or is_input. .bridge_gold
            documents = coref_reader.read_file(input_f)
            for doc in documents:
                doc.markable2mention(is_input)
                doc.clusters_to_tokens()
                bridge_dev_corpus.append(doc)

    # Conll 2012 data
    conll_dir = '../InfoExtraction/project2-coref/conll-2012_auto_only/test'
    #conll_dir = 'Coref_conll_IS_files_only/dev'
    # You have a corpus now use it!
    bootstrapper = Bootstrapper(bridge_dev_corpus)
    bootstrapper.run(conll_dir)

    # Playing around with seedwords. Could use this to output lists of word pairs as a tsv or csv to include as
    # Appendix or something.
    #
    # seeds = bootstrapper.get_seedwords(np_only=True, head_only=True)
    # print(seeds)
    # print(len(seeds))
    # seeds2 = bootstrapper.get_seedwords(head_only=True)
    # print(seeds2)
    # print(len(seeds2))