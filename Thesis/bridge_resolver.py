# Chester Palen-Michel
# 3/28/18
# Master's Thesis
import codecs
import os
import random

from collections import defaultdict

from CoreReader import CoreReader, WordPair
from XML_ISNotesReader import ISFile
from sklearn import svm
from sklearn.feature_extraction import DictVectorizer
import numpy as np
from nltk.corpus.reader.nombank import NombankCorpusReader
from nltk.corpus import wordnet as wn

from tfidf import Tfidf
from generalizer import Generalizer

class BridgeResolver:
    # TODO feature ideas: antecedent bridge-chain size, name type antecedent-head_word
    # TODO nombank?, wordnet distance?, syntactic features? ( would require storing other selected bridges in doc? )

    """
    Resolves bridging relations in ISNotes.
    -------------------------------------------------------------
    General Process:

    Should consider making a thing to create the gold data again....
    Read in data, make bridging documents.
    Make Markables out of all NPs.
    Consider each NP's head whether is anaphor or not.
    If is anaphor, search previous N sentences.
    if antecedent found add mentions as pair to bridge_links (mention_pair objects) (namedtuple)
    Write named tuple pairs to a file in format appropriate for the scorer that I made.
    -------------------------------------------------------------
    """
    def __init__(self):
        self.wordpairs = defaultdict(list)
        self.gen_wordpairs = defaultdict(list)
        self.classifier = svm.SVC(probability=True)
        self.tfidf_dict = None
        self.pronouns = {'myself', 'ourselves', 'yourself', 'yourselves', 'himself', 'herself', 'itself', 'mine',
                         'ours', 'yours', 'his', 'hers', 'theirs', 'that', 'which', 'who', 'whom', 'whose', 'i', 'me',
                         'she', 'he', 'her', 'him', 'it', 'this', 'that', 'myself', 'yourself', 'himself', 'herself',
                         'itself', 'we', 'us', 'they', 'them', 'their' 'those', 'these', 'ourselves', 'yourselves',
                         'themselves', 'our', 'my', 'your'}
        self.comparatives = {'other', 'another', 'such', 'different', 'similar', 'additional', 'comparable',
                             'same', 'further', 'extra'}
        self.numbers = {'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten'}
        self.definites = {'the', 'that', 'this', 'those', 'these', 'one'}
        self.indef = {'a', 'an', 'some'}
        self.generalizer = Generalizer()
        self.generalizer.load_pickle('../Resources_Thesis/general_map_single.pkl')

    def simple_rule_run(self):
        # Iterate over file structure and resolve files.
        coref_reader = CoreReader()
        for dir, sub_dir, files in os.walk('Coref_conll_IS_files_only'):
            for f in files:
                input_f = codecs.open(os.path.join(dir, f), 'r', encoding='utf-8', errors='ignore')  # coref input file
                is_input = ISFile(os.path.join(dir, f).replace('.v4_auto_conll', '_entity_level.xml')
                                  .replace('Coref_conll_IS_files_only', 'ISAnnotationWithoutTrace/'))
                bridge_gold = open(os.path.join(dir, f).replace('.v4_auto_conll', '.bridge_gold')
                                   .replace('Coref_conll_IS_files_only', 'bridging_gold'), 'w')
                documents = coref_reader.read_file(input_f)
                bridging_pairs = []
                for doc in documents:
                    doc.markable2mention(is_input)
                    doc.clusters_to_tokens()

                    # Create the gold files for testing here.
                    bridge_gold.write(doc.write_bridges())

                    # Load bootstrapped resource
                    self.read_wordpairs('../Thesis2/onto_results_100/wordpairs.txt')

                    # Resolve bridges
                    print('docid: ', doc.docID)
                    bridging_pairs.extend(self.rulebased_resolve(doc))
                    print('bridges: ', self.rulebased_resolve(doc))

                # Output
                self.write_bridgepairs(bridging_pairs, os.path.join(dir, f).replace('.v4_auto_conll', '.bridge_sys')
                                           .replace('Coref_conll_IS_files_only', 'bridging_sysout'))

    def read_wordpairs(self, filepath):
        with codecs.open(filepath, 'r') as wordpairfile:
            lines = wordpairfile.readlines()
            for line in lines:
                pair = line.split('\t')
                # add anaphor as key and antecedent as values but only take last word (head)
                self.wordpairs[pair[0].strip().split('_')[-1]].append(pair[1].strip().split('_')[-1])

    def train_classifier(self, docs):
        bridgepairs = self.get_bridgepairs(docs)
        self.v = DictVectorizer(sparse=False)
        D = [self.feature_vectors(bridgepair) for bridgepair in bridgepairs]
        X = self.v.fit_transform(D)
        Y = self.labels(bridgepairs)
        self.classifier.fit(X, Y)

    def stat_resolve(self, doc):
        #TODO only consider "potential anaphors"
        bridges = []
        possible_anaphors = [mention for sent in doc.sentences for mention in sent.mentions
                             if self.is_possible_anaphor(mention, sent.mentions)]
        # for sent in doc.sentences:
        #     for mention in sent.mentions:
        #         if self.is_possible_anaphor(mention, sent.mentions):
        for mention in possible_anaphors:
            antecedents = self.stat_search_prev(doc, mention.sentence_index, mention)
            for antecedent in antecedents:
                bridges.append(WordPair(anaphor=mention, antecedent=antecedent))
        return bridges

    def rulebased_resolve(self, doc):
        # Returns list of bridges found by checking all previous mentions for words in resource
        bridges = []
        for sent in doc.sentences:
            for mention in sent.mentions:
                word_to_check = self.mention_in_wordpair(mention)
                if word_to_check:
                    antecedent_match = self.search_prev(doc, sent.index, mention.span_start,
                                                        self.wordpairs[word_to_check])
                    if antecedent_match:
                        bridges.append(WordPair(anaphor=mention, antecedent=antecedent_match))
        return bridges

    def mention_in_wordpair(self, mention):
        for word in mention.tokens:
            if word.token in self.wordpairs:
                return word.token

    def write_bridgepairs(self, bridgepairlist, outfilepath):
        with open(outfilepath, 'w') as outfile:
            for bp in bridgepairlist:
                if bp.anaphor.markable_id and bp.antecedent.markable_id:
                    outfile.write('\t'.join([bp.anaphor.underscore_span(), bp.anaphor.markable_id,
                                             bp.antecedent.underscore_span(), bp.antecedent.markable_id]) + '\n')
       #let's not use this one for printing gold. mentions are going to be paired in different structure and just
    def stat_search_prev(self, doc, sent_index, anaphor):
        antecedents = []
        anaphor_start = anaphor.span_start
        # Check same sentence.
        predicteds_and_scores = []
        for mention in doc.sentences[sent_index].mentions:
            if mention.span_end < anaphor_start:
                predicted = self.classifier.decision_function(np.array(self.v.transform(self.feature_vectors(Bridgepair(anaphor, mention)))))
                if predicted > 0:
                    predicteds_and_scores.append((mention, predicted[0]))
                #TODO keep all predicted scores and take the highest one or two, maybe set a cut off score?
                # if predicted[0] == 1:
                #     return [mention]
                    #antecedents.append(mention)
        # Check all other previous sentences
        i = sent_index - 1
        while i > 0 and i > sent_index -2:
            for mention in doc.sentences[i].mentions:
                predicted = self.classifier.decision_function(np.array(self.v.transform(self.feature_vectors(Bridgepair(anaphor, mention)))))
                if predicted > 0:
                    predicteds_and_scores.append((mention, predicted[0]))
                # if predicted[0] == 1:
                #     return [mention]
                    #antecedents.append(mention)
            i -= 1
        predicteds_and_scores.sort(key=lambda x: x[1], reverse=True)
        if predicteds_and_scores:
            return [predicteds_and_scores[0][0]]
        else:
            return []
        #return antecedents

    def search_prev(self, doc, sent_index, anaphor_start, potential_antecedents):
        # Check same sentence.
        for mention in doc.sentences[sent_index].mentions:
           # print('mention to check: ', mention)
            if mention.span_end < anaphor_start:
                for word in mention.tokens:
                    if word.token in potential_antecedents:
                        return mention
        # Check all other previous sentences
        i = sent_index -1
        while i > 0:
            for mention in doc.sentences[i].mentions:
                #print('mention to check further out: ', mention)
                for word in mention.tokens:
                    if word.token in potential_antecedents:
                        return mention
            i -= 1
        return None

    def get_bridgepairs(self, docs):
        training_samples = []
        for doc in docs:
            bridgepairs = doc.get_bridgepairs()
            for bridgepair in bridgepairs:
                training_samples.append(Bridgepair(bridgepair.anaphor, bridgepair.antecedent, True))
                negatives = self.neg_samples(bridgepair, doc)
                training_samples.extend(negatives)
        return training_samples

    def feature_vectors(self, bridgepair):
        feature_vec = {
        'pair_in_resource': self.pair_in_resource(bridgepair),
        'anaphor_pp': self.anaphor_pp(bridgepair),
        'definite': self.definite(bridgepair),
        'determiner': self.determiner(bridgepair),
        'len_anaphor': self.len_mention(bridgepair.anaphor), # may be bad?
        'len_antecedent': self.len_mention(bridgepair.antecedent),
        'anaphor_head_in_ante': self.anaphor_head_in_ante(bridgepair),
        'ante_tfidf_anap_resource' : self.ante_tfidf_anap_resource(bridgepair),
        'tfidf_ante': self.tfidf_ante(bridgepair),
        'ante_head_in_anaphor' : self.ante_head_mod_anap(bridgepair),
        'wn_dist' : self.wn_dist(bridgepair),
        'meronym' : self.meronym(bridgepair) or self.holonym(bridgepair),
        'same_head' : self.same_head(bridgepair), # lowers recall, not sure if gain in precision is enough...
        'generalized_resource' : self.gen_resource(bridgepair)
        }
        # 'ante-ne_anap-head': self.ante_ne_anap_head(bridgepair)  # this feature ruins everything :( seemingly
        # 'number-plural' : self.number_plural(bridgepair)  # slight decrese, may be worth it later?
        return feature_vec

    def labels(self, bridgepairs):
        return [1 if b.is_bridge else 0 for b in bridgepairs]

    def neg_samples(self, bridgepair, doc):
        random.seed(5)
        between_mentions = self.mentions_between(bridgepair, doc)
        if len(between_mentions) == 1:
            return [Bridgepair(bridgepair.anaphor, between_mentions[0], False)]
        elif len(between_mentions) == 2:
            return [Bridgepair(between_mentions[1], between_mentions[0], False)]
        elif len(between_mentions) == 3:
            return [Bridgepair(between_mentions[0], bridgepair.antecedent, False),
                    Bridgepair(between_mentions[2], between_mentions[1], False)]
        elif len(between_mentions) >= 4:
            random.shuffle(between_mentions)
            return [Bridgepair(between_mentions[0], between_mentions[1], False),
                    Bridgepair(between_mentions[2], between_mentions[3], False)]
        # elif len(between_mentions) >= 6:
        #     random.shuffle(between_mentions)
        #     return [Bridgepair(between_mentions[0], between_mentions[1], False),
        #             Bridgepair(between_mentions[2], between_mentions[3], False),
        #             Bridgepair(between_mentions[4], between_mentions[5], False)]
        else:
            return []

    def mentions_between(self, bridgepair, doc):
        """
        Finds all mentions between the anaphor and antecedent mentions of a bridgepair.
        """
        if bridgepair.anaphor.sentence_index == bridgepair.antecedent.sentence_index:
            between_mentions = [m for m in doc.sentences[bridgepair.anaphor.sentence_index].mentions
                                if
                                m.span_start > bridgepair.antecedent.span_end and m.span_end < bridgepair.anaphor.span_start]
        else:
            between_mentions = []
            between_mentions.extend([m for m in doc.sentences[bridgepair.antecedent.sentence_index].mentions
                                     if m.span_start > bridgepair.antecedent.span_end])
            between_mentions.extend([m for m in doc.sentences[bridgepair.anaphor.sentence_index].mentions
                                     if m.span_end > bridgepair.anaphor.span_start])
            for i in range(bridgepair.antecedent.sentence_index + 1, bridgepair.anaphor.sentence_index):
                between_mentions.extend(doc.sentences[i].mentions)
        return between_mentions

    def pair_in_resource(self, bridgepair):
        if bridgepair.anaphor.head.token in self.wordpairs:
            if bridgepair.antecedent.head.token in self.wordpairs[bridgepair.anaphor.head.token]:
                return 1
        return 0
#head based version


# Loose version
        # for w in bridgepair.anaphor.tokens:
        #     if w.token in self.wordpairs:
        #         for ante in  bridgepair.antecedent.tokens:
        #             if ante.token in self.wordpairs[w.token]:
        #                 return 1
        # return 0

    def anaphor_pp(self, bridgepair):
        for token in bridgepair.anaphor.tokens:
            if token.pos == 'IN':
                return 1
        return 0

    def definite(self, bridgepair):
        definites = {'the', 'that', 'this', 'those', 'these', 'one'}
        for w in bridgepair.anaphor.tokens:
            if w.token.lower() in definites:
                return 1
        return 0

    def determiner(self, bridgepair):
        if bridgepair.anaphor.token_str_list[0] in self.definites:
            return 'def'
        elif bridgepair.anaphor.token_str_list[0] in self.indef:
            return 'indef'
        else:
            return 'no_det'

    def len_mention(self, mention):
        if len(mention.tokens) <=2:
            return len(mention.tokens)
        elif len(mention.tokens) > 2 and len(mention.tokens) <=7:
            return 4
        elif len(mention.tokens) > 7 and len(mention.tokens) <= 14:
            return 5
        else:
            return 6

    def ante_ne_anap_head(self, bridgepair):
        return bridgepair.antecedent.head.name_ent + '-' + bridgepair.anaphor.head.token

    def anaphor_head_in_ante(self, bridgepair):
        return bridgepair.anaphor.head.token in  bridgepair.antecedent.token_str_list and \
               bridgepair.anaphor.head.token != bridgepair.antecedent.head.token

    def tfidf_ante(self, bridgepair):
        for w in bridgepair.antecedent.token_str_list:
            if w in self.tfidf_dict[bridgepair.antecedent.filename]:
                return 1
        return 0

    def ante_tfidf_anap_resource(self, bridgepair):
        return bridgepair.anaphor.head.token in self.wordpairs and self.tfidf_ante(bridgepair)

    def is_possible_anaphor(self, mention, mentions):
        return mention.head.token.lower() not in self.pronouns and self.no_comp_premod(mention) and \
               self.no_other_ments(mention, mentions) and not mention.head.pos.startswith('NNP')

    def no_comp_premod(self, mention):
        return not set(mention.token_str_list).intersection(self.comparatives)

    def no_other_ments(self, mention, mentions):
        for ment in mentions:
            mention_string = ' '.join(mention.token_str_list)
            ment_string = ' '.join(ment.token_str_list)
            if ment_string != mention_string:
                if mention_string.find(ment_string) != -1:
                    return False
        return True

    def ante_head_mod_anap(self, bridgepair):
        return bridgepair.antecedent.head.token.lower() in self.premod(bridgepair.anaphor)

    def premod(self, mention):
        return mention.token_str_list[:mention.token_str_list.index(mention.head.token)]

    def number_plural(self, bridgepair):
        for num in self.numbers:
            if num in [t.lower() for t in bridgepair.anaphor.token_str_list] and \
                (bridgepair.antecedent.head.pos == 'NNS' or bridgepair.antecedent.head.pos == 'NNPS'):
                return True
        return False

    def same_head(self, bridgepair):
        return bridgepair.anaphor.head.token == bridgepair.antecedent.head.token

    def meronym(self, bridgepair):
        anaphor_synsets = wn.synsets(bridgepair.antecedent.head.token)
        all_synsets = []
        for synset in anaphor_synsets:
            all_synsets.extend(synset.part_meronyms())
            all_synsets.extend(synset.substance_meronyms())
            all_synsets.extend(synset.member_meronyms())
        return bridgepair.anaphor.head.token in self.lemma_names(all_synsets)

    def holonym(self, bridgepair):
        anaphor_synsets = wn.synsets(bridgepair.anaphor.head.token)
        all_synsets = []
        for synset in anaphor_synsets:
            all_synsets.extend(synset.part_holonyms())
            all_synsets.extend(synset.substance_holonyms())
            all_synsets.extend(synset.member_holonyms())
        return bridgepair.antecedent.head.token in self.lemma_names(all_synsets)

    def lemma_names(self, synsets):
        # give me all lemma names for the list of synsets given.
        return [lemma.name() for synset in synsets for lemma in synset.lemmas()]

    def wn_dist(self, bridgepair):
        anaph_sets = wn.synsets(bridgepair.anaphor.head.token, pos=wn.NOUN)
        ante_sets = wn.synsets(bridgepair.antecedent.head.token, pos=wn.NOUN)
        highest_sim = 0.0 # this is essentially an impossibly high score for wn similarity
        for anaph_set in anaph_sets:
            for ante_set in ante_sets:
                sim = anaph_set.lch_similarity(ante_set)
                if sim > highest_sim:
                    highest_sim= sim
        if highest_sim < 0.5:
            return 'low'
        elif highest_sim >= 0.5 and highest_sim < 1:
            return 'lowish'
        elif highest_sim >= 1.0 and highest_sim < 1.5:
            return 'mid'
        elif highest_sim >= 1.5 and highest_sim < 2.0:
            return 'highish'
        elif highest_sim >= 2.0 and highest_sim < 2.5:
            return 'high'
        elif highest_sim >= 2.5 and highest_sim < 3:
            return 'higher'
        else:
            return 'really_high'

    def gen_resource(self, bridgepair):
        gen_anaphor = self.generalizer.generalize(bridgepair.anaphor.head.token)
        gen_antecedent = self.generalizer.generalize(bridgepair.antecedent.head.token)
        if gen_anaphor in self.gen_wordpairs:
            if gen_antecedent in self.gen_wordpairs[gen_anaphor]:
                return True
        return False

    def read_gen_wordpairs(self, filepath):
        with codecs.open(filepath, 'r') as wordpairfile:
            lines = wordpairfile.readlines()
            for line in lines:
                pair = line.split('\t')
                # add anaphor as key and antecedent as values but only take last word (head)
                self.gen_wordpairs[pair[0].strip()].append(pair[1].strip())


class Bridgepair:
    def __init__(self, anaphor, antecedent, is_bridge=False):
        self.anaphor = anaphor
        self.antecedent = antecedent
        self.is_bridge = is_bridge # default false since likely more non examples

if __name__ == "__main__":
    resolver = BridgeResolver()
    #resolver.simple_rule_run()

    # do training of classifier, function to extract features and mentions use heads!
    # classify the tests in each fold,
    # should be able to write each of the files in the tests in each fold to a single dir
    # can then compare with the gold directory directly. No worries. :)

    # Load bootstrapped resource
    resolver.read_wordpairs('../Resources_Thesis/concattenated/wordpairs.txt')
    resolver.read_gen_wordpairs('../Resources_Thesis/concattenated/gen_wordpairs.txt')
    # load tfidf
    tfidf = Tfidf()
    tfidf.load_docs('Coref_conll_IS_files_only/test')
    resolver.tfidf_dict = tfidf.top_n_dict(15)

    coref_reader = CoreReader()
    corpus = []
    for dir, sub_dir, files in os.walk('Coref_conll_IS_files_only/test'):
        for f in files:
            input_f = codecs.open(os.path.join(dir, f), 'r', encoding='utf-8', errors='ignore')  # coref input file
            is_input = ISFile(os.path.join(dir, f).replace('.v4_auto_conll', '_entity_level.xml')
                              .replace('Coref_conll_IS_files_only', 'ISAnnotationWithoutTrace/'))
            bridge_gold = open(os.path.join(dir, f).replace('.v4_auto_conll', '.bridge_gold')
                               .replace('Coref_conll_IS_files_only', 'bridging_gold'), 'w')
            documents = coref_reader.read_file(input_f)

            bridging_pairs = []
            for doc in documents:
                doc.markable2mention(is_input)
                doc.clusters_to_tokens()
                doc.filename = f
                corpus.append(doc)
                # Create the gold files for testing here.
                bridge_gold.write(doc.write_bridges())
                for sent in doc.sentences:
                    for mention in sent.mentions:
                        mention.filename = f
    # TODO someday should try to see if possible to run on all of corpus and not just test
    train_test_tup = []
    for i in range(10):
        test = corpus[i * 4: (i * 4)+4]
        train = corpus[0: (i * 4) ] + corpus[(i*4) + 4 : len(corpus)]
        train_test_tup.append((train, test))

    for tup in train_test_tup:
        resolver.train_classifier(tup[0])
        for test_file in tup[1]:
            print(test_file.filename)
            bridgepairs = resolver.stat_resolve(test_file) # write results in here?
            resolver.write_bridgepairs(bridgepairs, 'bridging_sysout/test/' +
                                       test_file.filename.replace('.v4_auto_conll', '.bridge_sys'))
