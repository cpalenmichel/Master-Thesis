# Chester Palen-Michel
# 3/28/18
# Master's Thesis
import codecs
import os

from collections import defaultdict

from CoreReader import CoreReader, WordPair
from XML_ISNotesReader import ISFile
from sklearn import svm


class BridgeResolver:
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
        self.classifier = svm.SVC()

    def read_wordpairs(self, filepath):
        with codecs.open(filepath, 'r') as wordpairfile:
            lines = wordpairfile.readlines()
            for line in lines:
                pair = line.split('\t')
                # add anaphor as key and antecedent as values but only take last word (head)
                self.wordpairs[pair[0].strip().split('_')[-1]].append(pair[1].strip().split('_')[-1])

    def train_classifier(self, docs):
        # for doc in docs, for bridgepair in doc
        # bridgepair get features, mentions between bridgepair get features
        X = [] # Features for each potential pair
        Y = [] # Labels
        self.classifier.fit(X, Y)
        pass

    def stat_resolve(self, doc):
        # TODO will also need a 10 fold cross validation setup with the ISNotes files. --read all, then do splits as needed.


        pass

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

if __name__ == "__main__":
    resolver = BridgeResolver()
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

            bridging_pairs =[]
            for doc in documents:
                doc.markable2mention(is_input)
                doc.clusters_to_tokens()

                # Create the gold files for testing here.
                bridge_gold.write(doc.write_bridges())

                # Load bootstrapped resource
                resolver.read_wordpairs('resources/vec_results_iterations_50/wordpairs.txt')

                # Resolve bridges
                print('docid: ', doc.docID)
                bridging_pairs.extend(resolver.rulebased_resolve(doc))
                print('bridges: ', resolver.rulebased_resolve(doc))

            # Output
            resolver.write_bridgepairs(bridging_pairs, os.path.join(dir,f).replace('.v4_auto_conll', '.bridge_sys')
                                       .replace('Coref_conll_IS_files_only', 'bridging_sysout'))