# Chester Palen-Michel
# 4/28/18
import codecs
import os
from collections import namedtuple

from CoreReader import CoreReader
from XML_ISNotesReader import ISFile
from anaphora_recognizer import Recognizer

WordPair = namedtuple('WordPair', ['anaphor', 'antecedent'])


class Experimenter:
    def __init__(self):
        self.corpus = []
        self.train_test_tup = []

    def read_corpus(self):
        coref_reader = CoreReader()
        for dir, sub_dir, files in os.walk('Coref_conll_IS_files_only'):
            for f in files:
                input_f = codecs.open(os.path.join(dir, f), 'r', encoding='utf-8',
                                      errors='ignore')  # coref input file
                is_input = ISFile(os.path.join(dir, f).replace('.v4_auto_conll', '_entity_level.xml')
                                  .replace('Coref_conll_IS_files_only', 'ISAnnotationWithoutTrace/'))
                documents = coref_reader.read_file(input_f)
                for doc in documents:
                    doc.markable2mention(is_input)
                    doc.clusters_to_tokens()
                    doc.filename = f
                    doc.fullpath = os.path.join(dir, f)
                    self.corpus.append(doc)

    def cross_val_splits(self, n):
        # n should be 4 for just 40 doc test set, and 5 if using whole corpus
        for i in range(10):
            test = self.corpus[i * n: (i * n) + n]
            train = self.corpus[0: (i * n)] + self.corpus[(i * n) + n: len(self.corpus)]
            self.train_test_tup.append((train, test))

    def run_anaphoricity_exp(self, recognizer):
        for tup in self.train_test_tup:
            recognizer.train_classifier(tup[0])
            for test_file in tup[1]:
                print(test_file.filename)
                bridgepairs = self.anaphoricity_resolve(test_file, recognizer)  # write results in here?
                self.write_bridgepairs(bridgepairs, test_file.fullpath.replace(
                    'Coref_conll_IS_files_only', 'anaphoricity_sysout/').replace('.v4_auto_conll', '.bridge_sys'))

    def anaphoricity_resolve(self, doc, recognizer):
        bridges = []
        anaphors = [mention for sent in doc.sentences for mention in sent.mentions
                             if recognizer.recognize(mention, doc)]
        for anaphor in anaphors:
            bridges.append(WordPair(anaphor=anaphor, antecedent=anaphor))
        return bridges

    def write_bridgepairs(self, bridgepairlist, outfilepath):
        with open(outfilepath, 'w') as outfile:
            for bp in bridgepairlist:
                if bp.anaphor.markable_id and bp.antecedent.markable_id:
                    outfile.write('\t'.join([bp.anaphor.underscore_span(), bp.anaphor.markable_id,
                                             bp.antecedent.underscore_span(), bp.antecedent.markable_id]) + '\n')
if __name__ == '__main__':
    experimenter = Experimenter()
    experimenter.read_corpus()
    experimenter.cross_val_splits(5)

    recognizer = Recognizer()
    recognizer.load_feature_dict('features/anaphor_features.json')
    experimenter.run_anaphoricity_exp(recognizer)

    # TODO train antecedent selector

    # TODO run an experiemnt for antecedent selection with gold anaphors -- write predicted pairs to file, scorer on it as normal
    # TODO run an experiment for the combined anaphor recognition and antecedent selection -- as we were doing before.
    # TODO Add in features from the gigaword bootstrapping, other bootstrapping, Hou's prep pattern and log-liklihood ratio