# Chester Palen-Michel
import codecs
import json
from collections import defaultdict

import os

from CoreReader import CoreReader
from XML_ISNotesReader import ISFile


class Extractor:
    """
    Base class for feature extractors.
    Loads ISNotes corpus. Writes feature dict to json file.
    """
    def __init__(self):
        self.instance_dict = defaultdict(dict)
        self.corpus = []

    def load_corpus(self):
        coref_reader = CoreReader()
        for dir, sub_dir, files in os.walk('Coref_conll_IS_files_only'):
            for f in files:
                input_f = codecs.open(os.path.join(dir, f), 'r', encoding='utf-8', errors='ignore')  # coref input file
                is_input = ISFile(os.path.join(dir, f).replace('.v4_auto_conll', '_entity_level.xml')
                                  .replace('Coref_conll_IS_files_only', 'ISAnnotationWithoutTrace/'))
                documents = coref_reader.read_file(input_f)
                for doc in documents:
                    doc.markable2mention(is_input)
                    doc.clusters_to_tokens()
                    doc.filename = f
                    self.corpus.append(doc)

    def ment_doc_instances(self):
       return [(ment, doc) for doc in self.corpus for sent in doc.sentences for ment in sent.mentions]

    def write_json(self, outfilepath):
        with open(outfilepath, 'w') as fp:
            json.dump(self.instance_dict, fp)


    def load_feature_dict(self):
        pass
        #TODO load a past feature_dict to update it.