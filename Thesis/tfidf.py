# Chester Palen-Michel
# Brandeis University
# 4/9/18
import codecs
import os

from collections import defaultdict
from collections import Counter

from math import log

from CoreReader import CoreReader


class Tfidf:
    def __init__(self):
        self.doc_word_dict = defaultdict(Counter)
        self.doc_names = []

    def tf(self, word, doc):
        # count of word in doc. dict[doc][word] / sum(dict[doc][word] for word in dict[doc]
        return float(self.doc_word_dict[doc][word]) / sum([self.doc_word_dict[doc][w] for w in self.doc_word_dict[doc]])

    def idf(self, word, doc):
        return log( float(len(self.doc_names)) / len([d for d in self.doc_word_dict if word in self.doc_word_dict[d]]))
        # math.log(len(docs) / # of docs containing word, so len([doc for doc in dict[docs] if word in dict[doc])

    def load_docs(self, dir_path):
        # use corereader to open all docs count their words and put in the main dict.
        coref_reader = CoreReader()
        for dir, sub_dir, files in os.walk(dir_path):
            for f in files:
                input_f = codecs.open(os.path.join(dir, f), 'r', encoding='utf-8', errors='ignore')  # coref input file
                documents = coref_reader.read_file(input_f)
                self.doc_names.append(f)
                for doc in documents:
                    self.doc_word_dict[f].update([w.token for sent in doc.sentences for w in sent.words])

    def tf_idf(self, word, doc):
        return self.tf(word, doc) * self.idf(word, doc)

    def top_n(self, doc, n):
        # given a doc, find tf-idf for each word in doc, put in a list and sort it then take the top n in the list.
        # could do a variant taking the top % of the words in doc.
        word_scores = [(w, self.tf_idf(w, doc)) for w in self.doc_word_dict[doc]]
        word_scores.sort(key=lambda tup: tup[1], reverse=True)
        return word_scores[:n]

    def top_n_dict(self, n):
        d = {}
        for f in self.doc_names:
            d.update({f:[tup[0] for tup in self.top_n(f, n)]})
        return d

if __name__ == '__main__':

    tfidf = Tfidf()
    tfidf.load_docs('Coref_conll_IS_files_only/')
    for f in tfidf.doc_names:
        print('file: ', f, 'words: ', [tup[0] for tup in tfidf.top_n(f, 25)]) #TODO could try variations of how many to take?