# Chester Palen-Michel
# 4/7/2018
import codecs
import os

from Document import Document
from Sentence import Sentence
from Word import Word
from nltk.corpus import TaggedCorpusReader

class AncReader:
    def __init__(self):
        pass

    def read_file(self, file):

        lines = file.readlines()
        sent_index = 0
        doc_to_return = Document()
        for line in lines:
            if not line.isspace():
                words = []
                tokens = line.split()
                for i in range(len(tokens)):
                    word_pos = tokens[i].split('_')
                    word = Word()
                    word.token = word_pos[0]
                    word.index = i
                    if len(word_pos) > 1:
                        word.pos = word_pos[1]
                    else:
                        word.pos = '_'
                    words.append(word)
                sent = Sentence(words[:], sent_index)
                sent_index += 1
                doc_to_return.sentences.append(sent)
                words.clear()
        return [doc_to_return]

    def read_dir(self):
        # read entire directory as a single doc?
        corpus = TaggedCorpusReader('../../ANC', '.*', '_')
        corpus.tagged_sents()

if __name__ == '__main__':
    anc_reader = AncReader()
    input_dir = "../../ANC"
    # read in entire corpus
    corpus = []
    for dir, sub_dir, files in os.walk(input_dir):
        for f in files:
            input_f = codecs.open(os.path.join(dir, f), 'r', encoding='utf-8', errors='ignore')  # coref input file
            documents = anc_reader.read_file(input_f)
            for doc in documents:
                corpus.append(doc)