# Chester Palen-Michel
# Brandeis University
# 2/14/18

""" A class to read in coreference data files and output a document class."""

from Document import *
from Sentence import Sentence
from Word import *
from XML_ISNotesReader import ISFile


class CoreReader:

    def __init__(self):
        pass

    def read_file(self, file):
        #read in a single file from corpus
        #parse to sentence class and word class but fully document class
        doc_to_return = Document()
        lines = file.readlines()
        pre_sent = []
        sent_index = 0
        is_offset = 1
        doc_list = []
        for line in lines:
            if not line.startswith('#'):
                if line.isspace():
                    doc_to_return.sentences.append(Sentence(pre_sent.copy(), sent_index))
                    pre_sent.clear()
                    sent_index +=1
                else:
                    entries = line.split()
                    # Set Doc Id
                    if not doc_to_return.docID:
                        doc_to_return.get_doc_id(entries[0])

                    # Construct word
                    word = Word()
                    word.set_entries(entries)
                    pre_sent.append(word)
                    # Create mapping for resolving difference in ISNotes offset and OntoNotes
                    #doc_to_return.coref_to_ISoffset[is_offset] = (doc_to_return.header, sent_index, int(entries[2]))
                    #is_offset += 1
            elif line.startswith('#begin'):
                doc_to_return = Document()
                doc_to_return.header = line
            else:
                doc_to_return.end = line
                doc_list.append(doc_to_return)
                sent_index = 0
                doc_to_return = Document()

        # for sent in doc_to_return.sentences:
        #     sent.get_names()  # May no longer be needed. since exists in make_mentions in sentence_init

        # Construct gold coreference clusters
        # for doc in doc_list:
        #     for sent in doc.sentences:
        #         for m in sent.mentions:
        #             if m.gold_cluster_id is not None:
        #                 doc.gold_clusters[m.gold_cluster_id].append(m)

        return doc_list

# For testing things early in development

if __name__ == '__main__':
    f = open('Coref_conll_IS_files_only/wsj_1004.v4_auto_conll')
    isfile = ISFile('ISAnnotationWithoutTrace/wsj_1004_entity_level.xml')
    coref_reader = CoreReader()
    doc_list = coref_reader.read_file(f)
    print('Testing file reader')
    print('length of doc list = ', len(doc_list))
    for doc in doc_list:
        doc.markable2mention(isfile)
        #print(doc.coref_to_ISoffset)
        for s in doc.sentences:
            print()
            print(s.index)
            for mention in s.mentions:
                print(mention)
    # for s in doc.sentences:
    #     print(s.mention_r2l_order())

        # print()
        # for w in s.words:
            # print(w.token)
            # print(w.name_ent)
         # for i in s.mentions:
         #     print(i)
             # for t in i.tokens:
             #     print(t)
    #
    # doc.make_ment_map()
    # print(doc.ment_map)
    # resolver = Resolver()
    # resolver.resolve(doc)
    # print(doc.clusters)
