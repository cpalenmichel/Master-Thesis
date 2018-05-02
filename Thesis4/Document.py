# Chester Palen-Michel
# Brandeis University
# 2/14/18

""" A class to represent the contents of a document"""
from collections import defaultdict, namedtuple

from Mention import Mention

WordPair = namedtuple('WordPair', ['anaphor', 'antecedent'])
class Document:

    def __init__(self):
        self.docID = None
        self.sentences = []
        self.clusters = defaultdict(list)  # {cluster_id : [m1, m2, m3]}
        self.ment_map = {}
        self.header = None
        self.end = None
        self.new_cluster_num = 0
        self.coref_to_ISoffset = {}
        self.mark2ment_map = {}
        self.gold_clusters = defaultdict(list)
        self.filename = None
        self.fullpath = None

    def get_doc_id(self, doc_id):
        self.docID = doc_id

    def make_ment_map(self):
        for sentence in self.sentences:
            self.ment_map[sentence.index] = {}
            for m in sentence.mentions:
                self.ment_map[sentence.index][(m.span_start, m.span_end)] = m

    def make_mark2ment_map(self):
        for sentence in self.sentences:
            for m in sentence.mentions:
                if m.markable_id is not None:
                    self.mark2ment_map[m.markable_id] = m

    def clusters_to_tokens(self):
        # Adds cluster ids from mentions to individual tokens. Use before writing to file.
        self.new_cluster_num +=1    # Set different from the other clustered mention ids
        for sent in self.sentences:
            temp = defaultdict(list)
            for m in sent.mentions:
                if m.cluster_id == -1:
                    m.cluster_id = self.new_cluster_num
                    self.new_cluster_num += 1
                if m.span_start == m.span_end:
                    temp[m.span_start].append('(' + str(m.cluster_id) + ')')
                else:
                    temp[m.span_start].append('(' + str(m.cluster_id))
                    temp[m.span_end].append(str(m.cluster_id) + ')')
            for key in temp:
                sent.words[key].cluster_id = '|'.join(temp[key])

    def write_lines(self):
        ret = [self.header]
        for s in self.sentences:
            for w in s.words:
                line = [w.file_name, w.part_number, str(w.index), w.token, w.pos, w.parse_bit, w.pred_lemma,
                        w.frame_id, w.word_sense, w.speaker, w.orig_name_ent]
                line.extend(w.pred_args)
                line.append(str(w.cluster_id))
                line = '\t'.join(line)
                ret.append(line + '\n')
            ret.append('\n')
        ret.append(self.end)
        return ret

    def markable2mention(self, isfile):
        for markable in isfile.markables:
            start_tuple = self.coref_to_ISoffset[markable.span_start]
            start = start_tuple[2]
            if start_tuple[0] != self.header:
                break # ensure this is the right document
            end_tuple = self.coref_to_ISoffset[markable.span_end]
            end = end_tuple[2]
            sent_index = start_tuple[1]
            found = False
            for m in self.sentences[sent_index].mentions:
                if m.span_start == start_tuple[2] and m.span_end == end_tuple[2]:
                    m.set_bridge_attributes(markable)
                    found = True
            if not found:
                new_mention = Mention(self.sentences[sent_index].words[start:end +1], start, end, sent_index)#,
                                      #self.sentences[sent_index].tree) TODO FASTER?
                new_mention.set_bridge_attributes(markable)
                self.sentences[sent_index].mentions.append(new_mention)
        self.make_mark2ment_map()

    def write_bridges(self):
        lines = []
        for s in self.sentences:
            for m in s.mentions:
                bridged_from = m.bridged_from
                if bridged_from is not None:
                    bridged_from_mention = self.mark2ment_map[bridged_from]
                    if bridged_from_mention.gold_cluster_id is not None:
                        bridge_antecedents = [m for m in
                                              self.gold_clusters[bridged_from_mention.gold_cluster_id] if m.markable_id is not None]
                    else:
                        bridge_antecedents = [bridged_from_mention]
                    # replace m.head.token with underscore_span()
                    lines.append('\t'.join([m.head.token, m.markable_id,
                                            ';'.join([m.head.token for m in bridge_antecedents]),
                                            '|'.join([m.markable_id for m in bridge_antecedents])]))
        return '\n'.join(lines)

    def get_bridgepairs(self):
        # Return a list of bridge_pairs in document
        ret = []
        for s in self.sentences:
            for m in s.mentions:
                if m.bridged_from:
                    antecedent = self.mark2ment_map[m.bridged_from]
                    ret.append(WordPair(m, antecedent))
                    if m.second_bridged_from:
                        ret.append(WordPair(m, self.mark2ment_map[m.second_bridged_from]))
                        if m.third_bridged_from:
                            ret.append(WordPair(m, self.mark2ment_map[m.third_bridged_from]))
        return ret