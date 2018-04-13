# Chester Palen-Michel
# Brandeis University
# 2/14/18

"""A class to represent each word and other features"""

class Word():
    # includes each field from a line of a document.
    # parse_bit, mention, named-entity, and pred-arg may not be useful at the word level, but is avaiblable anyhow.

    def __init__(self):
        #self.file_name = None
        #self.part_number = None
        self.token = None
        self.index = None
        self.pos = None
        #self.parse_bit = None
        #self.pred_lemma = None
        #self.frame_id = None
        #self.word_sense = None
        #self.speaker = None
        #self.orig_name_ent = None
        #self.pred_args = None
        #self.mention = None
        #self.cluster_id = '-'
        #self.name_ent = None
        #self.markable_id = None

    def set_entries(self, entries):
        #self.file_name = entries[0]
        #self.part_number = entries[1]
        self.index = int(entries[2])
        self.token = entries[3]
        self.pos = entries[4]
        #self.parse_bit = entries[5]
        #self.pred_lemma = entries[6]
        #self.frame_id = entries[7]
        #self.word_sense = entries[8]
        #self.speaker = entries[9]
        #self.orig_name_ent = entries[10]
        #self.pred_args = entries[11: -1]
        #self.mention = entries[-1]

    def __repr__(self):
        return self.token