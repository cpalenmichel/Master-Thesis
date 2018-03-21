# Chester Palen-Michel
# Brandeis University
# 2/14/18

""" A class to represent each sentence """
from nltk import Tree, defaultdict

from Mention import Mention


class Sentence:

    # Should have word objects in list, any features provided that are appropriate for the whole sentence.
    # a reconstruction of the syntactic tree might be useful here.
    
    def __init__(self, words, sent_index):
        self.words = words
        self.string_tokens = [w.token for w in words]
        self.pos_tokens = ['N' if tok.pos.startswith('N') else 'X' for tok in words]
        self.tree = self.build_tree(words)
        self.index = sent_index
        self.mentions = self.make_mentions(words)
        self.string_form = ' '.join(self.string_tokens)

    def __repr__(self):
        return ' '.join([w.token for w in self.words])

    def build_tree(self, words):
        str_to_tree = ""
        for w in words:
           str_to_tree = str_to_tree + w.parse_bit.replace('*', '(' + w.pos + ' ' + w.token + ')')
        return Tree.fromstring(str_to_tree)

    def make_mentions(self, words):
        ret = []
        self.get_names()  # get names any time mentions are made
        starts = defaultdict(list)
        ends = defaultdict(list)
        for w in words:
            start_ids=[int(id.replace(')', '').replace('(', '')) for id in w.mention.split('|') if id.startswith('(')]
            for id in start_ids:
                starts[id].append(w)
            end_ids = [int(id.replace(')', '').replace('(', '')) for id in w.mention.split('|') if id.endswith(')')]
            for id in end_ids:
                ends[id].append(w)

        for id in starts:
            while starts[id]:
                s = starts[id].pop()
                e = ends[id].pop()
                ret.append(Mention(words[s.index:e.index+1], s.index, e.index, self.index, self.tree, id))
                ret.reverse() # reverse so in sentence order
        return ret      #Should these mentions be in breadth first tree search order?


    def get_names(self):
        temp = '*'
        for w in self.words:
            if w.orig_name_ent.startswith('(') and w.orig_name_ent.endswith(')'):
                w.name_ent = w.orig_name_ent.replace('(', '').replace(')', '')
            elif w.orig_name_ent.startswith('('):
                temp = w.orig_name_ent.replace('(', '').replace('*', '')
                w.name_ent = temp
            elif w.orig_name_ent.endswith(')'):
                w.name_ent = temp
                temp = '*'
            else:
                w.name_ent = temp

    def mention_l2r_order(self):
        # Left right tree traversal order
        ret = []
        temp = self.mentions.copy()
        nps = [' '.join(sub.leaves())  for sub in self.tree.subtrees()]
        for np in nps:
            for men in temp:
                if men.str_men_match(np):
                    temp.remove(men)
                    ret.append(men)
        # This is not ideal, but need to include non-constituents that may be skipped.
        if temp:
            ret.extend(temp)
        return ret

    def mention_r2l_order(self):
        # Right to left tree traversal order
        ret = []
        temp = self.mentions.copy()
        nps = [' '.join(sub.leaves()) for sub in backwards_subtrees(self.tree)]
        for np in nps:
            for men in temp:
                if men.str_men_match(np):
                    temp.remove(men)
                    ret.append(men)
        # This is not ideal, but need to include non-constituents that may be skipped.
        if temp:
            ret.extend(temp)
        return ret

def backwards_subtrees(tree):
    yield tree
    for child in tree[::-1]:
        if isinstance(child, Tree):
            for subtree in backwards_subtrees(child):
                    yield subtree




