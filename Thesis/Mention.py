# Chester Palen-Michel
# Brandeis University
# 2/14/18

""" Class to represent an individual mention rather than an individual token which I represent with Word.
    Features are as follows:
    number: 's','p'
    gender: 'm', 'f'
    person: 1, 2, 3
    animacy: 'a', 'i'
    NER: 'GPE', 'LOC', 'PERSON', 'ORG', 'DATE', 'TIME', 'NORP', etc

    None is also possible if the feature is undefined and should be treated as a wildcard.
"""
from nltk import Tree
from nltk.corpus import names

class Mention(object):

    def __init__(self, tokens=None, span_start=None, span_end=None, sentence_index=None, tree=None, gold_id =None):
        self.tokens = tokens
        self.token_str_list = [t.token for t in self.tokens]
        self.span_start = span_start
        self.span_end = span_end
        self.sentence_index = sentence_index
        self.cluster_id = -1 # -1 for singleton/no cluster yet.
        self.first_in_cluster = True # set to False if not the first mention in cluster.
        self.subtree = self.make_subtree(tree)
        self.head = self.find_head(self.subtree)
        self.features = {'number': None, 'gender': None, 'person': None, 'animacy': None, 'NER': None}
        self.gold_cluster_id = gold_id

        # bridging attributes
        # id
        self.markable_id = None
        self.is_status = None
        self.mediated_type = None
        # Bridging only attributes
        self.bridged_from = None
        self.bridge_type = None
        self.second_bridged_from = None
        self.third_bridged_from = None

        # Pronouns
        self.female_pronouns = {'she', 'her', 'hers', 'herself'}
        self.male_pronouns = {'he', 'him', 'his', 'himself'}
        self.sing_pronouns = {'i', 'me', 'she', 'he', 'her', 'him', 'it', 'this', 'that', 'myself', 'yourself',
                              'himself', 'herself', 'itself'}
        self.plural_pronouns = {'we', 'us', 'they', 'them', 'their' 'those', 'these', 'ourselves', 'yourselves', 'themselves'}
        self.per1_pronouns = {'i', 'we', 'me', 'us', 'myself', 'ourselves'}
        self.per2_pronouns = {'you'}
        self.per3_pronouns = {'she', 'he', 'it', 'its', 'her', 'him', 'they', 'their', 'them', 'this', 'that'}
        self.animate_pronouns = {'i', 'you', 'he', 'she', 'we', 'him', 'her', 'me', 'us'}
        self.inanimate_pronouns = {'it', 'this', 'these', 'those'}
        self.relative_pronouns = {'that', 'which', 'who', 'whom', 'whose'}
        self.reflexive_pronouns = {'myself', 'ourselves', 'yourself', 'yourselves', 'himself', 'herself', 'itself'}
        self.possessive_pronouns = {'mine', 'ours', 'yours', 'his', 'hers', 'theirs'}
        self.pronouns = self.female_pronouns.union(self.male_pronouns, self.sing_pronouns, self.plural_pronouns,
                                                   self.per1_pronouns, self.per2_pronouns, self.per3_pronouns,
                                                   self.animate_pronouns, self.inanimate_pronouns,
                                                   self.relative_pronouns, self.reflexive_pronouns,
                                                   self.possessive_pronouns)
        self.female_names = names.words('female.txt')
        self.male_names = names.words('male.txt')
        self.set_features()

    def __repr__(self):
        return ' '.join([t.token for t in self.tokens])

    def str_men_match(self, string):
        return self.tok_span() == string

    def tok_span(self):
        return ' '.join([t.token for t in self.tokens])

    def underscore_span(self):
        return '_'.join([t.token for t in self.tokens])

    def set_features(self):
        head = self.find_head(self.subtree)
        if head is not None:
            # NER
            # check NER of head, otherwise leave wildcard.
            if head.name_ent == '*':
                self.features['NER'] = None
            else:
                self.features['NER'] = head.name_ent
            # Number, Gender, Animacy and Person features
            self.number_features(head)
            self.gender_features(head)
            self.person_features(head)
            self.animate_features(head)

    def animate_features(self, head):
        if head.pos.startswith('PRP'):
            if head.token.lower() in self.animate_pronouns:
                self.features['animacy'] = 'a'
            elif head.token.lower() in self.inanimate_pronouns:
                self.features['animacy'] = 'i'
        elif head.name_ent == 'PERSON':
            self.features['animacy'] = 'a'
        elif head.name_ent == 'GPE' or head.name_ent == 'LOC' or head.name_ent == 'DATE' or head.name_ent == 'TIME':
            self.features['animacy'] = 'i'

    def person_features(self, head):
        if head.pos.startswith('PRP'):
            # Person
            if head.token.lower() in self.per1_pronouns:
                self.features['person'] = 1
            elif head.token.lower() in self.per2_pronouns:
                self.features['person'] = 2
            elif head.token.lower() in self.per3_pronouns:
                self.features['person'] = 3

    def gender_features(self, head):
        if head.pos.startswith('PRP'):
            if head.token.lower() in self.male_pronouns:
                self.features['gender'] = 'm'
            elif head.token.lower() in self.female_pronouns:
                self.features['gender'] = 'f'
        # Gender if not pronoun stop searching after first match with names since first name more likely to
        # indicate gender
        elif head.name_ent == 'PERSON':
            for t in self.tokens:
                if t.token in self.male_names:
                    self.features['gender'] = 'm'
                    break
                elif t.token in self.female_names:
                    self.features['gender'] = 'f'
                    break
                elif t.token == 'Mr.':
                    self.features['gender'] = 'm'
                elif t.token == 'Mrs.' or t.token == 'Ms.' or t.token == 'Miss':
                    self.features['gender'] = 'f'
        else:
            self.features['gender'] = None

    def number_features(self, head):
        # Org could be sing or plural
        if head.name_ent == 'ORG':
            self.features['number'] = None
        elif head.pos.startswith('PRP'):
            # Number
            if head.token.lower() in self.sing_pronouns:
                self.features['number'] = 's'
            elif head.token.lower() in self.plural_pronouns:
                self.features['number'] = 'p'
         # Number if not pronoun
        elif head.pos == 'NNPS' or head.pos == 'NNS':
            self.features['number'] = 'p'
        elif head.pos == 'NNP' or head.pos == 'NN':
            self.features['number'] = 's'
        else:
            self.features['number'] = None

    def find_head(self, np):

        noun_tags = ['NN', 'NNS', 'NNP', 'NNPS', 'PRP$', 'PRP']
        if np == None:
           ret = None
        elif np.label() == 'NP' or np.label() == 'PRP$' or np.label() == 'PRP':
            top_level_trees = [np[i] for i in range(len(np)) if type(np[i]) is Tree]
            # search for a top-level noun
            top_level_nouns = [t for t in top_level_trees if t.label() in noun_tags]
            if len(top_level_nouns) > 0:
                # if you find some, pick the rightmost one
                ret = top_level_nouns[-1][0]
            else:
                # search for a top-level np
                top_level_nps = [t for t in top_level_trees if t.label() == 'NP']
                if len(top_level_nps) > 0:
                    # if you find some, pick the head of the rightmost one
                    ret =  self.find_head(top_level_nps[-1])
                else:
                    # search for any noun
                    nouns = [p[0] for p in np.pos() if p[1] in noun_tags]
                    if len(nouns) > 0:
                        # Choose right most
                        ret = nouns[-1]
                    else:
                        # return the rightmost word
                        ret = np.leaves()[-1]
        else: ret = None # TODO Figure out how to make this work for non-nouns eventually
        # So can have the token object and not just string
        tok = [tok for tok in self.tokens if ret == tok.token]
        if tok:
            return tok[0]
        elif len(self.tokens) == 1:
            return self.tokens[0]
        else:
            return None

    def make_subtree(self, tree):
        # Returns subtree that matches the string extent of mention
        # May return none if annotation of mention is not a constituent
        if tree != None and type(tree) == Tree:
            for sub in tree.subtrees():
                if self.str_men_match(' '.join(sub.leaves())):
                    return sub

    def set_bridge_attributes(self, markable):
        # Sets mention's bridging attributes with values from a markable from IS Notes
        self.markable_id = markable.id
        self.is_status = markable.is_status
        self.mediated_type = markable.mediated_type
        # Bridging only attributes
        self.bridged_from = markable.bridged_from
        self.bridge_type = markable.bridge_type
        self.second_bridged_from = markable.second_bridged_from
        self.third_bridged_from = markable.third_bridged_from

    def isNP(self):
        # Returns true if mention's subtree is a Noun/NounPhrase
        if self.subtree:
            label = self.subtree.label()
            return label.startswith('N')
        else:
            return False