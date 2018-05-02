# Chester Palen-Michel
# 4/28/18
import json
import re
from nltk.corpus import gazetteers

from feature_extractor import Extractor


class AnaphorExtractor(Extractor):
    def __init__(self):
        super(AnaphorExtractor, self).__init__()
        self.pronouns = {'myself', 'ourselves', 'yourself', 'yourselves', 'himself', 'herself', 'itself', 'mine',
                         'ours', 'yours', 'his', 'hers', 'theirs', 'that', 'which', 'who', 'whom', 'whose', 'i', 'me',
                         'she', 'he', 'her', 'him', 'it', 'this', 'that', 'myself', 'yourself', 'himself', 'herself',
                         'itself', 'we', 'us', 'they', 'them', 'their' 'those', 'these', 'ourselves', 'yourselves',
                         'themselves', 'our', 'my', 'your', 'all', 'another', 'any', 'anybody', 'anyone', 'anything',
                         'both', 'each', 'either', 'everybody', 'everyone', 'everything', 'few', 'little', 'many',
                         'more', 'much', 'most', 'nobody', 'none', 'nothing', 'neither', 'one', 'other', 'others',
                         'some', 'somebody', 'something', 'someone', 'this', 'that', 'these', 'those'}
        self.comparatives = {'other', 'another', 'such', 'different', 'similar', 'additional', 'comparable',
                             'same', 'further', 'extra'}
        self.numbers = {'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten', 'twenty',
                        'thirty', 'forty', 'fifty', 'sixty', 'seventy', 'eighty', 'ninety', 'hundreds', 'thousands',
                        'millions'}
        self.demonstratives = {'that', 'this', 'those', 'these'}
        self.indef = {'a', 'an', 'one', 'some', 'any', 'either', 'neither', 'no', 'all', 'each', 'every', 'another',
                      'whichever', 'which', 'what', 'whatever'}
        self.spatial_temporal = {'ﬁnal', 'ﬁrst', 'last', 'next', 'prior', 'succeed', 'second', 'nearby', 'previous',
                                 'close', 'above', 'adjacent', 'behind', 'below', 'bottom', 'early', 'formal', 'future',
                                 'before', 'after', 'earlier', 'later'}
        self.poss_pronouns = {'my', 'your', 'his', 'her', 'our', 'their', 'its'}
        self.possessives = {'\'s', 's\''}
        self.general_quantifier = {'all', 'no', 'neither', 'every', 'any', 'most'}
        self.inq_building_parts = None
        self.inq_roles = None
        self.sem_per = None
        self.sem_rel = None
        self.sem_role = None
        self.sem_loc = None
        self.sem_org = None

    def get_features(self):
        for instance in self.ment_doc_instances():
            ment = instance[0]
            doc = instance[1]
            self.instance_dict[self.keyify(ment, doc)] = self.make_features(ment, doc)

    def keyify(self, ment, doc):
        return ment.underscore_span() + ';' + doc.docID + ';' +  str(ment.sentence_index)

    def make_features(self, mention, doc):
        feature_vec = {
            'anaphor_pp': self.anaphor_pp(mention),
            'definite': self.definite(mention),
            'determiner': self.determiner(mention),
            'len_anaphor': self.len_mention(mention),  # may be bad?
            'spatial_temporal_mod': self.spatial_temporal_mod(mention),
            'no_premod_comp': self.no_comp_premod(mention),
            'np_type': self.np_type(mention),
            'np_num': self.np_number(mention),
            'sent_first': self.sent_first(mention, doc),
            'has_child': self.child_ments(mention, self.get_mentions([doc])),
            'is_child': self.is_child_ment(mention, self.get_mentions([doc])),
            'premod_countries': self.premod_countries(mention),
            'set_element': self.is_set_element(mention),
            'building_parts': self.isbuilding(mention),
            'inq_roles': self.isinqrole(mention),
            'premod_gen_quant': self.premod_gen_quant(mention),
            'isYear': self.is_year(mention),
            'ifclause': self.ifclause(mention, doc),
            'modalpresent': self.modality(mention, doc),
            'sem_category': self.sem_category(mention),
            'gram_category': self.gram_category(mention),
            'arg_ratio': self.arg_ratio_threshold(mention),
            'gold_chain_len': len(doc.gold_clusters[mention.gold_cluster_id]),
            'wordnet_relational': self.wordnet_relational(mention)
        }
        feature_vec.update(self.prevment_features(mention, doc))
        feature_vec.update(self.unigrams(mention))
        return feature_vec

    def anaphor_pp(self, mention):
        for token in mention.tokens:
            if token.pos == 'IN':
                return 1
        return 0

    def definite(self, mention):
        definites = {'the', 'that', 'this', 'those', 'these', 'one'}
        for w in mention.tokens:
            if w.token.lower() in definites:
                return 1
        return 0

    def determiner(self, mention):
        premods = self.premod(mention)
        if set(premods).intersection(self.possessives):
            return 'poss'
        elif set(premods).intersection(self.poss_pronouns):
            return 'pos_pro'
        elif 'the' in premods:
            return 'def'
        elif set(premods).intersection(self.indef):
            return 'indef'
        elif set(premods).intersection(self.demonstratives):
            return 'demon'
        else:
            return 'no_det'

    def len_mention(self, mention):
        if len(mention.tokens) <= 2:
            return len(mention.tokens)
        elif len(mention.tokens) <= 7:
            return 4
        elif len(mention.tokens) <= 14:
            return 5
        else:
            return 6

    def unigrams(self, mention):
        return dict([(w, True)for w in mention.token_str_list])

    def prevment_features(self, mention, doc):
        fullprevmatch = False
        fullprevtime = 0
        partialpre = False
        partialtime = 0
        contentpre = False
        doc_first_ment = True if not self.prev_mentions(mention, doc) else False

        for m in self.prev_mentions(mention, doc):
            if m.underscore_span() == mention.underscore_span():
                fullprevmatch = True
                fullprevtime += 1
            if m.head == mention.head:
                partialpre = True
                partialtime += 1
            if set(self.content_words(m)).intersection(set(self.content_words(mention))):
                contentpre = True

        return {'fullprevment': fullprevmatch,
                'fullprevtime': self.bin_times(fullprevtime),
                'partialpre': partialpre,
                'partialtime': self.bin_times(partialtime),
                'contentpre': contentpre,
                'doc_first_ment': doc_first_ment,
                'coherence_gap': self.coherence_gap(mention)
                }

    def coherence_gap(self, mention):
        return mention.gold_cluster_id is None and self.no_comp_premod(mention) and self.no_prop_names(mention)

    def spatial_temporal_mod(self, mention):
        return True if set(mention.token_str_list).intersection(self.spatial_temporal) else False

    def np_type(self, mention):
        head = mention.head
        if head.pos == 'NN' or head.pos == 'NNS':
            return 'common'
        elif head.pos == 'NNP' or head.pos == 'NNPS':
            return 'proper'
        elif head.pos.startswith('PRP') or head.token in self.pronouns:
            return 'pronoun'
        else:
            return 'other'

    def np_number(self, mention):
        if mention.head.pos == 'NNS' or mention.head.pos == 'NNPS':
            return 'plural'
        elif mention.head.pos == 'NN' or mention.head.pos == 'NNP':
            return 'sing'
        else:
            return 'other'

    def bin_times(self, num_times):
        if num_times > 2:
            return 'many'
        else:
            return str(num_times)

    def prev_mentions(self, mention, doc):
        same_sent_mentions = [m for m in doc.sentences[mention.sentence_index].mentions if
                              m.span_start < mention.span_start]
        prev_sent_mentions = [m for s in range(0, mention.sentence_index) for m in doc.sentences[s].mentions]
        return same_sent_mentions + prev_sent_mentions

    def no_comp_premod(self, mention):
        return False if set(mention.token_str_list).intersection(self.comparatives) else True

    def no_prop_names(self, mention):
        for t in mention.tokens:
            if t.pos == 'NNP' or t.pos == 'NNPS':
                return True
        return False

    def content_words(self, mention):
        return [t.token for t in mention.tokens if t.pos in {'NN', 'NNS', 'NNP', 'NNPS', 'JJ'}]

    def premod(self, mention):
        return mention.token_str_list[:mention.token_str_list.index(mention.head.token)]

    def get_mentions(self, docs):
        ret = []
        for doc in docs:
            for sent in doc.sentences:
                ret.extend(sent.mentions)
        return ret

    def sent_first(self, mention, doc):
        # if no other mention has a lower span index, it is the first mention in sentence
        for m in doc.sentences[mention.sentence_index].mentions:
            if m.span_start < mention.span_start:
                return False
        return True

    def child_ments(self, mention, mentions):
        for ment in mentions:
            if mention.span_start <= ment.span_start and mention.span_end >= ment.span_end:
                return True
            return False

    def is_child_ment(self, mention, mentions):
        for ment in mentions:
            if ment.span_start <= mention.span_start and ment.span_end >= mention.span_end:
                return True
        return False

    def premod_countries(self, mention):
        for word in self.premod(mention):
            if word in gazetteers.words('countries.txt'):
                return True
        return False

    def is_set_element(self, mention):
        if mention.head.token in {'one', 'some', 'none', 'many', 'most'}.union(self.indef) or \
                        'each' in self.premod(mention) or 'one' in self.premod(
            mention) or mention.head.token in self.numbers:
            return True
        else:
            return False

    def premod_gen_quant(self, mention):
        return True if set(self.premod(mention)).intersection(self.general_quantifier) else False

    def is_year(self, mention):
        if re.findall(r'^\d\d\d\d$', mention.head.token):
            return True
        else:
            return False

    def ifclause(self, mention, doc):
        for w in doc.sentences[mention.sentence_index].words:
            if w.token == 'if':
                return True
        return False

    def modality(self, mention, doc):
        for w in doc.sentences[mention.sentence_index].words:
            if w.pos == 'MD':
                return True
        return False

    def gram_category(self, mention):
        if mention.head.spacy_tok is not None:
            print(mention.head.spacy_tok.dep_)
            return self.bin_gram_role(mention.head.spacy_tok.dep_)
        else:
            return 'na'

    def bin_gram_role(self, dependency):
        if dependency == 'nsubj':
            return 'nsubj'
        elif dependency == 'dobj':
            return 'dobj'
        elif dependency == 'pobj':
            return 'pobj'
        elif dependency == 'nsubjpass':
            return 'nsubjpass'
        elif dependency == 'oprd' or dependency == 'prd':
            return 'pred'
        else:
            return 'other'

    def wordnet_relational(self, mention):
        return mention.head.token in self.relationalwords

    def isbuilding(self, mention):
        return mention.head.token in self.inq_building_parts

    def isinqrole(self, mention):
        return mention.head.token in self.inq_roles

    def sem_category(self, mention):
        if mention.head.name_ent is not None and mention.head.name_ent != '*':
            return mention.head.name_ent
        elif mention.head.token in self.sem_rel:
            return 'RelPer'
        elif mention.head.token in self.sem_role:
            return 'RolePer'
        elif mention.head.token in self.sem_per:
            return 'PERSON'
        elif mention.head.token in self.sem_org:
            return 'ORG'
        elif mention.head.token in self.sem_loc:
            return 'LOC'
        else:
            return 'NA'

    def load_inquirer(self, filepath, sem_type):
        # use type to specify which type of list we are loading
        # types: building, role
        with open(filepath, 'r') as inq_f:
            lines = [line.strip().lower() for line in inq_f.readlines()]
            if sem_type == 'building':
                self.inq_building_parts = lines
            elif sem_type == 'role':
                self.inq_roles = lines

    def load_sem_class(self, filepath, sem_type):
        # use type to specify which type of list we are loading
        # types: per, role, rel, org, loc
        with open(filepath, 'r') as sem_f:
            lines = [line.strip().lower() for line in sem_f.readlines()]
            if sem_type == 'per':
                self.sem_per = lines
            elif sem_type == 'role':
                self.sem_role = lines
            elif sem_type == 'rel':
                self.sem_rel = lines
            elif sem_type == 'org':
                self.sem_org = lines
            elif sem_type == 'loc':
                self.sem_loc = lines

    def load_arg_ratios(self, filepath):
        self.arg_ratios = json.load(open(filepath, 'r'))

    def arg_ratio_threshold(self, mention):
        if mention.head in self.arg_ratios:
            if self.arg_ratios[mention.head] > .5:
                return True
            else:
                return False
        else:
            return False

    def load_wordnet_relationals(self, filepath):
        relationals_f = open(filepath, 'r')
        lines = relationals_f.readlines()
        self.relationalwords = [line.strip() for line in lines]

if __name__ == '__main__':
    extractor = AnaphorExtractor()
    extractor.load_corpus()

    # Load Resources
    extractor.load_inquirer('../Resources_Thesis/inq_building_parts.txt', 'building')
    extractor.load_inquirer('../Resources_Thesis/inq_role.txt', 'role')
    extractor.load_sem_class('../Resources_Thesis/role_sem_class.txt', 'role')
    extractor.load_sem_class('../Resources_Thesis/person_sem_class.txt', 'per')
    extractor.load_sem_class('../Resources_Thesis/relative_sem_class.txt', 'rel')
    extractor.load_sem_class('../Resources_Thesis/location_sem_class.txt', 'loc')
    extractor.load_sem_class('../Resources_Thesis/organization_sem_class.txt', 'org')
    extractor.load_arg_ratios('../Resources_Thesis/arg_ratios.json')
    extractor.load_wordnet_relationals('../Resources_Thesis/nombank_wordnet_relationals.txt')

    extractor.get_features()
    extractor.write_json('features/anaphor_features.json')
