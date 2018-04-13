# Chester Palen-Michel
# 2/22/18
# Sieve class

from nltk import Tree
from nltk.corpus import stopwords
import codecs


class Sieve:
    def __init__(self):
        self.female_pronouns = {'she', 'her', 'hers', 'herself'}
        self.male_pronouns = {'he', 'him', 'his', 'himself'}
        self.sing_pronouns = {'i', 'me', 'she', 'he', 'her', 'him', 'it', 'its' 'this', 'that', 'myself', 'yourself',
                              'himself', 'herself', 'itself'}
        self.plural_pronouns = {'we', 'us', 'they', 'their', 'them', 'those', 'these', 'ourselves', 'yourselves',
                                'themselves'}
        self.per1_pronouns = {'i', 'we', 'me', 'us', 'myself', 'ourselves'}
        self.per2_pronouns = {'you'}
        self.per3_pronouns = {'she', 'he', 'it', 'its', 'her', 'him', 'they', 'their', 'them', 'this', 'that'}
        self.animate_pronouns = {'i', 'you', 'he', 'she', 'we', 'him', 'her', 'me', 'us'}
        self.inanimate_pronouns = {'it', 'this', 'these', 'those', 'its'}
        self.relative_pronouns = {'that', 'which', 'who', 'whom', 'whose'}
        self.reflexive_pronouns = {'myself', 'ourselves', 'yourself', 'yourselves', 'himself', 'herself', 'itself'}
        self.possessive_pronouns = {'mine', 'ours', 'yours', 'his', 'hers', 'theirs', 'its'}
        self.pronouns = self.female_pronouns.union(self.male_pronouns, self.sing_pronouns, self.plural_pronouns,
                                                   self.per1_pronouns, self.per2_pronouns, self.per3_pronouns,
                                                   self.animate_pronouns, self.inanimate_pronouns,
                                                   self.relative_pronouns, self.reflexive_pronouns,
                                                   self.possessive_pronouns)

    def coref(self, anaphor, antecedent, doc):
        # Used to determine if 2 mentions should be clustered together.
        pass

    def cluster_path(self, doc):
        # the typical traversal to check if mentions are co-referent
        for s in doc.sentences:
            ments = s.mention_l2r_order()
            for i, anaphor in enumerate(ments):
                # only check prev mentions that are singleton or first in cluster
                if anaphor.first_in_cluster:
                    resolved = False
                    # check for coreference in same sentence
                    prev_index = i-1
                    while(not resolved) and prev_index >= 0:
                        resolved = self.coref(anaphor, ments[prev_index], doc)
                        if not resolved:
                            prev_index -= 1
                    if resolved:
                        self.check_and_set_clusters(doc, anaphor, ments[prev_index], doc.new_cluster_num)
                    else:
                        # if still not resolved, look at previous sentences
                        j = s.index -1
                        while (not resolved) and j >= 0:
                            #j = j-1
                            k = 0
                            prev_sent_ments = doc.sentences[j].mention_r2l_order()
                            while(not resolved) and k < len(prev_sent_ments):
                                resolved = self.coref(anaphor, prev_sent_ments[k], doc)
                                if resolved:
                                    self.check_and_set_clusters(doc, anaphor, prev_sent_ments[k], doc.new_cluster_num)
                                k += 1
                            j-=1

    def set_clusters(self, doc, anaphor, antecedent, id):
        # Track whether first in cluster
        if doc.clusters[id]:
            anaphor.first_in_cluster = False
            antecedent.first_in_cluster = False
        else:
            anaphor.first_in_cluster = False
        # add un-clustered mentions to clusters
        if antecedent.cluster_id == -1:
            doc.clusters[id].append(antecedent)
        elif anaphor.cluster_id == -1:
            doc.clusters[id].append(anaphor)
        # merge clusters if cluster ids are different
        elif antecedent.cluster_id != anaphor.cluster_id:
            doc.clusters[antecedent.cluster_id].extend(doc.clusters[anaphor.cluster_id])
            del doc.clusters[anaphor.cluster_id]
        # assign cluster id to mentions
        antecedent.cluster_id = id
        anaphor.cluster_id = id

    def get_clusterid(self, anaphor, antecedent):
        if antecedent.cluster_id != -1:
            return antecedent.cluster_id
        elif anaphor.cluster_id != -1:
            return anaphor.cluster_id
        else:
            return -1

    def check_and_set_clusters(self, doc, anaphor, antecedent, cluster_id):
        temp_cluster_id = self.get_clusterid(anaphor, antecedent)
        if temp_cluster_id != -1:
            self.set_clusters(doc, anaphor, antecedent, temp_cluster_id)
        else:
            doc.new_cluster_num += 1
            self.set_clusters(doc, anaphor, antecedent, doc.new_cluster_num)


class ExactMatchSieve(Sieve):
    # Only clusters mentions that have exact string match. Higher F-score if pronouns allowed, but at cost of precision.
    def coref(self, anaphor, antecedent, doc):
        return anaphor.tok_span() == antecedent.tok_span() and \
               anaphor.tok_span().lower() not in self.pronouns and antecedent.tok_span().lower() not in self.pronouns

class PronounMatch(Sieve):

    def coref(self, anaphor, antecedent, doc):
        if anaphor.tok_span().lower() in self.pronouns:
            # Prep anaphor for matching
            # check anaphor is pronoun.
            if anaphor.cluster_id == -1:
                united_anaphor = self.unify_features([anaphor])
            else:
                united_anaphor = self.unify_features(doc.clusters[anaphor.cluster_id])
            # Prep antecedent for matching
            # check antecedent is in cluster or not and unify features.
            if antecedent.cluster_id == -1:
                united_antecedent = self.unify_features([antecedent])
            else:
                united_antecedent = self.unify_features(doc.clusters[antecedent.cluster_id])
            return self.check_features(united_anaphor, united_antecedent)
        else: return False

    def check_features(self, anaph_feats, antec_feats):
        for f in anaph_feats:
            if not self.check_feat_set(anaph_feats[f], antec_feats[f]):
                return False
        return True

    def check_feat_set(self, s1, s2):
        if len(s1) == 0 or len(s2) == 0:
            return True
        else:
            return bool(s1.intersection(s2))

    def unify_features(self, men_list):
        ret = {'number': set() , 'gender': set(), 'person': set(), 'animacy': set(), 'NER': set()}
        # take each cluster and add features to the features unless None, in which case skip and leave empty
        for mention in men_list:
            for feature in mention.features:
                value = mention.features[feature]
                if value is not None:
                    ret[feature].add(value)
        return ret

class StrictHeadMatch(Sieve):
    # True if
    # ~~head of anaphor, matches any head in antecedent cluster.
    # ~~word inclusion, remove stopwords from both. if all anaphor words in antecedent: True, else false NOTE this is for entire cluster content words
    # --compatible modifiers? if can find all noun and adj before head, check they are same?
    # TODO add compatible modifiers restriction to actually make strict match, then take away for and sieve 5 as in paper
    # ~~not i within i between anaphor and antecedent
    # ban pronouns

    def coref(self, anaphor, antecedent, doc):

        return self.head_match(anaphor, antecedent, doc) and not self.i_within_i(anaphor, antecedent, doc) and \
               self.word_inclusion(anaphor, antecedent, doc) and anaphor.tok_span().lower() not in self.pronouns

    def word_inclusion(self, anaphor, antecedent, doc):
        stops = stopwords.words()
        anaph_content = [t.token for t in anaphor.tokens if t.token not in stops]
        if antecedent.cluster_id == -1:
            ante_content = [t.token for t in antecedent.tokens if t.token not in stops]
        else:
            ante_content = []
            for m in doc.clusters[antecedent.cluster_id]:
                ante_content.extend([t.token for t in m.tokens if t.token not in stops])
        for w in anaph_content:
            if w not in ante_content:
                return False
        return True

    def head_match(self, anaphor, antecedent, doc):
        head_anaph = anaphor.head
        if head_anaph is not None:
            if antecedent.cluster_id == -1:
                if antecedent.head is not None:
                    heads_antec = [antecedent.head.token]
                else:
                    heads_antec = []
            else:
                heads_antec = [a.head.token for a in doc.clusters[antecedent.cluster_id]
                               if a.head is not None]
            return head_anaph.token in heads_antec
        else:
            return False

    def i_within_i(self, anaphor, antecedent, doc):
        if anaphor.sentence_index != antecedent.sentence_index:
            return False
        else:
            t = doc.sentences[anaphor.sentence_index].tree

            def isearch(tree, m1, m2):
                for child in tree:
                    if type(child) is Tree:
                        if ' '.join(child.leaves()) == m1.tok_span():
                            match = ' '.join(tree.leaves()).find(m2.tok_span())
                            if match != -1:
                                return True
                        isearch(child, m1, m2)
                return False

            return isearch(t, anaphor, antecedent) or isearch(t, antecedent, anaphor)

class VariantHeadMatch(StrictHeadMatch):
    # Strict head match but without word inclusion restriction.
    def coref(self, anaphor, antecedent, doc):
        return self.head_match(anaphor, antecedent, doc) and not self.i_within_i(anaphor, antecedent, doc) and \
               anaphor.tok_span().lower() not in self.pronouns

class PreciseConstructs(Sieve):
    def __init__(self):
        Sieve.__init__(self)
        self.demonyms = None

    def coref(self, anaphor, antecedent, doc):
        return self.is_demonym(anaphor, antecedent, doc) or self.is_acronym(anaphor, antecedent, doc)

    def load_demonyms(self, tsv_file_path):
        demonyms = {}
        with codecs.open(tsv_file_path, 'r', encoding='utf-8', errors='ignore') as demonyms_tsv:
            lines = demonyms_tsv.readlines()
            for line in lines:
                place_dem_tup = tuple(line.split('\t'))
                if len(place_dem_tup) >=2: #ensure there are entries
                    demonyms[place_dem_tup[0].strip()] = place_dem_tup[1].strip()
            self.demonyms = demonyms

    def is_demonym(self, anaphor, antecedent, doc):

        if anaphor.tok_span() in self.demonyms:
            dem = self.demonyms[anaphor.tok_span()]
            if antecedent.cluster_id == -1:
                if antecedent.tok_span() == dem:
                    return True
            else:
                for m in doc.clusters[antecedent.cluster_id]:
                    if ' '.join(m.token_str_list) == dem:
                        return True
            return False
        else:
            return False

    def is_acronym(self, anaphor, antecedent, doc):
        # if all caps, make acronym of first letter of each token. Check if matches.
        pos_acronym = self.get_acronym(antecedent)
        if antecedent.cluster_id == -1:  #just one mention to check
            return pos_acronym == anaphor.tok_span()
        else:
            return True if anaphor.tok_span() in [self.get_acronym(m) for m in doc.clusters[antecedent.cluster_id]]\
                else False

    def get_acronym(self, mention):
        antecedent_tokens = mention.token_str_list
        return ''.join([tok[0] for tok in antecedent_tokens if tok[0].isupper()])

class RelaxedHead(StrictHeadMatch):
    # Head of anaphor matches any token in the set of tokens in the antecedent mentions
    def coref(self, anaphor, antecedent, doc):
        ante_head = antecedent.head
        anaph_head = anaphor.head
        if ante_head is not None and anaph_head is not None:
            return self.rel_match(anaphor, antecedent, doc) and self.word_inclusion(anaphor, antecedent, doc) \
               and ante_head.name_ent == anaph_head.name_ent and ante_head.name_ent != '*' and \
               anaph_head.name_ent != '*'
        else:
            return False

    def rel_match(self, anaphor, antecedent, doc):
        head_anaph = anaphor.head
        if head_anaph is not None:
            if antecedent.cluster_id == -1:
                words_antec = antecedent.token_str_list
            else:
                words_antec = [m.token_str_list for m in doc.clusters[antecedent.cluster_id]]
            return head_anaph.token in words_antec
        else:
            return False