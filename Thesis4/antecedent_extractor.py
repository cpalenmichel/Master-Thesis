# Chester Palen-Michel
# 4/28/18
from collections import namedtuple

from feature_extractor import Extractor


WordPair = namedtuple('WordPair', ['anaphor', 'antecedent'])


class AntecedentExtractor(Extractor):
    def __init__(self):
        super(AntecedentExtractor, self).__init__()
        self.sent_window = 2

    def get_features(self):
        for ment, doc in self.ment_doc_instances():
            ment_group = self.mentions_in_window(ment, self.sent_window, doc)
            instances = self.make_instances(ment, ment_group)
            for instance in instances:
                self.make_features(instance, ment_group, doc)
            # TODO each mention: generate all instances of men-men pairs
            # TODO generate group of potential antecedents for mention, to pass just list of mentions.
            pass


        # Do mention_doc extraction as before. but for each mention, take all mentions 2 sents prior as a group
        # make key a ment-ment pair
        # when calling the make_features function should be passing in ment-ment pair and all mentions in mention group
            #  of 2 sents window.
        # also cool if could make this an n sent window. :)


    def mentions_in_window(self, ment, n_sents, doc):
        ret = []
        for mention in doc.sentences[ment.sentence_index].mentions:
            if mention.span_end < ment.span_start:
                ret.append(mention)
        i = ment.sentence_index - 1
        while i >= 0 and i >= ment.sentence_index - n_sents:
            ret.extend(doc.sentences[i].mentions)
            i -= 1
        return ret

    def make_instances(self, ment, ment_group):
        ret = []
        for mg in ment_group:
            ret.append(WordPair(ment, mg))
        return ret

    def make_features(self, instance, ment_group):
        # TODO
        pass


if __name__ == '__main__':
    extractor = AntecedentExtractor()
    extractor.load_corpus()
    extractor.get_features()
    extractor.write_json('output.json')
