# Chester Palen-Michel
# 4/28/18
import json
from collections import defaultdict

from sklearn import svm
from sklearn.feature_extraction import DictVectorizer


class Recognizer:
    """
    Anaphora recognizer. Use in Resolution to determine likely anaphor candidates.
    Also test for its own sake how well performs.
    """
    def __init__(self):
        self.features_to_exclude = 'determiner,gold_chain_len'
        self.anaphoricity_cl = svm.SVC(class_weight={1:12, 0:1}, kernel='rbf')
        self.feature_lookup = defaultdict(dict)

    def train_classifier(self, docs):
        self.w = DictVectorizer(sparse=True)
        D, Y = self.anaphor_features(docs)
        X = self.w.fit_transform(D)
        self.anaphoricity_cl.fit(X, Y)

    def anaphor_features(self, docs):
        features = []
        labels = []
        for doc in docs:
            for sent in doc.sentences:
                for ment in sent.mentions:
                    labels.append(self.anaphoric_labels(ment))
                    feature_vec = self.feature_lookup[self.keyify(ment, doc)]
                    features.append(feature_vec)
        return (features, labels)

    def keyify(self, ment, doc):
        return ment.underscore_span() + ';' + doc.docID + ';' + str(ment.sentence_index)

    def anaphoric_labels(self, m):
        return 1 if m.bridged_from is not None else 0

    def recognize(self, mention, doc):
        #returns 1 if is anaphoric and 0 if not
        return self.anaphoricity_cl.predict(self.w.transform(self.feature_lookup[self.keyify(mention, doc)]))

    def load_feature_dict(self, feature_filepath):
        if self.features_to_exclude == "None":
            self.feature_lookup = json.load(open(feature_filepath, 'r'))
        else:
            loaded_json = json.load(open(feature_filepath, 'r'))
            for key in loaded_json:
                for feat_name in loaded_json[key]:
                    if feat_name not in self.features_to_exclude.split(','):
                        self.feature_lookup[key][feat_name] = loaded_json[key][feat_name]