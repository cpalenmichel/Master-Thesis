# Chester Palen-Michel
# Bridging Scorer for Master's Thesis
# Brandeis University
# 3/14/18

import sys
import os
from collections import defaultdict


class Scorer:
    """Scorer for Bridging anaphora resolution. Takes two parameters: key/gold directory and sys-output dir
     Be sure that the files in both directories are equivalent except for extensions <.bridge_gold> and <.bridge_sys>
     Outputs Precision, Recall and F1-score.
     A bridge is considered 'correct' if it's anaphor is also in key file and the system output's antecedent for a given
     anaphor is in the gold coreference chain of the key file's antecedent.
     If division by 0 in any of scoring metrics, will return 0"""

    def __init__(self):
        self.gold_bridge_anaphors = 0
        self.total_links_predicted = 0
        self.correct_links_predicted = 0

    def score(self, bridge_gold_dir, bridge_system_dir):
        # TODO make sure bridge_system_dir is file or dir to match bridge_gold?

        for dir, sub_dir, files in os.walk(bridge_gold_dir):
            for f in files:
                gold_f = open(os.path.join(bridge_gold_dir, f), 'r')
                sys_f = open(os.path.join(bridge_system_dir, f.replace('.bridge_gold', '.bridge_sys')))
                self.count(gold_f, sys_f)

        print('Precision: ', self.precision())
        print('Recall: ', self.recall())
        print('F-score: ', self.fscore())

    def count(self, gold, sys):
        anaph_ante_goldtups = [(line.split('\t')[1].strip(), line.split('\t')[3].strip()) for line in gold.readlines()]
        anaph_ante_systups = [(line.split('\t')[1].strip(), line.split('\t')[3].strip()) for line in sys.readlines()]
        self.gold_bridge_anaphors += len(anaph_ante_goldtups)
        self.total_links_predicted += len(anaph_ante_systups)
        self.correct_links_predicted += self.count_correct(anaph_ante_goldtups, anaph_ante_systups)

    def count_correct(self, gold_tups, sys_tups):
        # Get count of correct pairs in gold vs correct pairs in system
        ret = 0
        gold_dict = defaultdict(set)
        for gold_tup in gold_tups:
            gold_dict[gold_tup[0]] = set(gold_tup[1].split('|'))
        # If sys anaphor is in gold anaphor, then check if sys antecedent in gold antecedents for given anaphor.
        # For each match increase ret by 1.
        for sys_tup in sys_tups:
            sys_anaphor = sys_tup[0]
            sys_antecedent = sys_tup[1]
            if sys_antecedent in gold_dict[sys_anaphor]:
                ret += 1
        return ret

    def recall(self):
        return float(self.correct_links_predicted)/ self.gold_bridge_anaphors if self.gold_bridge_anaphors != 0 else 0

    def precision(self):
        return float(self.correct_links_predicted) / self.total_links_predicted if self.total_links_predicted != 0 else 0

    def fscore(self):
        denominator = (self.precision() + self.recall())
        return 2 * (self.precision() * self.recall()) / denominator if denominator != 0 else 0


if __name__ == '__main__':
    # sys arguments: <key dir> <sys_out_dir>
    keys_dir = sys.argv[1]
    sysout_dir = sys.argv[2]
    scorer = Scorer()
    scorer.score(keys_dir, sysout_dir)



