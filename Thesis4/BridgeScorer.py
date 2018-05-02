# Chester Palen-Michel
# Bridging Scorer for Master's Thesis
# Brandeis University
# 3/14/18

import sys
import os
from collections import defaultdict


class Scorer:
    """
     python3 BridgingScorer.py <keysdir> <sysoutdir>
     Scorer for Bridging anaphora resolution. Takes two parameters: key/gold directory and sys-output dir
     Be sure that the files in both directories are equivalent except for extensions <.bridge_gold> and <.bridge_sys>
     Outputs Precision, Recall and F1-score.
     A bridge is considered 'correct' if it's anaphor is also in key file and the system output's antecedent for a given
     anaphor is in the gold coreference chain of the key file's antecedent.
     If division by 0 in any of scoring metrics, will return 0"""

    def __init__(self):
        # Bridging resolution
        self.gold_bridge_anaphors = 0
        self.total_links_predicted = 0
        self.correct_links_predicted = 0
        # Anaphora recognition
        self.sys_anaphors = 0
        self.gold_anaphors = 0
        self.correct_anaphors = 0

    def score(self, bridge_gold_dir, bridge_system_dir):
        for dir, sub_dir, files in os.walk(bridge_gold_dir):
            for f in files:
                print('file:', f)
                gold_f = open(os.path.join(dir, f), 'r')
                sys_f = open(os.path.join(dir, f.replace('.bridge_gold', '.bridge_sys')).replace(bridge_gold_dir, bridge_system_dir))
                self.count(gold_f, sys_f)
        print('Bridging Scores:')
        print('Precision: ', self.precision())
        print('Recall: ', self.recall())
        print('F-score: ', self.fscore())
        print()
        print('Anaphor Recognition Scores: ')
        print('Precision: ', self.anaph_precision())
        print('Recall: ', self.anaph_recall())
        print('F-score: ', self.anaph_fscore())

    def count(self, gold, sys):
        anaph_ante_goldtups = [(line.split('\t')[1].strip(), line.split('\t')[3].strip()) for line in gold.readlines()]
        anaph_ante_systups = [(line.split('\t')[1].strip(), line.split('\t')[3].strip()) for line in sys.readlines()]
        self.gold_bridge_anaphors += len(anaph_ante_goldtups)
        self.total_links_predicted += len(anaph_ante_systups)
        self.correct_links_predicted += self.count_correct(anaph_ante_goldtups, anaph_ante_systups)

        sys_anaphors_list = [anaph_ante[0] for anaph_ante in anaph_ante_systups]
        gold_anaphors_list = [anaph_ante[0] for anaph_ante in anaph_ante_goldtups]
        correct_anaphors_list = [anaph for anaph in sys_anaphors_list if anaph in gold_anaphors_list]
        print('sys_anaphors: ', set(sys_anaphors_list))
        print('gold_anaphors_list:', gold_anaphors_list)
        print('correct_anaphors:', set(correct_anaphors_list))
        self.sys_anaphors += len(set(sys_anaphors_list))
        self.gold_anaphors += len(gold_anaphors_list)
        self.correct_anaphors += len(set(correct_anaphors_list))

    def count_correct(self, gold_tups, sys_tups):
        # Get count of correct pairs in gold vs correct pairs in system
        ret = 0
        gold_dict = defaultdict(set)
        for gold_tup in gold_tups:
            gold_dict[gold_tup[0]] = set(gold_tup[1].split('|'))
        # If sys anaphor is in gold anaphor, then check if sys antecedent in gold antecedents for given anaphor.
        # For each match increase ret by 1.
        sys_dict = defaultdict(set)
        for sys_tup in sys_tups:
            sys_dict[sys_tup[0]].add(sys_tup[1])

        for anaphor in sys_dict:
            if sys_dict[anaphor].intersection(gold_dict[anaphor]):
                ret += 1
        return ret

    def recall(self):
        return float(self.correct_links_predicted) / self.gold_bridge_anaphors if self.gold_bridge_anaphors != 0 else 0

    def precision(self):
        return float(self.correct_links_predicted) / self.total_links_predicted if self.total_links_predicted != 0 else 0

    def fscore(self):
        denominator = (self.precision() + self.recall())
        return 2 * (self.precision() * self.recall()) / denominator if denominator != 0 else 0

    # Anaphor prediction scores
    def anaph_recall(self):
        return float(self.correct_anaphors) / self.gold_bridge_anaphors if self.gold_bridge_anaphors != 0 else 0

    def anaph_precision(self):
        return float(self.correct_anaphors) / self.sys_anaphors if self.sys_anaphors != 0 else 0

    def anaph_fscore(self):
        denominator = (self.anaph_precision() + self.anaph_recall())
        return 2 * (self.anaph_precision() * self.anaph_recall()) / denominator if denominator != 0 else 0

if __name__ == '__main__':
    # sys arguments: <key dir> <sys_out_dir>
    keys_dir = sys.argv[1]
    sysout_dir = sys.argv[2]
    scorer = Scorer()
    scorer.score(keys_dir, sysout_dir)



