# Chester Palen-Michel
# Brandeis University
# 2/14/18

""" Main coreference resolver class. applies each rule to each document."""

import sys
import codecs
import os
from CoreReader import CoreReader
from Sieve import ExactMatchSieve, PronounMatch, StrictHeadMatch, VariantHeadMatch, PreciseConstructs, RelaxedHead
from XML_ISNotesReader import ISFile


class Resolver:
    def __init__(self):
        pass

    def resolve(self, doc):
        # apply each sieve to the document here
        # each sieve should take document and c and return cluster (parent class for sieve )

        # (1) Exact String Match
        exact_match = ExactMatchSieve()
        exact_match.cluster_path(doc)

        # (2) Precise Constructs: Demonyms and Acronyms
        precise_constr = PreciseConstructs()
        precise_constr.load_demonyms('demonyms.tsv')
        precise_constr.cluster_path(doc)

        # (3) Strict Head Match
        strict_head = StrictHeadMatch()
        strict_head.cluster_path(doc)

        # (4) Variant on Strict Head Match
        varhead = VariantHeadMatch()
        varhead.cluster_path(doc)

        # (5) Relaxed Head Matching
        rel_head = RelaxedHead()
        rel_head.cluster_path(doc)

        # (6) Pronoun Match (coming soon to a theatre near you)
        pronoun_match = PronounMatch()
        pronoun_match.cluster_path(doc)

    def run_resolver(self, in_file, out_file, is_file, bridge_gold, run_all_files=True):

        with codecs.open(in_file, 'r', encoding='utf-8', errors='ignore') as in_file:
            # All files
            if run_all_files:
                    coref_reader = CoreReader()
                    doc_list = coref_reader.read_file(in_file)
                    for doc in doc_list:
                        doc.markable2mention(is_file)
                        self.resolve(doc)
                        doc.clusters_to_tokens()
                        out_file.writelines(doc.write_lines())

            # Single File
            else:
                coref_reader = CoreReader()
                documents = coref_reader.read_file(in_file)
                for doc in documents:
                    doc.markable2mention(is_file)
                    self.resolve(doc)
                    doc.clusters_to_tokens()
                    out_file.writelines(doc.write_lines())
                    bridge_gold.write(doc.write_bridges())

if __name__ == '__main__':

    resolver = Resolver()
    run_all = False  # change to False to run single file

    # For dumping all in single files for scorer
    if run_all:
        if len(sys.argv) < 5:
            print("Please provide all parameters. <input path> <output path> <gold input path> <gold output path> <file ending>")
        output_path = sys.argv[2]
        input_path = sys.argv[1]
        goldin_path = sys.argv[3]
        goldout_path = sys.argv[4]
        file_extension = sys.argv[5]

        output_f = codecs.open(output_path, 'w', encoding='utf-8', errors='ignore')
        goldout_f = codecs.open(goldout_path, 'w', encoding='utf-8', errors='ignore')

        for dir, sub_dir, files in os.walk(input_path):
            print('Resolving files in ', dir)
            for f in files:
                print('Resolving file: ', f)
                if f.endswith(file_extension):
                    is_file = 'ISAnnotationWithoutTrace/' + f.split('.')[0] + '_entity_level.xml'
                    resolver.run_resolver(os.path.join(dir, f), output_f, ISFile(is_file))

        for dir, subdir, files in os.walk(goldin_path):
            print('Compiling a single gold file from ', dir)
            for f in files:
                if f.endswith(file_extension):
                    file = codecs.open(os.path.join(dir, f), 'r', encoding='utf-8', errors='ignore')
                    goldout_f.write(file.read())
    # For full directory structure of output preserved

    # output_path = sys.argv[2]
    # if not os.path.isdir(sys.argv[2]):
    #     os.mkdir(sys.argv[2])
    # for dir, sub_dir, files in os.walk(sys.argv[1]):
    #     logging.info('Resolving files in %', dir)
    #     structure = os.path.join(output_path, dir[len(sys.argv[1])+1:])
    #     if not os.path.isdir(structure):
    #         os.mkdir(structure)
    #     for f in files:
    #         resolver.run_resolver(os.path.join(dir,f),os.path.join(structure, f))
    else:
        # # For testing a single file
        # input_f = 'Coref_conll_IS_files_only/dev/wsj_1017.v4_auto_conll'
        # is_input = ISFile('ISAnnotationWithoutTrace/dev/wsj_1017_entity_level.xml')
        # output_f = codecs.open('test', 'w', encoding='utf-8', errors='ignore')
        # bridge_gold = open('bridged_gold/', 'w')
        # resolver.run_resolver(input_f, output_f, is_input, bridge_gold, False)

        # To Run resolver just to get gold bridging files
        output_f = codecs.open('test', 'w', encoding='utf-8', errors='ignore')
        input_dir = 'Coref_conll_IS_files_only/test'
        for dir, sub_dir, files in os.walk(input_dir):
            for f in files:
                # do things to the files.
                input_f = input_dir + '/' + f  # coref input file

                is_input = ISFile('ISAnnotationWithoutTrace/test/' + f.replace('.v4_auto_conll', '_entity_level.xml'))# isfile needed to get markables, make from coref filename
                bridge_gold = open('bridging_gold/test/' + f.replace('.v4_auto_conll', '.bridge_gold'), 'w')# write to this, make from coref file name or is_input. .bridge_gold
                resolver.run_resolver(input_f, output_f, is_input, bridge_gold, False)