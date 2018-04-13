import codecs
import os

input_dir = '../../ANC'
for dir, sub_dir, files in os.walk(input_dir):
    for f in files:
        input_f = codecs.open(os.path.join(dir, f), 'r', encoding='utf-8', errors='ignore')  # coref input file
        lines = input_f.readlines()
        new_lines = []
        for line in lines:
            line = line.encode('ascii', errors='ignore').decode()
            line = str(line)
            tokens = line.split()
            new_tokens = []
            for token in tokens:
                splittoken = token.split('_')
                if len(splittoken) < 2:
                    splittoken.append('XX')
                new_tokens.append('_'.join(splittoken))
            new_lines.append( ' '.join(new_tokens) + '\n')
        outfile = open('../../prepANC/' + f, 'w')
        for newline in new_lines:
            outfile.write(newline)
