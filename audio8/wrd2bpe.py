import logging
import os
from argparse import ArgumentParser
from audio8.text import BPEVectorizer
from eight_mile.utils import revlut
parser = ArgumentParser()
parser.add_argument("--root_dir")
parser.add_argument("--train_dataset", type=str, help='Dataset (by name), e.g. train-clean-360')
parser.add_argument("--valid_dataset", type=str, help='Dataset (by name), e.g. dev-other')
parser.add_argument("--subword_model_file", type=str, help="The BPE model file", required=True)
parser.add_argument("--subword_vocab_file", type=str, help="The BPE subword vocab", required=True)
parser.add_argument("--emit_begin_tok", type=str, default=[])
parser.add_argument("--emit_end_tok", type=str, default=[])
parser.add_argument("--lower", action='store_true')
args = parser.parse_args()


vec = BPEVectorizer(args.subword_model_file, args.subword_vocab_file, args.emit_begin_tok, args.emit_end_tok)
i2w = revlut(vec.vocab)
num_vocab = max(i2w.keys())
with open(os.path.join(args.root_dir, 'dict.bpe.txt'), 'w') as wf:
    for i in range(num_vocab):
        wf.write(i2w[i] + '\n')

train_file = os.path.join(args.root_dir, args.train_dataset)
valid_file = os.path.join(args.root_dir, args.valid_dataset)

files = [train_file, valid_file]
input_files = [f.replace('.tsv', '.wrd') for f in files]
output_files = [f.replace('.wrd', '.bpe') for f in input_files]


for inf, outf in zip(input_files, output_files):
    print(outf)
    with open(inf) as rf, open(outf, 'w') as wf:
        for line in rf:
            line = line.strip()
            if args.lower:
                line = line.lower()
            tok = line.split()
            outline = ' '.join([i2w[x] for x in vec.run(tok)])
            wf.write(outline + '\n')
