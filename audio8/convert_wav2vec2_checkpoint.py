from audio8.wav2vec2 import Wav2Vec2Model, Wav2Vec2AcousticModel, load_fairseq_bin
import argparse
from eight_mile.utils import str2bool
import os
from audio8.text import read_vocab_file
import torch

parser = argparse.ArgumentParser(description='Convert a wav2vec2 checkpoint to 8-mile')
parser.add_argument('--model', help='A file path pointing to a wav2vec2 model checkpoint, either pretrained or fine-tuned')
parser.add_argument('--ctc', help="Is the checkpoint a CTC checkpoint", type=str2bool, default=False)
parser.add_argument('--target_dir', help='This is the target directory where we will put the checkpoints')
parser.add_argument('--vocab_file', help='If this is a CTC checkpoint, we need a vocab')
parser.add_argument('--num_heads', default=12, type=int)
parser.add_argument('--num_layers', default=12, type=int)
parser.add_argument('--d_model', default=768, type=int)
parser.add_argument("--num_vq_vars", type=int, default=320)
parser.add_argument("--num_vq_groups", type=int, default=2)
parser.add_argument("--final_dim", type=int, default=256)
parser.add_argument('--d_ff', type=int)
args = parser.parse_args()

output_file = args.model.replace('.pt', '-a8.pth')
if not args.target_dir:
    args.target_dir = os.path.dirname(args.model)
output_file = os.path.join(args.target_dir, output_file)
print(f"Write checkpoint to {output_file}")

if args.ctc:
    vocab = read_vocab_file(args.vocab_file)
    model = Wav2Vec2AcousticModel(num_labels=len(vocab), d_model=args.d_model, num_heads=args.num_heads, num_layers=args.num_layers, d_ff=args.d_ff)
    unmapped = load_fairseq_bin(model, args.model, ctc=True)


else:
    model = Wav2Vec2Model(num_vq_vars=args.num_vq_vars, num_vq_groups=args.num_vq_groups,
                          num_layers=args.num_layers, num_heads=args.num_heads, d_ff=args.d_ff, d_model=args.d_model,
                          final_dim=args.final_dim)
    unmapped = load_fairseq_bin(model, args.model)

if unmapped['missing'] or unmapped['unexpected']:
    raise Exception(unmapped)

torch.save(model.state_dict(), output_file)
