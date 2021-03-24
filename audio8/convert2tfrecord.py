"""Pretraining using 8-mile API

"""
import logging
import time
import math
import numpy as np
import os
import tfrecord
from pathlib import Path
from argparse import ArgumentParser
from audio8.data import AudioTextLetterDataset
from audio8.text import TextVectorizer, read_vocab_file
from multiprocessing import Queue, Process


logger = logging.getLogger(__file__)


class Worker(Process):
    def __init__(self, q, reader, prefix):
        super().__init__()
        self.q = q
        self.reader = reader
        self.prefix = prefix

    def run(self):
        for t in iter(self.q.get, None):
            if t is None:
                break
            idx, batch_list = t

            filename = f"{self.prefix}-{idx}.tfrecord"
            writer = tfrecord.TFRecordWriter(filename)
            for i, batch in enumerate(batch_list):

                batch = self.reader.read_batch(batch)
                batch['signal_lengths'] = (batch['signal_lengths'], 'int',)
                signal = batch['signal']
                batch['batch_size'] = (signal.shape[0], 'int',)
                batch['signal'] = (signal.reshape(-1), 'float',)
                batch['token_ids'] = (batch['token_ids'].reshape(-1), 'int',)
                batch['token_lengths'] = (batch['token_lengths'], 'int',)
                del batch['files']
                writer.write(batch)
                if i > 100:
                    break
            writer.close()
            idx_file = f"{self.prefix}-{idx}.index"
            tfrecord.tools.tfrecord2idx.create_index(filename, idx_file)


def convert_dataset(ds, num_batches, num_batches_per_shard, num_workers, prefix):
    start_time = time.time()

    q = Queue()
    j = 0
    for i in range(0, num_batches, num_batches_per_shard):
        q.put((j, ds.batches[i:i + num_batches_per_shard],))
        j += 1
    for i in range(num_workers):
        Worker(q, ds, prefix).start()
    for i in range(num_workers):
        q.put(None)
    elapsed = time.time() - start_time
    return elapsed


def convert():
    parser = ArgumentParser()
    parser.add_argument("--basedir", type=str)
    parser.add_argument("--root_dir")
    parser.add_argument("--max_sample_len", type=int, help="Max sample length")

    parser.add_argument("--train_dataset", type=str, help='Dataset (by name), e.g. train-clean-360')
    parser.add_argument("--valid_dataset", type=str, help='Dataset (by name), e.g. dev-other')
    parser.add_argument("--input_sample_rate", type=int, default=16_000)
    parser.add_argument("--target_sample_rate", type=int, default=16_000)
    parser.add_argument("--dict_file", type=str, help="Dictionary file", default='dict.{}.txt')
    parser.add_argument("--dataset_key", default="LibriSpeech", help="dataset key for basedir")
    parser.add_argument("--vocab_file", help="Vocab for output decoding")
    parser.add_argument("--target_tokens_per_batch", type=int, default=800_000)
    parser.add_argument("--target_type", type=str, choices=["wrd", "ltr", "bpe"], default="ltr")
    parser.add_argument("--num_train_shards", type=int, default=200, help="Number of train shards to generate")
    parser.add_argument("--num_valid_shards", type=int, default=10, help="Number of valid shards to generate")
    parser.add_argument("--num_workers", type=int, default=10, help="Number of worker processes")

    args = parser.parse_args()
    args.dict_file = args.dict_file.format(args.target_type)

    # Get the basedir to save results and checkpoints
    if args.basedir is None:
        args.basedir = f'{args.dataset_key}-{args.target_type}-{args.target_sample_rate}'
    os.makedirs(args.basedir, exist_ok=True)

    # Setup logger
    logging.basicConfig(level=logging.INFO)


    vocab_file = args.vocab_file if args.vocab_file else os.path.join(args.root_dir, args.dict_file)
    vocab = read_vocab_file(vocab_file)
    vec = TextVectorizer(vocab)

    train_dataset = os.path.join(args.root_dir, args.train_dataset)
    valid_dataset = os.path.join(args.root_dir, args.valid_dataset)

    train = AudioTextLetterDataset(
        train_dataset,
        vec,
        args.target_tokens_per_batch,
        args.max_sample_len,
        input_sample_rate=args.input_sample_rate,
        target_sample_rate=args.target_sample_rate,
        shuffle=False,
        is_infinite=False,
        tgt_type=args.target_type,
    )
    valid = AudioTextLetterDataset(
        valid_dataset,
        vec,
        args.target_tokens_per_batch,
        args.max_sample_len,
        input_sample_rate=args.input_sample_rate,
        target_sample_rate=args.target_sample_rate,
        distribute=False,
        shuffle=False,
        is_infinite=False,
        tgt_type=args.target_type,
    )
    num_batches_train = len(train.batches)
    num_batches_per_shard_train = math.ceil(num_batches_train / args.num_train_shards)


    num_batches_valid = len(valid.batches)
    num_batches_per_shard_valid = math.ceil(num_batches_valid / args.num_valid_shards)

    logger.info(f"Writing out {num_batches_per_shard_train} train records per shard")
    logger.info(f"Writing {num_batches_train} train records total")

    logger.info(f"Writing out {num_batches_per_shard_valid} valid records per shard")
    logger.info(f"Writing {num_batches_valid} valid records total")

    train_dir = os.path.join(args.basedir, Path(args.train_dataset).stem)
    os.makedirs(train_dir, exist_ok=True)
    valid_dir = os.path.join(args.basedir, Path(args.valid_dataset).stem)
    os.makedirs(valid_dir, exist_ok=True)

    convert_dataset(train, num_batches_train, num_batches_per_shard_train, args.num_workers, os.path.join(train_dir, 'train'))

    convert_dataset(valid, num_batches_valid, num_batches_per_shard_valid, args.num_workers, os.path.join(valid_dir, 'valid'))


if __name__ == "__main__":
    convert()
