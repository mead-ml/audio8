"""Wav2vec2 pretraining using 8-mile API

"""

from torchaudio.datasets import LIBRISPEECH
from torchaudio.datasets.utils import walk_files
from typing import Tuple, Dict
import logging
import json
import time
import numpy as np
import os
import torch.nn as nn
import random
import soundfile as sf
from torch.utils.data import DataLoader, IterableDataset
from eight_mile.utils import Offsets
from eight_mile.pytorch.optz import *
logger = logging.getLogger(__file__)


class LibriSpeechDataset(LIBRISPEECH):
    """The base class is non-trivial to figure out for cases when we have LS on the hard-drive


    Simplify by checking if the path already exists on the drive first
    Args:
        root (str): Path to the directory where the dataset is found or downloaded.
        url (str, optional): The URL to download the dataset from,
            or the type of the dataset to dowload.
            Allowed type values are ``"dev-clean"``, ``"dev-other"``, ``"test-clean"``,
            ``"test-other"``, ``"train-clean-100"``, ``"train-clean-360"`` and
            ``"train-other-500"``. (default: ``"train-clean-100"``)
        folder_in_archive (str, optional):
            The top-level directory of the dataset. (default: ``"LibriSpeech"``)
        download (bool, optional):
            Whether to download the dataset if it is not found at root path. (default: ``False``).
    """

    _ext_txt = ".trans.txt"
    _ext_audio = ".flac"

    def __init__(self,
                 graphemes: Dict[str, int],
                 root: str,
                 url: str = 'train-clean-100',
                 folder_in_archive: str = "LibriSpeech",
                 download: bool = False) -> None:

        self.graphemes = graphemes
        target_dir = os.path.join(root, folder_in_archive, url)
        if not os.path.exists(target_dir):
            super().__init__(root, url, folder_in_archive, download)
        self._path = target_dir
        walker = walk_files(
            self._path, suffix=self._ext_audio, prefix=False, remove_suffix=True
        )
        self._walker = list(walker)

    def __getitem__(self, n: int) -> Tuple[torch.Tensor, int, str, int, int, int]:
        """Load the n-th sample from the dataset.

        Args:
            n (int): The index of the sample to be loaded

        Returns:
            tuple: ``(waveform, sample_rate, utterance, speaker_id, chapter_id, utterance_id)``
        """
        item = super().__getitem__(n)
        unpadded = np.array([self.graphemes.get(v, Offsets.GO) for v in item[2]])
        src_length = item[0].shape[-1]
        tgt_length = len(unpadded)

        return {'src': item[0].squeeze(), 'src_length': src_length, 'tgt': unpadded, 'tgt_length': tgt_length}

def collate_fn(batch_list):
    B = len(batch_list)
    src_length = torch.tensor([f['src_length'] for f in batch_list])
    max_src_length = src_length.max().item()
    tgt_length = torch.tensor([f['tgt_length'] for f in batch_list])
    max_tgt_length = tgt_length.max().item()
    src = np.zeros((B, max_src_length), dtype=np.float32)
    tgt = np.zeros((B, max_tgt_length), dtype=np.int32)
    for i in range(B):
        l = len(batch_list[i]['src'])
        src[i, :l] = batch_list[i]['src']
        l = len(batch_list[i]['tgt'])
        tgt[i, :l] = batch_list[i]['tgt']
    return torch.from_numpy(src), src_length, torch.from_numpy(tgt), tgt_length

def find_next_fit(v, fits):
    sz = 0
    for fit in fits:
        if v < fit:
            sz = fit

    return sz


class AudioTextJSONDataset(IterableDataset):

    def __init__(self, buckets, json_file, vocab, target_tokens_per_batch, distribute=True, shuffle=True, max_dst_length=120):
        super().__init__()
        self.bucket_lengths = buckets
        self.max_dst_length = max_dst_length
        self.w2i = vocab # {}
        #with open(dict_file) as rf:
        #    for line in rf:
        #        token, index = line.split()
        #        self.w2i[token] = index
        self.json_file = json_file
        self.rank = 0
        self.world_size = 1
        self.files = []
        self.target_tokens_per_batch = target_tokens_per_batch
        self.shuffle = shuffle
        self.distribute = distribute
        if torch.distributed.is_initialized() and distribute:
            self.rank = torch.distributed.get_rank()
            self.world_size = torch.distributed.get_world_size()

        self._read_json_file(json_file)

    def _read_json_file(self, json_file):
        asc = sorted(self.bucket_lengths)
        self.files = {b: [] for b in asc}
        skipped = 0
        with open(json_file, "r") as f:
            self.directory = f.readline().strip()
            for line in f:
                j = json.loads(line)
                path = j['audio']
                x_length = j['x_length']
                count = find_next_fit(x_length, self.bucket_lengths)
                y_length = j['y_length']
                tokens = np.array([self.w2i[t] for t in j['tokens']])
                if count == 0:
                    print(x_length)
                    continue
                self.files[count].append((path, x_length, y_length, tokens))


    def _get_worker_info(self):
        return torch.utils.data.get_worker_info() if self.distribute else None

    def _init_read_order(self):
        # Each node has the same worker_info, so the unique offsets for each is
        # rank * num_workers + worker_id
        # and the total available workers is world_size * num_workers
        worker_info = self._get_worker_info()

        if worker_info is None:
            num_workers_per_node = 1
            node_worker_id = 0
        else:
            num_workers_per_node = worker_info.num_workers
            node_worker_id = worker_info.id
        all_workers = (self.world_size * num_workers_per_node)
        offset = self.rank * num_workers_per_node + node_worker_id
        read_file_order = list(range(offset, len(self.files), all_workers))
        if not read_file_order:
            if offset > 0:
                # This is probably wrong
                logger.warning(f"There are no files to read for worker {node_worker_id}, offset {offset}!" +
                               " This might mean that you are passing an incorrect training or validation directory")
            else:
                # This is definitely wrong
                raise Exception(f"No files of pattern {self.pattern} were found in {self.directory}!")
        return read_file_order, node_worker_id

    def __iter__(self):
        read_file_order, _ = self._init_read_order()
        keys = list(self.files.keys())
        # If we have multiple files per worker, possibly shuffle the file read order
        while True:
            if self.shuffle:
                random.shuffle(read_file_order)
            for bucket_idx in read_file_order:
                bucket = keys[bucket_idx]
                num_samples = self.target_tokens_per_batch // bucket
                audio_samples = []
                audio_lengths = []
                text_samples = []
                text_lengths = []
                for (file, x_length, y_length, tokens) in self.files[bucket]:

                    #text = np.array([self.w2i[t] for t in tokens])
                    zp_text = np.zeros(self.max_dst_length, dtype=np.int32)
                    zp_text[:len(tokens)] = tokens
                    text_lengths.append(len(tokens))
                    zp_audio = np.zeros(bucket, dtype=np.float32)
                    audio = self.process_sample(file)
                    zp_audio[:len(audio)] = audio
                    audio_lengths.append(len(audio))
                    audio_samples.append(zp_audio)
                    text_samples.append(zp_text)
                    if len(audio_samples) == num_samples:
                        pair = np.stack(audio_samples), np.stack(audio_lengths), np.stack(text_samples), np.stack(text_lengths)
                        audio_samples = []
                        audio_lengths = []
                        text_samples = []
                        text_lengths = []
                        yield pair
                if audio_samples:
                    pair = np.stack(audio_samples), np.stack(audio_lengths), np.stack(text_samples), np.stack(text_lengths)
                    yield pair

    def process_sample(self, file):
        """Read in a line and turn it into an entry.  FIXME, get from anywhere

        The entries will get collated by the data loader

        :param file:
        :return:
        """
        wav, _ = sf.read(file)
        wav = wav.astype(np.float32)
        return wav


class AudioFileDataset(IterableDataset):

    def __init__(self, manifest, max_length, target_tokens_per_batch, distribute=True, shuffle=True,  min_length=0):
        super().__init__()
        self.max_length = max_length
        self.manifest = manifest
        self.rank = 0
        self.world_size = 1
        self.files = []
        self.target_tokens_per_batch = target_tokens_per_batch
        self.shuffle = shuffle
        self.distribute = distribute
        if torch.distributed.is_initialized() and distribute:
            self.rank = torch.distributed.get_rank()
            self.world_size = torch.distributed.get_world_size()

        self._read_manifest(manifest, min_length)

    def _read_manifest(self, manifest, min_length):
        skipped = 0
        with open(manifest, "r") as f:
            self.directory = f.readline().strip()
            for line in f:
                items = line.strip().split("\t")
                sz = int(items[1])
                if min_length is not None and sz < min_length:
                    skipped += 1
                    continue
                self.files.append((os.path.join(self.directory, items[0]), sz,))

        sorted(self.files, key=lambda item: item[-1])
        logger.info(f"loaded {len(self.files)}, skipped {skipped} samples")

    def _get_worker_info(self):
        return torch.utils.data.get_worker_info() if self.distribute else None

    def _init_read_order(self):
        # Each node has the same worker_info, so the unique offsets for each is
        # rank * num_workers + worker_id
        # and the total available workers is world_size * num_workers
        worker_info = self._get_worker_info()

        if worker_info is None:
            num_workers_per_node = 1
            node_worker_id = 0
        else:
            num_workers_per_node = worker_info.num_workers
            node_worker_id = worker_info.id
        all_workers = (self.world_size * num_workers_per_node)
        offset = self.rank * num_workers_per_node + node_worker_id
        read_file_order = list(range(offset, len(self.files), all_workers))
        if not read_file_order:
            if offset > 0:
                # This is probably wrong
                logger.warning(f"There are no files to read for worker {node_worker_id}, offset {offset}!" +
                               " This might mean that you are passing an incorrect training or validation directory")
            else:
                # This is definitely wrong
                raise Exception(f"No files of pattern {self.pattern} were found in {self.directory}!")
        return read_file_order, node_worker_id

    def next_sample(self):
        read_file_order, _ = self._init_read_order()
        # If we have multiple files per worker, possibly shuffle the file read order
        while True:
            if self.shuffle:
                random.shuffle(read_file_order)
            for file_idx in read_file_order:
                file, _ = self.files[file_idx]
                yield self.process_sample(file, self.max_length)


    def process_sample(self, file, len):
        """Read in a line and turn it into an entry.  FIXME, get from anywhere

        The entries will get collated by the data loader

        :param file:
        :return:
        """
        wav, _ = sf.read(file)
        wav = wav.astype(np.float32)
        return wav[:len]

    def __iter__(self):

        min_length = self.max_length

        num_tokens_predicted = 0

        samples = []
        for sample in self.next_sample():

            if num_tokens_predicted < self.target_tokens_per_batch:
                min_length = min(min_length, len(sample))
                samples.append(sample)
                num_tokens_predicted = len(samples) * min_length
            else:
                batch = np.stack([s[:min_length] for s in samples])
                samples = []
                num_tokens_predicted = 0
                #logger.debug("(%d, %d) %d", batch.shape[0], batch.shape[1], np.product(batch.shape))
                yield batch


def find_fit(v, fits):
    truncate_to = 0
    for fit in fits:
        if v//fit:
            truncate_to = fit
        else:
            break
    return truncate_to


class BucketingAudioDataset(AudioFileDataset):

    def __init__(self, buckets, manifest, max_length, target_tokens_per_batch, distribute=True, shuffle=True,  min_length=0):
        self.bucket_lengths = buckets
        super().__init__(manifest, max_length, target_tokens_per_batch, distribute, shuffle, min_length)

    def _read_manifest(self, manifest, _):
        skipped = 0
        asc = sorted(self.bucket_lengths)
        self.files = {b: [] for b in asc}

        num_samples = 0
        with open(manifest, "r") as f:

            directory = f.readline().strip()
            for line in f:
                num_samples += 1
                items = line.strip().split("\t")
                sz = int(items[1])
                fname = os.path.join(directory, items[0])

                if sz < asc[0]:
                    skipped += 1
                    continue
                count = find_fit(sz, self.bucket_lengths)
                self.files[count].append((fname, sz))

        logger.info('Num samples %d, skipped %d', num_samples, skipped)

    def next_sample(self):
        read_file_order, _ = self._init_read_order()
        keys = list(self.files.keys())

        # If we have multiple files per worker, possibly shuffle the file read order
        while True:
            if self.shuffle:
                random.shuffle(read_file_order)
            for bucket_idx in read_file_order:
                bucket = keys[bucket_idx]
                for (file, _) in self.files[bucket]:
                    yield self.process_sample(file, bucket)


