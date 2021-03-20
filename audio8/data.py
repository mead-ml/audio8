"""Wav2vec2 pretraining using 8-mile API

"""


import numpy as np
import os
import torch.nn as nn
import random
import soundfile as sf
from torch.utils.data import DataLoader, IterableDataset
from eight_mile.utils import Offsets
from eight_mile.pytorch.optz import *

try:
    import scipy.signal
except:
    pass
logger = logging.getLogger(__file__)


class SoundfileAudioReader:
    def transform(self, audio):
        return audio.astype(np.float32)

    def read(self, file, max_length=-1):
        wav, _ = sf.read(file)
        wav = self.transform(wav)

        if max_length > 0:
            return wav[:max_length]

        return wav


class AudioResampleReader(SoundfileAudioReader):
    def __init__(self, sample_factor: float):
        """
        Resampl
        :param sample_factor:
        """
        self.sample_factor = sample_factor

    def transform(self, wav):
        """Do an FFT-based resample of the wav

        :param wav:
        :return:
        """
        num = int(len(wav) * self.sample_factor)
        resample = scipy.signal.resample(wav, num)
        return resample.astype(dtype=np.float32)


def pad_init(shp, dtype=np.int32):
    return np.zeros(shp, dtype=dtype)


def find_next_fit(v, fits):
    sz = 0
    for fit in fits:
        if v < fit:
            sz = fit

    return sz


def _is_batch_full(num_sentences, num_tokens, max_tokens, max_sentences):
    if num_sentences == 0:
        return False
    if max_sentences > 0 and num_sentences == max_sentences:
        return True
    if max_tokens > 0 and num_tokens > max_tokens:
        return True
    return False


def batch_by_size(
    indices, sizes, max_tokens=None, max_sentences=128,
):
    sample_len = 0
    sample_lens = []
    batch = []
    batches = []
    indices_view = indices

    for i in range(len(indices_view)):
        idx = indices_view[i]
        num_tokens = sizes[idx]
        sample_lens.append(num_tokens)
        sample_len = max(sample_len, num_tokens)

        assert (
            max_tokens <= 0 or sample_len <= max_tokens
        ), "sentence at index {} of size {} exceeds max_tokens " "limit of {}!".format(idx, sample_len, max_tokens)
        num_tokens = (len(batch) + 1) * sample_len

        if _is_batch_full(len(batch), num_tokens, max_tokens, max_sentences):
            batch_len = len(batch)
            batches.append(batch[:batch_len])
            batch = batch[batch_len:]
            sample_lens = sample_lens[batch_len:]
            sample_len = max(sample_lens) if len(sample_lens) > 0 else 0
        batch.append(idx)
    if len(batch) > 0:
        batches.append(batch)
    return batches


class AudioTextLetterDataset(IterableDataset):

    TGT_LETTER = 'ltr'
    TGT_BPE = 'bpe'
    TGT_WRD = 'wrd'

    def __init__(
        self,
        tsv_file,
        vec,
        target_tokens_per_batch,
        max_src_length=None,
        distribute=True,
        shuffle=True,
        max_dst_length=1200,
        tgt_type=TGT_LETTER,
        input_sample_rate=16_000,
        target_sample_rate=16_000,
        is_infinite=True,
    ):
        super().__init__()
        self.sample_factor = target_sample_rate / input_sample_rate
        self.reader = (
            AudioResampleReader(target_sample_rate / input_sample_rate)
            if input_sample_rate != target_sample_rate
            else SoundfileAudioReader()
        )
        self.min_src_length = 0  # TODO: remove?
        self.max_src_length = max_src_length
        self.max_dst_length = max_dst_length
        self.tgt_type = tgt_type
        self.vec = vec
        self.tsv_file = tsv_file
        self.rank = 0
        self.world_size = 1
        self.files = []
        self.max_elems_per_batch = target_tokens_per_batch
        self.shuffle = shuffle
        self.distribute = distribute
        if torch.distributed.is_initialized() and distribute:
            self.rank = torch.distributed.get_rank()
            self.world_size = torch.distributed.get_world_size()

        self._read_tsv_file(tsv_file)
        self.is_infinite = is_infinite

    def _read_transcription(self, transcription):
        return transcription.split()

    def get_or_unk_warn(self, t):
        if t not in self.vec.vocab:
            logger.debug('Word %s not in vocab', t)
            return Offsets.UNK
        return self.vec.vocab[t]

    def _read_tsv_file(self, tsv_file):
        self.files = []
        self.sizes = []
        self.tokens = []
        with open(tsv_file, "r") as f:
            self.directory = f.readline().strip()
            transcription_file = tsv_file.replace('tsv', self.tgt_type)
            with open(transcription_file) as rf:
                for i, (audio, transcription) in enumerate(zip(f, rf)):
                    basename, x_length = audio.split('\t')
                    path = os.path.join(self.directory, basename)
                    x_length = int(int(x_length) * self.sample_factor)

                    if x_length < self.min_src_length or (self.max_src_length and x_length > self.max_src_length):
                        continue
                    text = self._read_transcription(transcription)
                    if self.tgt_type != AudioTextLetterDataset.TGT_BPE:
                        tokens = self.vec.run(text)
                    # If the data is already BPE, we dont want to re-tokenize it, we just have to convert it to ints
                    # the assumption here is that if its BPE, the start and token are not part of the chunks
                    else:
                        go = (
                            [self.vec.vocab[t] for t in self.vec.internal.emit_begin_tok]
                            if self.vec.emit_begin_tok
                            else []
                        )
                        end = (
                            [self.vec.vocab[t] for t in self.vec.internal.emit_end_tok] if self.vec.emit_end_tok else []
                        )

                        tokens = go + [self.get_or_unk_warn(t) for t in text] + end
                        tokens = np.array(tokens, dtype=np.int)
                    self.files.append(path)
                    self.sizes.append(x_length)
                    self.tokens.append(tokens)
        # The idea in this code is to sort it by length, if you do that the index changes, which is why we
        # store the original index in the tuple at index 0
        keys = np.arange(len(self.files)) if not self.shuffle else np.random.permutation(len(self.files))
        indices = np.lexsort((keys, self.sizes))[::-1]

        self.batches = batch_by_size(indices, self.sizes, self.max_elems_per_batch, max_sentences=128,)

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
        all_workers = self.world_size * num_workers_per_node
        offset = self.rank * num_workers_per_node + node_worker_id
        read_file_order = list(range(offset, len(self.batches), all_workers))
        if not read_file_order:
            if offset > 0:
                # This is probably wrong
                logger.warning(
                    f"There are no files to read for worker {node_worker_id}, offset {offset}!"
                    + " This might mean that you are passing an incorrect training or validation directory"
                )
            else:
                # This is definitely wrong
                raise Exception(f"No files of pattern {self.pattern} were found in {self.directory}!")
        return read_file_order, node_worker_id

    def __iter__(self):

        read_order, _ = self._init_read_order()
        # If we have multiple files per worker, possibly shuffle the file read order
        if self.is_infinite:
            while True:
                if self.shuffle:
                    random.shuffle(read_order)
                for rd in read_order:
                    batch = self.batches[rd]
                    batch = self.read_batch(batch)
                    yield batch['signal'], batch['signal_lengths'], batch['token_ids'], batch['token_lengths'], batch[
                        'files'
                    ]
        else:
            if self.shuffle:
                random.shuffle(read_order)
            for rd in read_order:
                batch = self.batches[rd]
                batch = self.read_batch(batch)
                yield batch['signal'], batch['signal_lengths'], batch['token_ids'], batch['token_lengths'], batch[
                    'files'
                ]

    def read_batch(self, batch):
        zp_text = np.ones((len(batch), self.max_dst_length), dtype=np.long)
        audios = []
        audio_lengths = np.zeros(len(batch), dtype=np.int32)
        text_lengths = np.zeros(len(batch), dtype=np.long)
        files = []
        for i, idx in enumerate(batch):
            pth = self.files[idx]
            files.append(pth)
            tokens = self.tokens[idx]
            if len(tokens) > self.max_dst_length:
                raise Exception(f"Tokens too long {len(tokens)}")
            len_text = min(self.max_dst_length, len(tokens))
            zp_text[i, :len_text] = tokens[:len_text]
            audio = self.process_sample(pth)
            if self.max_src_length and len(audio) > self.max_src_length:
                raise Exception(f'Unexpected audio length {len(audio)}.  Max should be {self.max_src_length}')
            audios.append(audio.squeeze())
            audio_lengths[i] = len(audio)
            text_lengths[i] = len_text
        mx_src_seen = audio_lengths.max()
        zp_audio = np.zeros((len(batch), mx_src_seen), dtype=np.float32)
        for i, audio in enumerate(audios):
            zp_audio[i, : len(audio)] = audio
        mx_dst_seen = min(text_lengths.max(), self.max_dst_length)
        return {
            'signal': zp_audio[:, :mx_src_seen],
            'signal_lengths': audio_lengths,
            'token_ids': zp_text[:, :mx_dst_seen],
            'token_lengths': text_lengths,
            'files': files,
        }

    def process_sample(self, file):
        """Read in a line and turn it into an entry.  FIXME, get from anywhere

        The entries will get collated by the data loader

        :param file:
        :return:
        """
        return self.reader.read(file)


class AudioFileDataset(IterableDataset):
    def __init__(
        self,
        manifest,
        max_length,
        target_tokens_per_batch,
        distribute=True,
        shuffle=True,
        min_length=0,
        input_sample_rate=16_000,
        target_sample_rate=16_000,
    ):
        super().__init__()
        self.reader = (
            AudioResampleReader(target_sample_rate / input_sample_rate)
            if input_sample_rate != target_sample_rate
            else SoundfileAudioReader()
        )
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
        all_workers = self.world_size * num_workers_per_node
        offset = self.rank * num_workers_per_node + node_worker_id
        read_file_order = list(range(offset, len(self.files), all_workers))
        if not read_file_order:
            if offset > 0:
                # This is probably wrong
                logger.warning(
                    f"There are no files to read for worker {node_worker_id}, offset {offset}!"
                    + " This might mean that you are passing an incorrect training or validation directory"
                )
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
        return self.reader.read(file, len)

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
                # logger.debug("(%d, %d) %d", batch.shape[0], batch.shape[1], np.product(batch.shape))
                yield batch


def find_fit(v, fits):
    truncate_to = 0
    for fit in fits:
        if v // fit:
            truncate_to = fit
        else:
            break
    return truncate_to


class BucketingAudioDataset(AudioFileDataset):
    def __init__(
        self, buckets, manifest, max_length, target_tokens_per_batch, distribute=True, shuffle=True, min_length=0
    ):
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
