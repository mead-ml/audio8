"""Training using 8-mile API

"""
import logging
import time
import torch
import os
import contextlib
from argparse import ArgumentParser
from audio8.data import AudioTextLetterDataset
from audio8.text import TextVectorizer, read_vocab_list
from audio8.wav2vec2 import create_acoustic_model, load_fairseq_bin
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from eight_mile.utils import str2bool, Average, get_num_gpus_multiworker, Offsets, revlut
from eight_mile.pytorch.layers import save_checkpoint, init_distributed, sequence_mask, find_latest_checkpoint
from eight_mile.pytorch.optz import OptimizerManager
from audio8.ctc import CTCLoss, ctc_metrics, postproc_bpe, postproc_letters
from audio8.utils import create_lrs

logger = logging.getLogger(__file__)
Offsets.GO = 0
Offsets.PAD = 1
Offsets.VALUES[Offsets.GO] = '<s>'
Offsets.VALUES[Offsets.PAD] = '<pad>'
Offsets.VALUES[Offsets.EOS] = '</s>'
Offsets.VALUES[Offsets.UNK] = '<unk>'


def run_step(index2vocab, model, batch, loss_function, device, training=True, use_bpe=False, ctc_decoder=None):

    inputs, input_lengths, targets, target_lengths, _ = batch
    pad_mask = sequence_mask(input_lengths, inputs.shape[1]).to(device=device)
    inputs = inputs.to(device)
    targets = targets.to(device)

    logits, pad_mask = model(inputs, pad_mask)
    output_lengths = pad_mask.sum(-1)
    loss = loss_function(logits.transpose(1, 0), output_lengths, targets, target_lengths)
    num_tokens = target_lengths.sum().item()
    metrics = {}
    metrics['batch_size'] = inputs.shape[0]
    metrics['num_tokens'] = num_tokens
    if not training:
        postproc_fn = postproc_letters if not use_bpe else postproc_bpe
        logits = logits.detach().cpu()
        metrics.update(ctc_metrics(logits, targets, input_lengths, index2vocab, postproc_fn=postproc_fn))
        # In verbose, we will run the first item from the batch through the decoder and print it
        if ctc_decoder:
            transcriptions = ctc_decoder.run(logits[0].unsqueeze(0), n_best=1)
            transcription = ''.join(transcriptions[0])
            print(transcription)

    return loss, metrics


def train():
    parser = ArgumentParser()
    parser.add_argument("--basedir", type=str)
    parser.add_argument("--root_dir")
    parser.add_argument("--train_dataset", type=str, help='Dataset (by name), e.g. train-clean-360')
    parser.add_argument("--valid_dataset", type=str, help='Dataset (by name), e.g. dev-other')
    parser.add_argument("--input_sample_rate", type=int, default=16_000)
    parser.add_argument("--target_sample_rate", type=int, default=16_000)
    parser.add_argument("--dict_file", type=str, help="Dictionary file", default='dict.{}.txt')
    parser.add_argument("--dataset_key", default="LibriSpeech", help="dataset key for basedir")
    parser.add_argument("--grad_accum", type=int, default=2)
    parser.add_argument("--loss_reduction_type", type=str, default="sum", choices=["sum", "mean"])
    parser.add_argument("--d_model", type=int, default=768, help="Model dimension (and embedding dsz)")
    parser.add_argument("--d_ff", type=int, default=3072, help="FFN dimension")
    parser.add_argument("--d_k", type=int, default=None, help="Dimension per head.  Use if num_heads=1 to reduce dims")
    parser.add_argument("--num_heads", type=int, default=12, help="Number of heads")
    parser.add_argument("--num_layers", type=int, default=12, help="Number of layers")
    parser.add_argument("--num_train_workers", type=int, default=4, help="Number train workers")
    parser.add_argument("--max_sample_len", type=int, help="Max sample length")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout")
    parser.add_argument("--layer_drop", type=float, default=0.0, help="Layer Dropout")
    parser.add_argument("--lr_scheduler", type=str, default='cosine', help="The type of learning rate decay scheduler")
    parser.add_argument("--lr_alpha", type=float, default=0.0, help="parameter alpha for cosine decay scheduler")
    parser.add_argument("--optim", default="adamw", type=str, help="Optimizer to use (defaults to adamw)")
    parser.add_argument("--lr", type=float, default=1.0e-4, help="Learning rate")
    parser.add_argument("--clip", type=float, default=25.0, help="Clipping gradient norm")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay")
    parser.add_argument(
        "--restart_tt", type=str, help="Optional param for legacy checkpoints", choices=['step', 'ignore']
    )
    parser.add_argument("--restart_from", type=str, help="Option allows you to restart from a previous checkpoint")
    parser.add_argument("--warmup_steps", type=int, default=10000, help="Num warmup steps")
    parser.add_argument("--plateau_steps", type=int, default=0, help="Num plateau steps")
    parser.add_argument("--model_type", default="wav2vec2")
    parser.add_argument("--unfreeze_enc_after_step", default=10_000, type=int)
    parser.add_argument(
        "--timestep_masking", type=float, default=0.5, help="Timestep masking prob, gets divided by timestep_mask_len"
    )
    parser.add_argument("--timestep_mask_len", type=int, default=10, help="Num consecutive timesteps to mask")
    parser.add_argument(
        "--channel_masking", type=float, default=0.1, help="Channel masking prob, gets divided by channel_mask_len"
    )
    parser.add_argument("--channel_mask_len", type=int, default=64, help="Num consecutive channels to mask")
    parser.add_argument("--train_steps", type=int, default=320_000, help="Num training steps")
    parser.add_argument(
        "--valid_steps", type=int, default=1000, help="Maximum number of valid steps to evaluate each time"
    )
    parser.add_argument("--steps_per_checkpoint", type=int, default=2400, help="The number of steps per checkpoint")
    parser.add_argument("--verbose", type=str2bool, help="Verbose", default=False)
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cuda or cpu)"
    )
    parser.add_argument("--distributed", type=str2bool, default=False, help="Are we doing distributed training?")
    parser.add_argument("--vocab_file", help="Vocab for output decoding")
    parser.add_argument("--early_stopping_metric", type=str, help="Use early stopping on the key specified")
    parser.add_argument("--target_tokens_per_batch", type=int, default=700_000)
    parser.add_argument("--target_type", type=str, choices=["wrd", "ltr", "bpe"], default="ltr")
    parser.add_argument("--freeze_fx", type=str2bool, help="Freeze feature extractor", default=True)
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="Local rank for distributed training (-1 means use the environment variables to find)",
    )
    parser.add_argument("--lm")
    parser.add_argument("--beam", type=int, default=1, help="Beam size")
    parser.add_argument("--alpha", type=float, default=0.7)
    parser.add_argument("--beta", type=float, default=5.0)
    args = parser.parse_args()

    args.dict_file = args.dict_file.format(args.target_type)
    # Get the basedir to save results and checkpoints
    if args.basedir is None:
        args.basedir = f'{args.model_type}-{args.dataset_key}-{os.getpid()}'
    os.makedirs(args.basedir, exist_ok=True)

    # Setup logger
    logging.basicConfig(level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    num_gpus = get_num_gpus_multiworker()

    args.distributed = args.distributed or num_gpus > 1
    logger.info(f"Using {num_gpus} GPUs in this job.")

    if args.distributed:
        args.device, updated_local_rank = init_distributed(args.local_rank)
        args.local_rank = updated_local_rank

    if args.early_stopping_metric is not None:
        logger.info(f"Using {args.early_stopping_metric} for early stopping")
    else:
        logger.info("Early stopping will be turned off")

    vocab_file = args.vocab_file if args.vocab_file else os.path.join(args.root_dir, args.dict_file)
    vocab_list = read_vocab_list(vocab_file)
    ctc_decoder = None
    if args.verbose:
        from ctc import PrefixBeamSearch
        ctc_decoder = PrefixBeamSearch(
            vocab_list,
            alpha=args.alpha,
            beta=args.beta,
            beam=args.beam,
            lm_file=args.lm,
        )
    vocab = {v: i for i, v in enumerate(vocab_list)}

    vec = TextVectorizer(vocab)
    index2vocab = revlut(vocab)
    train_dataset = os.path.join(args.root_dir, args.train_dataset)
    valid_dataset = os.path.join(args.root_dir, args.valid_dataset)

    train_set = AudioTextLetterDataset(
        train_dataset,
        vec,
        args.target_tokens_per_batch,
        args.max_sample_len,
        input_sample_rate=args.input_sample_rate,
        target_sample_rate=args.target_sample_rate,
        shuffle=True,
        distribute=args.distributed,
        tgt_type=args.target_type,
    )
    valid_set = AudioTextLetterDataset(
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
    train_loader = DataLoader(train_set, batch_size=None, num_workers=args.num_train_workers, pin_memory=True)
    valid_loader = DataLoader(valid_set, batch_size=None, pin_memory=True)

    logger.info("Loaded datasets")

    num_labels = len(vocab)
    model = create_acoustic_model(num_labels, args.target_sample_rate // 1000, **vars(args)).to(args.device)
    use_bpe = True if args.target_type == 'bpe' else False
    loss_function = CTCLoss(reduction_type=args.loss_reduction_type).to(args.device)
    logger.info("Loaded model and loss")

    validate_on = min(args.train_steps // 2, args.steps_per_checkpoint)
    report_on = max(10, args.steps_per_checkpoint) // 10
    lr_sched = create_lrs(
        args.lr,
        args.train_steps,
        args.lr_scheduler,
        alpha=args.lr_alpha,
        warmup_steps=args.warmup_steps,
        plateau_steps=args.plateau_steps,
    )

    global_step = 0
    if args.restart_from:

        if args.restart_from.endswith('.pt'):
            # The pretrained fairseq checkpoints differ depending on whether they are pretrained wav2vec2
            # or wav2vec2-ctc.  First, we try loading as pretrained wav2vec2, then back off to ctc
            try:
                unmapped = load_fairseq_bin(model.encoder, args.restart_from)
            except:
                unmapped = load_fairseq_bin(model, args.restart_from, ctc=True, sr=args.target_sample_rate // 1000)
            print(unmapped)
            args.tick_type = None
        else:
            if os.path.isdir(args.restart_from):
                args.restart_from, _ = find_latest_checkpoint(args.restart_from)
            try:
                model.load_state_dict(torch.load(args.restart_from))
            except:
                print('Trying to load a8 checkpoint from pretrained wav2vec2 w/o CTC')
                unmapped = model.encoder.load_state_dict(torch.load(args.restart_from), strict=False)
                print(unmapped)
            if args.restart_tt:
                tick_type = args.restart_tt
            else:
                vec = args.restart_from.split("-")
                tick_type = vec[-2]

            if tick_type == 'step':
                vec = args.restart_from.split("-")
                step_num = int(vec[-1].split(".")[0])
                global_step = step_num
            else:
                logger.warning(f"Setting step number to 0")

            logger.info(
                "Restarting from a previous checkpoint %s.\n\tStarting at global_step=%d",
                args.restart_from,
                global_step,
            )

    optimizer = OptimizerManager(
        model, global_step, optim=args.optim, lr=args.lr, lr_function=lr_sched, weight_decay=args.weight_decay
    )
    logger.info("Model has {:,} parameters".format(sum(p.numel() for p in model.parameters() if p.requires_grad)))

    # Prepare model for distributed training if needed
    if args.distributed:
        # This program assume pure data parallelism, each model is on a single gpu
        # If we wanted to support model and data parallelism we would need to update
        # the selection of gpus based on rank, it would need to select multiple ids
        # based on rank, here we select only a single gpu and use it for input and
        # output.
        model = DistributedDataParallel(
            model, device_ids=[args.device], output_device=args.device, find_unused_parameters=True
        )
        _model = model.module
        logger.info("Model located on %s", args.device)
    else:
        _model = model
    model_base = os.path.join(args.basedir, 'checkpoint')

    train_itr = iter(train_loader)
    avg_loss = Average('average_train_loss')
    step_time = Average('average_step_time')
    batch_size_sent = Average('batch_size')
    batch_size_toks = Average('batch_toks')

    model.train()
    # All of our early stopping metrics currently need to be lower to be better, so set to high number initially
    best_metric = 1e8
    batch_size = torch.tensor(0, device=args.device, dtype=int)
    num_tokens_this_batch = torch.tensor(0, device=args.device, dtype=int)

    iters = 0
    start = time.time()

    optimizer.zero_grad()

    while optimizer.global_step < args.train_steps:

        try:

            if optimizer.global_step > args.unfreeze_enc_after_step:
                _model.freeze = False
            metrics = {}
            iters += 1
            is_dist_step = iters % args.grad_accum == 0
            with model.no_sync() if (args.distributed and not is_dist_step) else contextlib.ExitStack():
                # This loader will iterate for ever
                batch = next(train_itr)

                loss, step_metrics = run_step(
                    index2vocab, model, batch, loss_function, args.device,
                )
                num_tokens_this_batch += step_metrics['num_tokens']
                batch_size += step_metrics['batch_size']

                avg_loss.update(loss.item())
                loss.backward()
                if is_dist_step: 
                    # This will allreduce the gradients which will be scaled by 1/num_gpus
                    if args.distributed:
                        torch.distributed.all_reduce(batch_size)
                        torch.distributed.all_reduce(num_tokens_this_batch)

                    optimizer.scale_grads(num_gpus / batch_size)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
                    optimizer.step()
                    optimizer.zero_grad()
                    batch_size_sent.update(batch_size.cpu().item())
                    batch_size_toks.update(num_tokens_this_batch.cpu().item())
                    num_tokens_this_batch *= 0
                    batch_size *= 0
                    elapsed = time.time() - start
                    step_time.update(elapsed)
                    start = time.time()

                    if optimizer.global_step % report_on == 0:
                        if step_time.avg != 0:
                            steps_per_sec = 1.0 / step_time.avg
                            logging.info(
                                '%s, steps/min %f, LR %.6f, batch (samples %.2f, toks %.2f, toks/min %.2f)',
                                avg_loss,
                                steps_per_sec * 60,
                                optimizer.current_lr,
                                batch_size_sent.avg,
                                batch_size_toks.avg,
                                batch_size_toks.avg * steps_per_sec * 60,
                            )

                    if (
                        optimizer.global_step % validate_on == 0 and args.local_rank < 1
                    ):
                        train_token_loss = avg_loss.avg
                        metrics['average_train_loss'] = train_token_loss
                        avg_valid_loss = Average('average_valid_loss')

                        model.eval()
                        valid_start = time.time()
                        valid_itr = iter(valid_loader)
                        c_errors = 0
                        c_total = 0
                        w_errors = 0
                        w_total = 0

                        valid_metrics = {}
                        for j, batch in enumerate(valid_itr):
                            if j > args.valid_steps:
                                break

                            try:
                                with torch.no_grad():
                                    loss, valid_step_metrics = run_step(
                                        index2vocab,
                                        model,
                                        batch,
                                        loss_function,
                                        args.device,
                                        ctc_decoder=ctc_decoder,
                                        training=False,
                                        use_bpe=use_bpe,
                                    )
                                c_errors += valid_step_metrics['c_errors']
                                w_errors += valid_step_metrics['w_errors']
                                c_total += valid_step_metrics['c_total']
                                w_total += valid_step_metrics['w_total']
                                avg_valid_loss.update(loss.item())
                                elapsed = time.time() - valid_start
                                valid_token_loss = avg_valid_loss.avg
                                valid_metrics['average_valid_loss'] = valid_token_loss
                                valid_metrics['valid_elapsed_epoch'] = elapsed
                                valid_metrics['cer'] = (c_errors / c_total) * 100
                                valid_metrics['wer'] = (w_errors / w_total) * 100

                            except Exception as e:
                                logger.error(e)
                        logger.info(metrics)
                        logger.info(valid_metrics)
                        save_checkpoint(model, model_base, optimizer.global_step, tick_type='step')
                        if args.early_stopping_metric and valid_metrics[args.early_stopping_metric] < best_metric:
                            best_metric = valid_metrics[args.early_stopping_metric]
                            logger.info("New best metric %.4f", best_metric)
                            save_checkpoint(model, model_base, 0, tick_type='best')
                model.train()
        except Exception as e:
            logger.error(e)


if __name__ == "__main__":
    train()
