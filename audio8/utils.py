import eight_mile.optz
import eight_mile.pytorch.optz


def get_lr_decay(lr, decay_steps, sched_type='cosine', alpha=0.0):
    params = {'decay_steps': decay_steps, 'alpha': alpha}
    lr_decay = eight_mile.optz.create_lr_scheduler(lr_scheduler_type=sched_type, lr=lr, **params)
    return lr_decay


def create_lrs(lr, train_steps, sched_type='cosine', alpha=0.0, warmup_steps=10000, plateau_steps=0, **kwargs):
    lr_decay = get_lr_decay(lr, train_steps, sched_type, alpha)
    linear_warmup = eight_mile.pytorch.optz.WarmupLinearSchedulerPyTorch(warmup_steps, lr=lr)
    lr_sched = eight_mile.optz.CompositeLRScheduler(linear_warmup, lr_decay, plateau_steps, lr=lr)
    return lr_sched
