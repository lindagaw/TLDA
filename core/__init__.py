from .adapt import train_tgt
from .pretrain import eval_src, train_src
from .test import eval_tgt
from .mahalanobis import train_detector, eval_detector

from .baseline_funcs import train_baseline, eval_baseline

from .deep_coral_funcs import CORAL, train_coral, eval_coral


__all__ = (eval_src, train_src, train_tgt, eval_tgt, train_detector, eval_detector, \
            train_baseline, eval_baseline, \
            train_coral, eval_coral, CORAL)
