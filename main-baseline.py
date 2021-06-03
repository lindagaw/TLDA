"""Main script for ADDA."""
import pretty_errors
import os
from six.moves import urllib
opener = urllib.request.build_opener()
opener.addheaders = [('User-agent', 'Mozilla/5.0')]
urllib.request.install_opener(opener)

import params
from core import eval_src, eval_tgt, train_src, train_tgt, train_baseline, eval_baseline
from models import Discriminator, LeNetClassifier, LeNetEncoder, Baseline
from utils import get_data_loader, init_model, init_random_seed

if __name__ == '__main__':
    # init random seed
    init_random_seed(params.manual_seed)

    # load dataset
    src_data_loader = get_data_loader(params.src_dataset)
    src_data_loader_eval = get_data_loader(params.src_dataset, train=False)
    tgt_data_loader = get_data_loader(params.tgt_dataset)
    tgt_data_loader_eval = get_data_loader(params.tgt_dataset, train=False)

    # load models
    src_baseline = init_model(net=Baseline(),
                                restore='snapshots//src-baseline-final.pt')

    tgt_baseline = init_model(net=Baseline(),
                                restore='snapshots//tgt-baseline-final.pt')
    # train source baseline
    print("=== Training baseline for source domain ===")
    print(">>> Source Baseline <<<")
    print(src_baseline)

    src_baseline = train_baseline(src_baseline, src_data_loader)

    # train target baseline
    print("=== Training baseline for target domain ===")
    print(">>> Target Baseline <<<")
    print(tgt_baseline)

    tgt_baseline = train_baseline(tgt_baseline, tgt_data_loader)

    # eval source model on source data
    print("=== Evaluating source baseline for source domain ===")
    eval_baseline(src_baseline, src_data_loader_eval)

    # eval target model on target data
    print("=== Evaluating target baseline for target domain ===")
    eval_baseline(tgt_baseline, tgt_data_loader_eval)

    print('=====================================================')
    print('==================== TL/DA Magic ====================')
    print('=====================================================')

    print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
    # eval source model on target data
    print('=====================================================')
    print("=== Evaluating source baseline for target domain ===")
    print("=== This is what happens if no TL/DA is applied  ===")
    print("=== get source model's classification on target  ===")
    print('=====================================================')
    eval_baseline(src_baseline, tgt_data_loader_eval)
