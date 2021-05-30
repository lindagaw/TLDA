"""Main script for ADDA."""
import pretty_errors
import os
from six.moves import urllib
opener = urllib.request.build_opener()
opener.addheaders = [('User-agent', 'Mozilla/5.0')]
urllib.request.install_opener(opener)

import params
from core import eval_src, eval_tgt, train_src, train_tgt, train_coral, eval_coral, CORAL
from models import Discriminator, LeNetClassifier, LeNetEncoder, Coral
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
    src_coral = init_model(net=Coral(),
                                restore='snapshots//src-coral-final.pt')

    tgt_coral = init_model(net=Coral(),
                                restore='snapshots//tgt-coral-final.pt')
    # train source coral
    print("=== Training coral for source domain ===")
    print(">>> Source Coral <<<")
    print(src_coral)


    src_coral = train_coral(src_coral, src_data_loader)

    # train target coral
    print("=== Training coral for target domain ===")
    print(">>> Target Coral <<<")
    print(tgt_coral)

    tgt_coral = train_coral(tgt_coral, tgt_data_loader)

    # eval source model on source data
    print("=== Evaluating source coral for source domain ===")
    eval_coral(src_coral, src_data_loader_eval)

    # eval target model on target data
    print("=== Evaluating target coral for target domain ===")
    eval_coral(tgt_coral, tgt_data_loader_eval)

    print('=====================================================')
    print('==================== TL/DA Magic ====================')
    print('=====================================================')

    print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
    # eval source model on target data
    print('=====================================================')
    print("=== Evaluating source baseline for target domain ===")
    print("======= what happens when D-Coral is applied  =======")
    print('=====================================================')

    # eval source model on target data
    print("=== Evaluating target coral for target domain ===")
    eval_coral(src_coral, tgt_data_loader_eval)
