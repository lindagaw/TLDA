"""Main script for ADDA."""
import pretty_errors
import os
from six.moves import urllib
opener = urllib.request.build_opener()
opener.addheaders = [('User-agent', 'Mozilla/5.0')]
urllib.request.install_opener(opener)

import params_gradientReversal as params
from core import eval_src, eval_tgt, train_src, train_tgt, train_gradientReversal, eval_gradientReversal
from models import Discriminator, LeNetClassifier, LeNetEncoder, GradientReversal
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
    src_gradientReversal = init_model(net=GradientReversal(),
                                restore='snapshots//src-gradientReversal-final.pt')

    tgt_gradientReversal = init_model(net=GradientReversal(),
                                restore='snapshots//tgt-gradientReversal-final.pt')
    # train source gradientReversal
    print("=== Training gradientReversal for source domain ===")
    print(">>> Source GradientReversal <<<")
    print(src_gradientReversal)

    src_gradientReversal = train_gradientReversal(src_gradientReversal, src_data_loader)

    # train target gradientReversal
    print("=== Training gradientReversal for target domain ===")
    print(">>> Target GradientReversal <<<")
    print(tgt_gradientReversal)

    tgt_gradientReversal = train_gradientReversal(tgt_gradientReversal, tgt_data_loader)

    # eval source model on source data
    print("=== Evaluating source gradientReversal for source domain ===")
    eval_gradientReversal(src_gradientReversal, src_data_loader_eval)

    # eval target model on target data
    print("=== Evaluating target gradientReversal for target domain ===")
    eval_gradientReversal(tgt_gradientReversal, tgt_data_loader_eval)

    print('=====================================================')
    print('==================== TL/DA Magic ====================')
    print('=====================================================')

    print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
    # eval source model on target data
    print('=====================================================')
    print("=== Evaluating source gradientReversal for target domain ===")
    print("=== This is what happens if no TL/DA is applied  ===")
    print("=== get source model's classification on target  ===")
    print('=====================================================')
    eval_gradientReversal(src_gradientReversal, tgt_data_loader_eval)
