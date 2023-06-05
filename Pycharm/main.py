import time
import torch
import numpy as np
from train import train_network, init_network
from importlib import import_module
from tensorboardX import SummaryWriter
from dataload import build_dataset, build_iterator, get_time_dif

if __name__ == '__main__':
    x = import_module('TextRNN')
    config = x.Config()

    # load data
    start_time = time.time()
    print('Loading data...')
    vocab, train_data, dev_data, test_data = build_dataset(config)
    train_iter = build_iterator(train_data, config)
    dev_iter = build_iterator(dev_data, config)
    test_iter = build_iterator(test_data, config)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)

    # train
    config.n_vocab = len(vocab)
    model = x.Model(config).to(config.device)
    writer = SummaryWriter(log_dir=config.log_path + '/' + time.strftime('%m-%d_%H.%M', time.localtime()))
    print(model.parameters())
    train_network(model, train_iter, dev_iter, test_iter, config, writer)
