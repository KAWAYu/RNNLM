#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import torch
import main as mmm
import math


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_file', '-file', required=True)
    parser.add_argument('--corpus', '-corpus', required=True)
    parser.add_argument('--model', '-model', required=True)
    parser.add_argument('--gpu', '-gpu', default=-1, type=int)
    return parser.parse_args()


def main():
    args = parse()
    with open(args.model, 'rb') as f:
        model = torch.load(f)
        model.rnn.flatten_parameters()

    with open(args.corpus, 'rb') as f:
        corpus = torch.load(f)

    if args.gpu >= 0:
        if not torch.cuda.is_available():
            print("WORNING: You seem not have a CUDA device, so I'll run on CPU")
            device = torch.device('cpu')
        else:
            device = torch.device('cuda:%d' % args.gpu)
    else:
        device = torch.device('cpu')

    eval_batch_size = 10
    test_data = mmm.batchify(corpus.tokenize(args.test_file), eval_batch_size, device)
    ntokens = len(corpus.dictionary)
    criterion = torch.nn.CrossEntropyLoss()
    test_loss = 0

    model.eval()
    hidden = model.init_hidden(eval_batch_size)
    with torch.no_grad():
        for batch, i in enumerate(range(0, test_data.size(0) - 1, 30)):
            sources, targets = mmm.get_batch(test_data, i, 30)
            output, hidden = model(sources, hidden)
            output_flat = output.view(-1, ntokens)
            test_loss += len(sources) * criterion(output_flat, targets).item()
            hidden = mmm.repackage_hidden(hidden)
    test_loss /= len(test_data)
    print('-' * 89)
    print('| test loss {:5.2f} | test ppl {:8.2f}'.format(test_loss, math.exp(test_loss)))
    print('-' * 89)


if __name__ == '__main__':
    main()
