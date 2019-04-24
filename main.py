#!/usr/bin/python
# -*- coding: utf-8 -*-

import argparse
import time
import math
import os

import torch
import torch.nn as nn
import torch.onnx

import data
import nn_model


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data', '-train_data', type=str, required=True)
    parser.add_argument('--valid_data', '-valid_data', type=str, required=True)
    parser.add_argument('--corpus', '-corpus', type=str, default='corpus.obj')
    parser.add_argument('--model', '-model', type=str, default='LSTM',
                        help='type of recurrent net (RNN_TANH, RNN_RELU, LSTM, GRU)')
    parser.add_argument('--embed_size', '-emsize', type=int, default=200, help='word embedding size')
    parser.add_argument('--hidden_size', '-hdsize', type=int, default=200, help='hidden unit size')
    parser.add_argument('--layers', '-layers', type=int, default=2, help='number of layers')
    parser.add_argument('--learning_rate', '-lr', type=float, default=1, help='initial learning rate')
    parser.add_argument('--clip', '-cl', type=float, default=5.0, help='gradient clipping')
    parser.add_argument('--epochs', '-epc', type=int, default=40, help='number of epochs')
    parser.add_argument('--batch_size', '-bs', type=int, default=20, help='batch size')
    parser.add_argument('--bptt', '-bptt', type=int, default=35, help='seq length')
    parser.add_argument('--dropout', '-dr', type=float, default=0.3, help='dropout rate')
    parser.add_argument('--tied', '-tied', action='store_true', help='tie the word embedding and softmax weights')
    parser.add_argument('--seed', '-seed', type=int, default=1235, help='random seed')
    parser.add_argument('--gpu', '-gpu', type=int, default=-1, help='number of gpu')
    parser.add_argument('--log-interval', '-log', type=int, default=200, help='report interval')
    parser.add_argument('--save', '-save', type=str, default='model.pt', help='model file name')
    parser.add_argument('--onnx-export', '-onnx', type=str, default='', help='onnx export file path')
    return parser.parse_args()


def batchify(seq, bsz, device):
    nbatch = seq.size(0) // bsz
    seq = seq.narrow(0, 0, nbatch*bsz)
    seq = seq.view(bsz, -1).t().contiguous()
    return seq.to(device)


def repackage_hidden(h):
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)


def get_batch(source, i, bptt):
    seq_len = min(bptt, len(source) - 1 - i)
    _source = source[i: i + seq_len]
    target = source[i + 1: i + 1 + seq_len].view(-1)
    return _source, target


def export_onnx(model, path, bsz, seq_len, device):
    print('The model is also exported in ONNX format as {}'.format(os.path.realpath(path)))
    model.eval()
    dummy_input = torch.LongTensor(seq_len * bsz).zero_().view(-1, bsz).to(device)
    hidden = model.init_hidden(bsz)
    torch.onnx.export(model, (dummy_input, hidden), path)


def main():
    args = parse()
    torch.manual_seed(args.seed)
    if args.gpu >= 0:
        if not torch.cuda.is_available():
            print("WORNING: You seem not have a CUDA device, so I'll run on CPU")
            device = torch.device('cpu')
        else:
            device = torch.device('cuda:%d' % args.gpu)
    else:
        device = torch.device('cpu')

    corpus = data.Corpus()
    corpus.add_dict(args.train_data)
    with open(args.corpus, 'wb') as f:
        torch.save(corpus, f)

    eval_batch_size = 5
    train_data = batchify(corpus.tokenize(args.train_data), args.batch_size, device)
    val_data = batchify(corpus.tokenize(args.valid_data), eval_batch_size, device)

    ntokens = len(corpus.dictionary)
    bptt = args.bptt
    clip = args.clip
    model = nn_model.RNNLM(args.model, ntokens, args.embed_size, args.hidden_size, args.layers, args.dropout, args.tied).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())

    best_val_loss = None

    for e in range(1, args.epochs+1):
        epoch_start_time = time.time()
        model.train()
        total_loss = 0
        start_time = time.time()
        hidden = model.init_hidden(args.batch_size)
        for batch, i in enumerate(range(0, train_data.size(0) - 1, bptt)):
            sources, targets = get_batch(train_data, i, bptt)
            hidden = repackage_hidden(hidden)
            output, hidden = model(sources, hidden)
            loss = criterion(output.view(-1, ntokens), targets)
            model.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()

            total_loss += loss.item()

            if batch % args.log_interval == 0 and batch > 0:
                cur_loss = total_loss / args.log_interval
                elapsed_time = time.time() - start_time
                print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | '
                      'loss {:5.2f} | ppl {:8.2f}'.format(
                       e, batch, len(train_data) // bptt, optimizer.param_groups[0]['lr'],
                       elapsed_time * 1000 / args.log_interval, cur_loss, math.exp(cur_loss)))
                total_loss = 0
                start_time = time.time()
        val_loss = 0
        model.eval()
        hidden = model.init_hidden(eval_batch_size)
        with torch.no_grad():
            for batch, i in enumerate(range(0, val_data.size(0) - 1, bptt)):
                sources, targets = get_batch(val_data, i, bptt)
                output, hidden = model(sources, hidden)
                output_flat = output.view(-1, ntokens)
                val_loss += len(sources) * criterion(output_flat, targets).item()
                hidden = repackage_hidden(hidden)
        val_loss /= len(val_data)
        print('-' * 89)
        print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | valid ppl {:8.2f}'.format(
            e, (time.time() - epoch_start_time), val_loss, math.exp(val_loss)))
        print('-' * 89)
        if best_val_loss is None or val_loss < best_val_loss:
            with open(args.save, 'wb') as f:
                torch.save(model, f)
            best_val_loss = val_loss

    if len(args.onnx_export) > 0:
        export_onnx(model, args.onnx_export, 1, args.bptt, device)


if __name__ == '__main__':
    main()
