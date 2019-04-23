#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from io import open
import torch


class Dictionary(object):
    def __init__(self):
        self.word2idx = {'<eos>': 0, '<unk>': 1}
        self.idx2word = ['<eos>', '<unk>']

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.word2idx)
        return self.word2idx[word]

    def getid(self, word):
        if word not in self.word2idx:
            return self.word2idx['<unk>']
        else:
            return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


class Corpus(object):
    def __init__(self):
        self.dictionary = Dictionary()

    def add_dict(self, path):
        assert os.path.exists(path)
        with open(path, 'r', encoding='utf-8') as fin:
            for line in fin:
                words = line.split(' ') + ['<eos>']
                for word in words:
                    self.dictionary.add_word(word)

    def tokenize(self, path):
        assert os.path.exists(path)
        with open(path, 'r', encoding='utf-8') as fin:
            ids = []
            for line in fin:
                words = line.split(' ') + ['<eos>']
                for word in words:
                    ids.append(self.dictionary.getid(word))
        return torch.LongTensor(ids)


if __name__ == '__main__':
    pass
