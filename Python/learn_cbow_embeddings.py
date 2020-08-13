# !pip install tensorflow
# !pip install gensim
# !pip install bz2


import re
import os
import bz2
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from gensim.models import Word2Vec

import tensorflow as tf
from tensorflow.python.keras import models, layers, optimizers
from tensorflow.keras.preprocessing.text import Tokenizer, text_to_word_sequence
from tensorflow.keras.preprocessing.sequence import pad_sequences

from sklearn.metrics import f1_score, roc_auc_score, accuracy_score

import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

from util import get_labels_and_texts, normalize_texts, train_text_split_function

if __name__ == '__main__':
    # loading train texts and train labels
    print("started reading files")
    X_train, Y_train = get_labels_and_texts('../input/amazonreviews/train.ft.txt.bz2')
    print("done with loading training data")
    print("dimensions of training data after loading")
    print("len of x_train :", len(X_train))
    print("len of y_train :", len(Y_train))

    # preprocessing the texts
    X_train = normalize_texts(X_train)
    print("done with preprocessing training texts")
    print("dimensions of training data after loading")
    print("len of x_train :", len(X_train))
    print("len of y_train :", len(Y_train))

    # tokenizing text inputs to be passed to gensim model
    X_train = [tf.keras.preprocessing.text.text_to_word_sequence(text) for text in X_train]
    print(len(X_train))
    print(len(X_train[0]))

    logging.root.level = logging.INFO

    # learning CBOW word embeddings (sg = 0)
    model_skipgram = Word2Vec(sentences=X_train, size=300, window=10, max_vocab_size=20000, iter=2, sg=0,
                              min_count=0, workers=500)

    # Summarize the loaded model
    print(model_skipgram)

    # Summarize vocabulary
    print(" length of vocab :", len(list(model_skipgram.wv.vocab)))

    # save model in word2vec format , so that it can be loaded later
    model_skipgram.wv.save_word2vec_format('model_skipgram.bin')

    # small checks on learned embeddings
    print(model_skipgram.most_similar(positive=['woman', 'king'], negative=['man']))
    print(model_skipgram.similarity('easy', 'difficult'))
