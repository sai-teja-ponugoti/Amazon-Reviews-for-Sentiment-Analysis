# importing required modules
import re
import os
import bz2
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# %matplotlib inline

import gensim
from gensim.models import Word2Vec

import tensorflow
from tensorflow.keras import models, layers, optimizers
from tensorflow.keras.preprocessing import text, sequence
from tensorflow.keras.preprocessing.text import Tokenizer, text_to_word_sequence
from tensorflow.keras.preprocessing.sequence import pad_sequences

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score

from util import get_labels_and_texts, normalize_texts, train_text_split_function, padding_seq

# decalring regex varaibles and compiling them
NON_ALPHANUM = re.compile(r'[\W]')
NON_ASCII = re.compile(r'[^a-z0-1\s]')
max_features = 20000
maxlen = 200


# function to intialize tensorflow tokenizer
# commented as using a dumped tokenizer to lower processign time
def tokenizer_fun():
    tokenizer = text.Tokenizer(char_level=False, oov_token=None, num_words=20000)
    tokenizer.fit_on_texts(train_texts)
    print("finished")
    return tokenizer


# function to read the dumped tokenizer
def read_tokenizer_from_file(path):
    f = open(path, "r")
    tokenizer_string = f.read()
    f.close()
    return tensorflow.keras.preprocessing.text.tokenizer_from_json(tokenizer_string)


# function to tokenize the text sentences to list of numbers
def fit_on_texts(data,tokenizer):
    return tokenizer.texts_to_sequences(data)


# function to read learned embeddings from learn_cbow_word2vec and learn_skipgram_word2vec
def read_learned_embeddings(path):
    return gensim.models.KeyedVectors.load_word2vec_format(path, binary=False)


# function to build embedding matrix to be given as first layer in selected model
def build_embedding_matrix(embeddings,tokenizer):
    # extracting embedded vectors and vocab from embeddings
    embedding_vectors = embeddings.wv.vectors
    vocab = list(embeddings.wv.vocab)

    # declaring a zeros matrix
    embeddings_matrix = np.random.uniform(-0.05, 0.05, size=(20000, 300))

    count = 0
    for word, i in tokenizer.word_index.items():  # i=0 is the embedding for the zero padding
        count += 1
        try:
            if word in vocab:
                index = vocab.index(word)
            if index != 0:
                embedding_vector = embedding_vectors[index]
            else:
                embedding_vector = np.zeros((300,))
        except KeyError:
            embedding_vector = np.zeros((300,))
        if embedding_vector is not np.zeros((300,)):
            embeddings_matrix[i] = embedding_vector
        if count % 1000 == 0:
            print(count)
        if count > max_features - 2:
            break

        return embeddings_matrix


# function to create a model ,print summary and return the model
def build_model(embeddings_matrix):
    sequences = layers.Input(shape=(maxlen,))
    embedded = layers.Embedding(input_dim=embeddings_matrix.shape[0], output_dim=embeddings_matrix.shape[1],
                                weights=[embeddings_matrix], trainable=False, input_length=maxlen)(sequences)
    x = layers.Conv1D(64, 3, activation='relu')(embedded)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPool1D(3)(x)
    x = layers.Conv1D(64, 5, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPool1D(5)(x)
    x = layers.Conv1D(64, 5, activation='relu')(x)
    x = layers.GlobalMaxPool1D()(x)
    x = layers.Flatten()(x)
    x = layers.Dense(100, activation='relu')(x)
    predictions = layers.Dense(1, activation='sigmoid')(x)
    model = models.Model(inputs=sequences, outputs=predictions)
    opt = optimizers.Adam(lr=0.001)

    # compiling model using otimizer and loss functions
    model.compile(
        optimizer=opt,
        loss='binary_crossentropy',
        metrics=['binary_accuracy']
    )
    model.summary()
    return model


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

    #     splitting the dataset to train and test
    X_train, X_val, Y_train, Y_val = train_text_split_function(X_train, Y_train, 0.2)
    print("done with splitting training texts")
    print("dimensions of after splitting are loading")
    print("len of X_train :", len(X_train))
    print("len of Y_train :", len(Y_train))
    print("len of X_val :", len(X_val))
    print("len of Y_val :", len(Y_val))

    # intializing tokenizer
    # tokenizer = tokenizer_fun()

    # dumping tokenizer to a text file, so that it can be loaded later
    # f = open("tokenizer.txt", "w")
    # f.write(tokenizer.to_json())
    # f.close()

    # reading dumped tokenizer
    print("reading dumped tokenizer")
    tokenizer = read_tokenizer_from_file("../input/learned-models/tokenizer_tensorflow.txt")

    print("tokenizer read, moving to fitting texts")

    # fit tokenizer on train and val texts
    X_train = fit_on_texts(X_train, tokenizer)
    print("finished tokenizing train texts")
    X_val = fit_on_texts(X_val, tokenizer)
    print("finished tokenizing val texts")

    # padding train and val texts
    X_train = padding_seq(X_train)
    print("finished padding train texts")
    X_val = padding_seq(X_val)
    print("finished padding val texts")

    embeddings = read_learned_embeddings("../input/amazonpro/model_cbow.bin")
    print("loaded embeddings from files")

    embeddings_matrix = build_embedding_matrix(embeddings, tokenizer)
    print("finished building embedding matrix")

    model = build_model(embeddings_matrix)

    # fitting the model on train data
    history = model.fit(
        X_train,
        Y_train,
        batch_size=32,
        epochs=6,
        validation_data=(X_val, Y_val), verbose=1)

    # Saving the model , so that it can used for testing
    model.save('model.h5')

    # free up th variables
    train_texts = None
    train_labels = None
    val_texts = None
    val_labels = None
    tokenizer = None
