from __future__ import annotations
from pprint import pprint
import random
import pickle
import re

import tensorflow as tf
from tensorflow.data import Dataset
import tensorflow.keras.layers as layers
from tensorflow.keras.models import Sequential
import tensorflow.keras.losses as losses
import tensorflow.keras.optimizers as optimizers

BATCH_SIZE = 16

vectorization = layers.TextVectorization(output_mode='binary')

"""
Goal of the dataset building is to get two Datasets, train
and test. Both should look like a list of pairs (example 
has batch size = 16):
(
    <tf.Tensor: shape=(16, 13910), dtype=float32, numpy=
    array([[0., 1., 1., ..., 0., 0., 0.],
       [0., 1., 1., ..., 0., 0., 0.],
       [0., 1., 1., ..., 0., 0., 0.],
       ...,
       [0., 1., 0., ..., 0., 0., 0.],
       [0., 0., 1., ..., 0., 0., 0.],
       [0., 0., 0., ..., 0., 0., 0.]], dtype=float32)>, 
    <tf.Tensor: shape=(16, 5), dtype=float32, numpy=
    array([[0., 0., 1., 0., 0.],
        [0., 0., 1., 0., 0.],
        [0., 1., 0., 0., 0.],
        ...,
        [0., 0., 1., 0., 0.],
        [0., 1., 0., 0., 0.],
        [0., 0., 1., 0., 0.]], dtype=float32)>
)
First half of the pair is the bag of words representation of
the sentence, the second is the one-hot encoding of the label
(for classification), or a length 1 vector with the star 
rating as its only value (for regression).
"""

def lines_to_pairs(lines: list[string]) -> list[(string, int)]:
    """Convert reviews to pairs: review, stars"""
    labels = []
    sentences = []
    for line in lines:
        # First char is num of stars
        labels.append(int(line[0]))
        # Skip stars, tab, and newline at end
        sentences.append(line[2:-1])
    data = list(zip(sentences, labels))
    random.shuffle(data)
    return data

def data_into_subsets(data: list[(string, int)], 
        ttsplit: float) -> list[str] | list[int] | list[str] | list[int]:
    """Separate data into the usual 4 subsets
    
    training data, training labels, validation data, validation labels
    Still have a long way to go to get where we want to go after this."""
    split_point = int(ttsplit * len(data))
    # Make the training dataset have a size that is evenly
    # divisible by the batch size. Notice that split_point is the size
    # of the *validation* set.
    split_point -= BATCH_SIZE - ((len(data) - split_point) % BATCH_SIZE)
    train_data = data[split_point:]
    # Now truncate the validation set to make its length divisible
    # by the batch size.
    split_point -= split_point - BATCH_SIZE
    valid_data = data[:split_point]
    train_sentences = [entry[0] for entry in train_data]
    train_labels = [entry[1] for entry in train_data]
    valid_labels = [entry[1] for entry in valid_data]
    valid_sentences = [entry[0] for entry in valid_data]
    return train_sentences, train_labels, valid_sentences, valid_labels

def configDataset(
        filename: string, 
        train_test_split: float=0.3) -> Dataset | Dataset | int:
    """Configure the datasets: train, validation
    
    Third return is the size of the vocabulary."""
    with open(filename, 'r') as fin:
        lines = fin.readlines()
    data = lines_to_pairs(lines)
    (train_sentences, train_labels, valid_sentences, 
            valid_labels) = data_into_subsets(data, train_test_split)
    # Use the next two lines for classification
    # train_labels = one_hot(train_labels)
    # valid_labels = one_hot(valid_labels)
    # Use the next two lines for regression
    # train_labels = one_dim(train_labels)
    # valid_labels = one_dim(valid_labels)
    # One-hot encoding
    train_one_hot = [[0,0,0,0,0].copy() for i in range(len(train_labels))]
    for i, entry in enumerate(train_labels):
        train_one_hot[i][entry] = 1
    valid_one_hot = [[0,0,0,0,0].copy() for i in range(len(valid_labels))]
    for i, entry in enumerate(valid_labels):
        valid_one_hot[i][entry] = 1
    # Break into batches
    tmplist = []
    for i in range(len(train_one_hot) // BATCH_SIZE):
        tmplist.append(train_one_hot[BATCH_SIZE * i:BATCH_SIZE * (i+1)])
    train_one_hot = tmplist
    # Attach the partial batch at the end
    # train_one_hot.append(train_one_hot_end)
    tmplist = []
    for i in range(len(valid_one_hot) // BATCH_SIZE):
        tmplist.append(valid_one_hot[BATCH_SIZE * i:BATCH_SIZE * (i+1)])
    valid_one_hot = tmplist
    # Attach the partial batch at the end
    # valid_one_hot.append(valid_one_hot_end)
    # Make the labels into Tensors
    train_labels_tf = [
        tf.convert_to_tensor(batch, dtype=tf.float32) for batch in train_one_hot
    ]
    valid_labels_tf = [
        tf.convert_to_tensor(batch, dtype=tf.float32) for batch in valid_one_hot
    ]
    # Now, move on to the training data
    # 'adapt' is a method that builds the vocabulary from the training set.
    vectorization.adapt(train_sentences)
    # Convert sentences to vectors. 
    # First, split into words, get rid of non-alphanumeric characters
    train_sentences = [sentence.strip().lower().split() for sentence in train_sentences]
    valid_sentences = [sentence.strip().lower().split() for sentence in valid_sentences]
    for sentence in train_sentences:
        for i, word in enumerate(sentence):
            tmp = re.sub(r"\W", '', word)
            if len(tmp) > 0:
                sentence[i] = tmp
    for sentence in valid_sentences:
        for i, word in enumerate(sentence):
            tmp = re.sub(r"\W", '', word)
            if len(tmp) > 0:
                sentence[i] = tmp
    vocab = vectorization.get_vocabulary()
    train_vectors = []
    for sentence in train_sentences:
        tmp_vec = [0] * (len(vocab) + 1)
        for word in sentence:
            if word in vocab:
                tmp_vec[vocab.index(word)] = 1
        train_vectors.append(tmp_vec)
    valid_vectors = []
    for sentence in valid_sentences:
        tmp_vec = [0] * (len(vocab) + 1)
        for word in sentence:
            if word in vocab:
                tmp_vec[vocab.index(word)] = 1
            else:
                tmp_vec[0] = 1
        valid_vectors.append(tmp_vec)
    for i, vector in enumerate(train_vectors):
        train_vectors[i] = vector[12:400]
    for i, vector in enumerate(valid_vectors):
        valid_vectors[i] = vector[12:400]
    tmplist = []
    for i in range(len(train_vectors) // BATCH_SIZE):
        tmplist.append(train_vectors[BATCH_SIZE * i: BATCH_SIZE * (i+1)])
    train_vectors = tmplist
    tmplist = []
    for i in range(len(valid_vectors) // BATCH_SIZE):
        tmplist.append(valid_vectors[BATCH_SIZE * i: BATCH_SIZE * (i+1)])
    valid_vectors = tmplist
    train_vectors_tf = [
        tf.convert_to_tensor(batch, dtype=tf.float32) for batch in train_vectors
    ]
    valid_vectors_tf = [
        tf.convert_to_tensor(batch, dtype=tf.float32) for batch in valid_vectors
    ]
    train = Dataset.from_tensor_slices((train_vectors_tf, train_labels_tf))
    valid = Dataset.from_tensor_slices((valid_vectors_tf, valid_labels_tf))
    return train, valid, len(vocab)

def build_model() -> Model:
    model = Sequential()
    model.add(layers.Dense(8192, activation='relu'))
    model.add(layers.Dense(2048, activation='relu'))
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dense(8, activation='relu'))
    model.add(layers.Dense(5, activation='softmax'))
    model.compile(
        loss = losses.CategoricalCrossentropy(),
        optimizer = optimizers.Adam(learning_rate=10**(-6)),
        metrics = ['accuracy'],
    )
    return model


def main():
    train, valid, vocab_size = configDataset('movieReviews.txt')

    model = build_model()
    model.fit(
        x = train,
        validation_data = valid,
        epochs = 75,
        batch_size = BATCH_SIZE,
    )
    model.save('live_bag_o_words')
    vocab = vectorization.get_vocabulary()
    with open('vocabulary.dat', 'wb') as fout:
        pickle.dump(vocab, fout)


if __name__ == "__main__":
    main()