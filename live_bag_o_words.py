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
is classification and has batch size = 16):
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
    """Convert reviews to pairs: review, stars
    
    Reviews are strings, stars are integers."""
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
    Still have a long way to go to get where we want to go after this.
    We still have sentences as strings and labels as integers.
    Also make both train and validation sets into a size divisible by the
        batch size."""
    split_point = int(ttsplit * len(data))
    # Make the training dataset have a size that is evenly
    # divisible by the batch size. Notice that split_point is the size
    # of the *validation* set.
    split_point -= BATCH_SIZE - ((len(data) - split_point) % BATCH_SIZE)
    train_data = data[split_point:]
    # Now truncate the validation set to make its length divisible
    # by the batch size.
    split_point -= split_point % BATCH_SIZE
    valid_data = data[:split_point]
    train_sentences = [entry[0] for entry in train_data]
    train_labels = [entry[1] for entry in train_data]
    valid_labels = [entry[1] for entry in valid_data]
    valid_sentences = [entry[0] for entry in valid_data]
    return train_sentences, train_labels, valid_sentences, valid_labels

def one_hot(labels: list[int]) -> list[list[float]]:
    """Carry out one-hot encoding, convert entries to float."""
    one_hots = [[0., 0., 0., 0., 0.].copy() for i in range(len(labels))]
    for i, entry in enumerate(labels):
        one_hots[i][entry] = 1.0
    return one_hots

def one_dim(labels: list[int]) -> list[float]:
    """Make each entry into a list of length 1, change to float"""
    new_labels = [[entry * 1.0] for entry in labels]
    return new_labels

def batch(data: list[float] | list[list[float]]
        ) -> list[list[float]] | list[list[list[float]]]:
    """Break data into batches.
    
    Assumes that length of input is divisible by batch size, if not
        just loses any entries beyond last multiple of batch size."""
    tmplist = []
    for i in range(len(data) // BATCH_SIZE):
        tmplist.append(data[BATCH_SIZE * i:BATCH_SIZE * (i+1)])
    return tmplist

def clean_sentences(sentences: list[str]) -> list[list[str]]:
    """Convert each string into its component words."""
    sentences = [sentence.strip().lower() for sentence in sentences]
    # Get rid of all non-word characters using regular expression
    for i, sentence in enumerate(sentences):
        sentences[i] = re.sub(r'[^a-z\ ]', '', sentence)
    return sentences

def into_vectors(sentences: list[list[str]], vocab: list[str]
        ) -> list[list[float]]:
    """Convert list of words to vectors using bag of words method."""
    vectors = []
    for sentence in sentences:
        tmp_vec = [0] * (len(vocab) + 1)
        for word in sentence:
            if word in vocab:
                tmp_vec[vocab.index(word)] = 1
        vectors.append(tmp_vec)
    return vectors
    
def batch_to_tensor(data: list[list[float]]) -> list[tf.Tensor]:
    """Convert each batch in data to a tf.Tensor"""
    return [tf.convert_to_tensor(batch) for batch in data]

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
    train_labels = one_dim(train_labels)
    valid_labels = one_dim(valid_labels)

    # Switch to sentences to get those to a similar stage.
    # Convert sentences to vectors. 
    # First, split into words, get rid of non-alphanumeric characters
    train_sentences = clean_sentences(train_sentences)    
    valid_sentences = clean_sentences(valid_sentences)

    # 'adapt' is a method that builds the vocabulary from the training set.
    vectorization.adapt(train_sentences)
    vocab = vectorization.get_vocabulary()
    # If you want to truncate the vocabulary tested
    # vocab = vocab[:400]

    # Convert sentences to lists of words, and use the vocab to
    #   convert sentences to vectors
    train_sentences = [sentence.split() for sentence in train_sentences]
    valid_sentences = [sentence.split() for sentence in valid_sentences]
    train_vectors = into_vectors(train_sentences, vocab)
    valid_vectors = into_vectors(valid_sentences, vocab)

    # Now the data and labels are both ready to be batched:
    train_labels = batch(train_labels)
    valid_labels = batch(valid_labels)
    train_vectors = batch(train_vectors)
    valid_vectors = batch(valid_vectors)
    # Convert each batch into a single Tensor
    train_labels_tf = batch_to_tensor(train_labels)
    valid_labels_tf = batch_to_tensor(valid_labels)
    train_vectors_tf = batch_to_tensor(train_vectors)
    valid_vectors_tf = batch_to_tensor(valid_vectors)
    # Put data and labels back together into a tf Dataset and we're done!
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
    model.add(layers.Dense(1))
    model.compile(
        loss = losses.MeanSquaredError(),
        optimizer = optimizers.Adam(learning_rate=10**(-6)),
        # metrics = ['accuracy'],
    )
    return model

def save_stuff(model: models.Model):
    """Save model and vocabulary to a file."""
    model_path = 'live_bag_o_words'
    model.save(model_path)
    vocab = vectorization.get_vocabulary()
    with open(model_path + '/vocabulary.dat', 'wb') as fout:
        pickle.dump(vocab, fout)


def main():
    train, valid, vocab_size = configDataset('movieReviews.txt')

    model = build_model()
    try:
        model.fit(
            x = train,
            validation_data = valid,
            epochs = 75,
            batch_size = BATCH_SIZE,
        )
    # This makes it save when you hit CTRL-C, before it exits.
    except KeyboardInterrupt:
        save_stuff(model)
    # This will get it to save if it gets to the end of the last epoch.
    save_stuff(model)

if __name__ == "__main__":
    main()