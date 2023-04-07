import random
import tensorflow as tf
import tensorflow.keras.layers as layers
from tensorflow.data import Dataset

binary_vectorization = layers.TextVectorization(output_mode='binary')

def vectorize_entry(sentence): 
    sentence = tf.expand_dims(sentence, -1)
    return binary_vectorization(sentence)

def datasetConfig(filename, batch=32, train_valid_split=0.30):
    # Set up movie reviews so it can be used as a bag of words
    with open(filename, 'r') as fin:
        lines = fin.readlines()
    sentences = []
    labels = []
    for line in lines:
        labels.append(int(line[0]))
        sentences.append(line[2:-1])
    # Zip will combine labels and sentences into a single iterator of 2-tuples,
    #   (sentence, label) so they'll stay together even when we shuffle.
    # Having hard time getting zip object to iterate as I thought
    #   it should, so I converted to list.
    data = list(zip(sentences, labels))

    # # Show a sampling of the input data
    # print("\n" * 3)
    # print("Sampling of input data:", end="\n\n")
    # for i in range(10):
    #     # equivalent to labels[i], sentences[i]
    #     print(f"{data[i][1]}\t{data[i][0]}")
    # print()

    # Set up train-test split
    random.shuffle(data)
    split_point = int(train_valid_split*(len(data)))
    valid_data = data[:split_point]
    train_data = data[split_point:]
    # Gonna need to separate them back out again now that they are shuffled.
    train_text = [entry[0] for entry in train_data]
    train_labels = [entry[1] for entry in train_data]
    valid_text = [entry[0] for entry in valid_data]
    valid_labels = [entry[1] for entry in valid_data]
    # Set up vocabulary list using adapt().
    # It is important to use *only* the training data for this.
    binary_vectorization.adapt(train_text)

    # # Show results of vectorizing the data
    # ex_sentence, ex_label = vectorize_entry(train_data[0][0], train_data[0][1])
    # print(ex_sentence)
    # print(ex_label)
    # for i, entry in enumerate(ex_sentence.numpy()[0]):
    #     if int(entry) == 1:
    #         print(binary_vectorization.get_vocabulary()[i])
    
    train_text_tf = [tf.convert_to_tensor(sentence) for sentence in train_text]
    train_text_ds = Dataset.from_tensor_slices(train_text_tf)
    train_text_ds = train_text_ds.map(vectorize_entry)
    train_labels_tf = [tf.convert_to_tensor(label) for label in train_labels]
    train_labels_ds = Dataset.from_tensor_slices(train_labels_tf)
    train = Dataset.zip((train_text_ds, train_labels_ds))
    # print(train.element_spec)
    valid_text_tf = [tf.convert_to_tensor(sentence) for sentence in valid_text]
    valid_text_ds = Dataset.from_tensor_slices(valid_text_tf)
    valid_text_ds = valid_text_ds.map(vectorize_entry)
    valid_labels_tf = [tf.convert_to_tensor(label) for label in valid_labels]
    valid_labels_ds = Dataset.from_tensor_slices(valid_labels_tf)
    valid = Dataset.zip((valid_text_ds, valid_labels_ds))
    # print(valid.element_spec)
    return train, valid

train, valid = datasetConfig('movieReviews.txt')
