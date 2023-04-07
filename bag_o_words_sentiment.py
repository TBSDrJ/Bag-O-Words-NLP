import random
import tensorflow as tf
import tensorflow.keras.layers as layers

binary_vectorization = layers.TextVectorization(output_mode='binary')

def vectorize_entry(sentence, label):
    print(sentence, label)
    sentence = tf.convert_to_tensor(sentence)
    sentence = tf.expand_dims(sentence, -1)
    label = tf.convert_to_tensor(label, dtype=tf.float32)
    return binary_vectorization(sentence), label


def datasetConfig(filename, batch=32, train_valid_split=0.30):
    # Set up movie reviews so it can be used as a bag of words
    with open(filename, 'r') as fin:
        lines = fin.readlines()
    sentences = []
    labels = []
    for line in lines:
        labels.append(int(line[0]))
        sentences.append(line[2:-1])
    # Zip will combine labels and sentences into a single iterator
    #   of 2-tuples, (sentence, label) so they'll stay together.
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
    # Set up vocabulary list using adapt().
    # It is important to use *only* the training data for this.
    train_text = [entry[0] for entry in train_data]
    binary_vectorization.adapt(train_text)
    # # Show results of vectorizing the data
    # ex_sentence, ex_label = vectorize_entry(train_data[0][0], train_data[0][1])
    # print(ex_sentence)
    # print(ex_label)
    # for i, entry in enumerate(ex_sentence.numpy()[0]):
    #     if int(entry) == 1:
    #         print(binary_vectorization.get_vocabulary()[i])
    
   

dataset = datasetConfig('movieReviews.txt')