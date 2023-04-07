import random
import tensorflow.keras.layers as layers


def datasetConfig(filename, batch=32, train_test_split=0.30):
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
    print("\n" * 3)
    print("Sampling of input data:", end="\n\n")
    for i in range(10):
        # equivalent to labels[i], sentences[i]
        print(f"{data[i][1]}\t{data[i][0]}")
        

dataset = datasetConfig('movieReviews.txt')