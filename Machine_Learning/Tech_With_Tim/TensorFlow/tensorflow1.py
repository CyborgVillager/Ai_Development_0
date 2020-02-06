from tensor_source import *


data = keras.datasets.imdb
(train_data,train_labels), (test_data, test_labels) = data.load_data(num_words=10000)
