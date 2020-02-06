from tensor_source import *


data = keras.datasets.imdb
(train_data,train_labels), (test_data, test_labels) = data.load_data(num_words=10000)

# This results aquire the integer encoded words of 'movie reviews' from keras.datasets.imdb
# Each of these int points to certain words, helps the user to find specific information of their choosing by
# making the movie review into a list, now need to work on a mapping to make it easier for humans to read it
print(train_data[0])