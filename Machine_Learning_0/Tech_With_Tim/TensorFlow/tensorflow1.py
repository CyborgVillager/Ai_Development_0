from tensor_source import *

data = keras.datasets.imdb
(train_data, train_labels), (test_data, test_labels) = data.load_data(num_words=10000)

# This results aquire the integer encoded words of 'movie reviews' from keras.datasets.imdb
# Each of these int points to certain words, helps the user to find specific information of their choosing by
# making the movie review into a list, now need to work on a mapping to make it easier for humans to read it
# print(train_data[0])

words_index = data.get_word_index()

# This breaks the tuple from line 12 into k & v -> key & value
# for words_index & add the keys from k:(v+3) into the dataset
words_index = {k: (v + 3) for k, v in words_index.items()}

# These words are from the training data set which are associated  with the keys
# as of now just adding 3 from line 16 to make our own values that stands for line 20-23
# PAD = padding, Unk = unknown
# this is a word map
words_index['<PAD>'] = 0  # just makes the movies the same length the padding will give its 'pads'
words_index['<START>'] = 1
words_index['<UNK>'] = 2
words_index['<UNUSED>'] = 3

# swap the values in the keys, which makes the dict to have the value 1st as in int goes to the word
# similiar to  print(train_data[0]) but for humans @ line 10.
# basically just reversing the info dict
reverse_word_index = dict([(value, key) for (key, value) in words_index.items()])


#  Pre-processing the data, makes the form consistent
train_data = keras.preprocessing.sequence.pad_sequences(train_data, value=words_index['<PAD>'],
                                                        padding='post', maxlen=250) # makes the padding for all the info

test_data = keras.preprocessing.sequence.pad_sequences(test_data, value=words_index['<PAD>'],
                                                        padding='post', maxlen=250)

# preprocessing  RESULT
print('this is preprocessing results')
print(len(train_data), len(test_data))
# Func Decoder for human lang -> English
def decoder_review(text):
    return ' '.join([reverse_word_index.get(index0, '?') for index0 in text])

# Space for Results
def space():
    print('\n')


# Defining the model
model = keras.Sequential()
model.add(keras.layers.Embedding(10000,16))
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(16, activation='relu'))
model.add(keras.layers.Dense(1,activation='sigmoid'))
# Checks whether or not the final review is good or bad


# Decoder Results
print('This is decoder review results')
space()
print(decoder_review(test_data[0]))
space()
print(decoder_review(test_data[1]))
