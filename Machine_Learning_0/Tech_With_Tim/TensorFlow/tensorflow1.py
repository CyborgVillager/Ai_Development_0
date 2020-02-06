from tensor_source import *

data = keras.datasets.imdb
(train_data, train_labels), (test_data, test_labels) = data.load_data(num_words=88000)

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
                                                        padding='post',
                                                        maxlen=250)  # makes the padding for all the info

test_data = keras.preprocessing.sequence.pad_sequences(test_data, value=words_index['<PAD>'],
                                                       padding='post', maxlen=250)
'''
# preprocessing  RESULT
print('this is preprocessing results')
print(len(train_data), len(test_data))
'''

# Func Decoder for human lang -> English
def decoder_review(text):
    return ' '.join([reverse_word_index.get(index0, '?') for index0 in text])


# Space for Results
def space():
    print('\n')


# Defining the model
model = keras.Sequential()
model.add(keras.layers.Embedding(88000, 16))
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(16, activation='relu'))
model.add(keras.layers.Dense(1, activation='sigmoid'))
# Checks whether or not the final review is good or bad
model.summary()

# binary_crossentropy -> 2 options for open nueron 0 or 1 the loss will calc the differences 0.2 to 0 .
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# split the training data into 2 sets
# Validation Data is used to check how well the model is performing based on
# the training data for the new data
# basic thinking for the computer

x_val = train_data[:10000]
x_train = train_data[10000:]

y_val = train_labels[:10000]
y_train = train_labels[10000:]

fit_Model = model.fit(x_train, y_train, epochs=40, batch_size=512, validation_data=(x_val, y_val), verbose=1)
'''
# results for model
results = model.evaluate(test_data, test_labels)
print(results)
print('This is the results for accuracy')
space()
'''
# Saving the model
# h5 -> extension for saved model in keras
model.save(('model.h5'))
# Load the model
model = keras.models.load_model('model.h5')

def review_encode(string0):
    encoded = [1]

    for word in string0:
        if word.lower() in words_index:
            encoded.append(words_index[word.lower()])
        else:
            encoded.append(2)
    return encoded


# Testing the model
with open('test.txt', encoding='utf-8') as f:
    for line in f.readlines():
        nline = line.replace(",", "").replace(".", "").replace("(", "").replace(")", "").replace(":", "")\
            .replace("\"","").strip().split(" ")
# encoding
        encode = review_encode(nline)
        encode = keras.preprocessing.sequence.pad_sequences([encode], value=words_index['<PAD>'],
                                                               padding='post', maxlen=250)
        predict = model.predict(encode)
        print(line)
        print(encode)
        print(predict[0])



'''
# Results for movie review
test_review = test_data[0]
prediction = model.predict([test_review])
print('Review: ')
print(decoder_review(test_review))
print('Predicition: ' + str(prediction[0]))
print('Actual: ' + str(test_labels[0]))
print(results)
'''

'''
# Decoder Results

space()
print('This is decoder review results')
print(decoder_review(test_data[0]))
space()
print(decoder_review(test_data[1]))
'''
