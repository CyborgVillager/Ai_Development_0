from tensor_source import *

data = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = data.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Shrinks the data
train_images = train_images/255.0
test_images = test_images/255.0

# Neural Network
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28,28)),
    # 128 nuerons
    # reatify linear unit
    keras.layers.Dense(128, activation='relu'),
    # 10 nuerons for output , softmax -> pick values from each nueron to = 2 one
    keras.layers.Dense(10,activation='softmax')
    ])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# epochs = how many times the model will see the information, how many times you will see the same images
model.fit(train_images,train_labels, epochs=5)

'''
# show image of the data
print(train_images[5])
plt.imshow(train_images[5], cmap=plt.cm.binary)
plt.show()
'''

test_loss, test_acc = model.evaluate(test_images, test_labels)
print('Tested Acc: ', test_acc)