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

'''
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('Tested Acc: ', test_acc)
'''
# Training the model
#prediction = model.predict(test_images)
# Predict for 1 image
prediction = model.predict(test_images)
# argmax -> gets the largest value and gets the largest index of that item
# loop through the images and see what the image is, for the user to validate the info
for index in range(5):
    plt.grid(False)
    plt.imshow(test_images[index],cmap=plt.cm.binary)
    plt.xlabel('Actual: ' + class_names[test_labels[index]])
    plt.title('Prediction: ' + class_names[np.argmax(prediction[index])])
    plt.show()



