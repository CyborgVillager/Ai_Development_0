from tensor_source import *

data = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = data.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Shrinks the data
train_images = train_images/255.0
test_images = test_images/255.0


# show image of the data
print(train_images[5])
plt.imshow(train_images[5], cmap=plt.cm.binary)
plt.show()