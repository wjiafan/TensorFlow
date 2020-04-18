# Never mind this statement, for compatibility reasons
from __future__ import absolute_import, division, print_function, unicode_literals
 
# Import TensorFlow and TensorFlow Datasets
import tensorflow as tf
import tensorflow_datasets as tfds
tfds.disable_progress_bar()
  
# Helper libraries
import math
import numpy as np
import matplotlib.pyplot as plt

import logging
logger = tf.get_logger()
logger.setLevel(logging.ERROR) # 只打印ERROR

dataset, metadata = tfds.load('fashion_mnist', as_supervised=True, with_info=True)
train_dataset, test_dataset = dataset['train'], dataset['test']
class_names = [ 'T-shirt/top','Trouser','Pullover','Dress','Coat','Sandal','Shirt','Sneaker','Bag','Ankle boot' ]

num_train_examples = metadata.splits['train'].num_examples
num_test_examples = metadata.splits['test'].num_examples
print("Number of training examples: {}".format(num_train_examples))
print("Number of test examples:     {}".format(num_test_examples))
print(tf.__version__)

def normalize(images, labels):
	images = tf.cast(images, tf.float32)
	images /= 255
	return images, labels
	   
train_dataset =  train_dataset.map(normalize)
test_dataset  =  test_dataset.map(normalize)

for image, label in test_dataset.take(1):
	break
image = image.numpy().reshape((28,28))

# 显示前25张图片，在每张图片下显示类别
plt.figure(figsize=(10,10))
i = 0
for (image, label) in test_dataset.take(25):
	image = image.numpy().reshape((28,28))
	plt.subplot(5,5,i+1)
	plt.xticks([])
	plt.yticks([])
	plt.imshow(image, cmap=plt.cm.binary)
	plt.xlabel(class_names[label])
	i += 1
plt.savefig('./Clothing2.png')
plt.show()








