from keras.applications import VGG16
from keras.applications.vgg16 import preprocess_input, decode_predictions
import numpy as np
from keras import backend as K
from keras.preprocessing import image
import tensorflow as tf
import matplotlib as plt

model = VGG16(weights='imagenet')

img_path = '../images/elephant.jpg'

img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

preds = model.predict(x)
print(decode_predictions(preds, top=3)[0])
print(np.argmax(preds[0]))


african_elephant_output = model.output[:, 386]

last_conv_layer = model.get_layer('block5_conv3')
with tf.GradientTape() as gtape:
    grads = gtape.gradient(african_elephant_output, last_conv_layer.output)

pooled_grads = K.mean(grads, axis=(0, 1, 2))

iterate = K.function([model.input], [pooled_grads, last_conv_layer.output[0]])

pooled_grads_value, conv_layer_output_value = iterate([x])

for i in range(512):
    conv_layer_output_value[:, :, i] *= pooled_grads_value[i]
heatmap = np.mean(conv_layer_output_value, axis=-1)


heatmap = np.maximum(heatmap, 0)
heatmap /= np.max(heatmap)
plt.matshow(heatmap)
plt.show()
