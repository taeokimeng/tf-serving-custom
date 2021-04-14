from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_v3 import preprocess_input, decode_predictions
import numpy as np


MODEL_NAME = "inception_v3"
MODEL_VERSION = 1

model = InceptionV3(weights='imagenet')
target_size = (299, 299)

img_path = 'images/lion.jpg'
img = image.load_img(img_path, target_size=target_size)
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

preds = model.predict(x)
# decode the results into a list of tuples (class, description, probability)
# (one such list for each sample in the batch)
print('Predicted:', decode_predictions(preds, top=3)[0])

model.save(f'models/{MODEL_NAME}/{MODEL_VERSION}')