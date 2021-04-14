import tensorflow as tf
import json
import requests
import base64
import numpy as np


data = {}
with open('./images/gf.jpg', mode='rb') as file:
    img = file.read()

# data = base64.encodebytes(img).decode("utf-8")
data = base64.b64encode(img).decode("utf-8")
print(data)

input_img = base64.b64decode(data)


def _preprocess(img_file):
    print("IMG_FILE\n", img_file)
    img_bytes = tf.reshape(img_file, [])
    print("IMG_BYTES\n", img_bytes)
    img = tf.io.decode_jpeg(img_bytes, channels=3)
    print("DECODE_JPEG\n", img)
    img = tf.image.convert_image_dtype(img, tf.float32)
    print("CONVERT\n", img)
    return tf.image.resize(img, (299, 299))


img = _preprocess(input_img)[tf.newaxis, :]
print(img)

labels = tf.io.read_file('./data/labels/ImageNetLabels.txt')
print(labels)
labels = tf.strings.split(labels, sep='\n')
print(labels)
