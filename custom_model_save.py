import tensorflow as tf
import tensorflow_hub as hub


class CustomMobileNet_string(tf.keras.Model):
    model_handler = "https://tfhub.dev/google/imagenet/mobilenet_v2_035_224/classification/4"

    def __init__(self):
        super(CustomMobileNet_string, self).__init__()
        self.model = hub.load(self.__class__.model_handler)
        self.labels = None

    # Design your API with 'tf.function' decorator
    @tf.function(input_signature=[tf.TensorSpec(shape=None, dtype=tf.string)])
    def call(self, input_img):
        def _preprocess(img_file):
            img_bytes = tf.reshape(img_file, [])
            img = tf.io.decode_jpeg(img_bytes, channels=3)
            img = tf.image.convert_image_dtype(img, tf.float32)
            return tf.image.resize(img, (224, 224))

        labels = tf.io.read_file(self.labels)
        labels = tf.strings.split(labels, sep='\n')
        img = _preprocess(input_img)[tf.newaxis, :]
        logits = self.model(img)
        get_class = lambda x: labels[tf.argmax(x)]
        class_text = tf.map_fn(get_class, logits, tf.string)
        return class_text  # index of the class


model_string = CustomMobileNet_string()
# Save the image labels as an asset, saved in 'Assets' folder
model_string.labels = tf.saved_model.Asset("./data/labels/ImageNetLabels.txt")
tf.saved_model.save(model_string, "./models/mobilenet_v2/1/")