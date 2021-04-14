import tensorflow as tf


class CustomInceptionV3(tf.keras.Model):
    def __init__(self):
        super(CustomInceptionV3, self).__init__()
        self.model = tf.saved_model.load('./models/inception_v3/1')
        self.labels = None

    # Design your API with 'tf.function' decorator
    @tf.function(input_signature=[tf.TensorSpec(shape=None, dtype=tf.string)])
    def call(self, input_img):
        def _preprocess(img_file):
            # img_file is base64 decoded data
            # img_file is byte string, and make it to a tensor
            img_bytes = tf.reshape(img_file, [])
            # Decode img_bytes to a uint8 tensor
            img = tf.io.decode_jpeg(img_bytes, channels=3)
            # Convert data type to float32
            img = tf.image.convert_image_dtype(img, tf.float32)
            # Resize the image as the desired input size of the model
            return tf.image.resize(img, (299, 299))

        # Read the label file and split line by line
        labels = tf.io.read_file(self.labels)
        labels = tf.strings.split(labels, sep='\n')
        # Preprocess the input image and create new axis. It is for a batch
        img = _preprocess(input_img)[tf.newaxis, :]
        # Input the image to the model and get the output (predictions of the each label)
        logits = self.model(img)
        # Get the label with the index of the largest value
        get_class = lambda x: labels[tf.argmax(x)]
        class_text = tf.map_fn(get_class, logits, tf.string)
        # Label string
        return class_text


# Load a SavedModel
model_string = CustomInceptionV3()
# Save the image labels as an asset, saved in 'Assets' folder
# You might need to remove the first line, "background"
model_string.labels = tf.saved_model.Asset("./data/labels/ImageNetLabels.txt")
# Save the model with the re-defined serving_default
tf.saved_model.save(model_string, "./models/inception_v3/2/")
