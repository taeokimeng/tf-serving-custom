# TensorFlow Custom Serving
TensorFlow Custom Serving is for customizing ```serving_default```, and it allows changing the data type of input and output of Serving.

## How to do
1. If you don't have a model, please save a model as SavedModel.
    * Run ```model_save.py``` to save a model.

2. Customize ```serving_default``` of the SavedModel.
   * Run ```custom_image_classifier.py```.
   
3. Update ```models/models.config``` according to the generated models.

4. Deploy models with TensorFlow Serving.
~~~
./serving_run.sh
~~~

5. Request a prediction and get the response.
   * Run ```prediction_request.py```.
   