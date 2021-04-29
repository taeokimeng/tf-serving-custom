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
   
## Serving with Kubernetes
1. Install [kubectl](https://kubernetes.io/docs/tasks/tools/install-kubectl-linux/) on Linux.
2. Install [minikube](https://minikube.sigs.k8s.io/docs/start/) to test locally.
3. Prepare an image for deployment
~~~
# Run TF Serving
docker run -d --name serving_base tensorflow/serving

# Copy the SavedModel
docker cp ./models/. serving_base:/models

# Copy tf_serving_entrypoint.sh
docker cp tf_serving_entrypoint.sh serving_base:/usr/bin/tf_serving_entrypoint.sh

# Commit
docker commit serving_base YOUR_DOCKER_HUB_USERNAME/inception_v3

# Stop and remove
docker kill serving_base
docker rm serving_base
~~~

4. Push the image to Docker Hub.
~~~
docker push YOUR_DOCKER_HUB_USERNAME/inception_v3
~~~

5. Run minikube. (To test locally)
~~~
minikube start
~~~

6. Start deployment and service.
~~~
kubectl apply -f ./inception_v3_k8s.yaml

# You can check with
kubectl get deployments
kubectl get pods
kubectl get services
kubectl describe service inceptionv3-service
~~~

7. Expose an external ip. (minikube use)
~~~
minikube tunnel
~~~

8. Run ```prediction_request.py``` and check the prediction.