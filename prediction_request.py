import json
import requests
import base64


with open('./images/gf.jpg', mode='rb') as file:
    img = file.read()
data = {"inputs": [{"b64": base64.encodebytes(img).decode("utf-8")}]}

# Making the request
# localhost can be an external ip (Serving with Kubernetes)
json_response = requests.post("http://localhost:8501/v1/models/inception_v3/versions/2:predict", data=json.dumps(data))
# print(json_response.request)
# print(json_response.content)

predictions = json.loads(json_response.text)['outputs']
print("Predictions: ", predictions)

# This is for original prediction request
'''
data = json.dumps({"signature_name": "serving_default", "instances": input_image.tolist()})
headers = {"content-type": "application/json"}
json_response = requests.post(f'http://localhost:{port}/v1/models/{modeL_name}/versions/{ver}:predict',
                              data=data,
                              headers=headers)

predictions = json.loads(json_response.text)['predictions']
# label = decode_predictions(np.array(predictions))
label = modules.decode_predictions(np.array(predictions))
'''