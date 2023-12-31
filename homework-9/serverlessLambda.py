import numpy as np
import tflite_runtime.interpreter as tflite


from io import BytesIO
from urllib import request

from PIL import Image

def download_image(url):
    with request.urlopen(url) as resp:
        buffer = resp.read()
    stream = BytesIO(buffer)
    img = Image.open(stream)
    return img


def prepare_image(img, target_size):
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize(target_size, Image.NEAREST)
    return img


def make_X_array(img):
    x_array = np.array(img, dtype='float32')
    X_array = np.array([x_array])
    X_array = X_array / 255.0
    return X_array


interpreter = tflite.Interpreter(model_path='bees-wasps-v2.tflite')
interpreter.allocate_tensors()

input_index = interpreter.get_input_details()[0]['index']
output_index = interpreter.get_output_details()[0]['index']

classes = ['bee', 'wasp']
threshold = 0.5

# url = 'https://habrastorage.org/webt/rt/d9/dh/rtd9dhsmhwrdezeldzoqgijdg8a.jpeg'

def predict(url):
    image = download_image(url)
    image = prepare_image(image, (150, 150))
    X = make_X_array(image)
    
    interpreter.set_tensor(input_index, X)
    interpreter.invoke()
    preds = interpreter.get_tensor(output_index)
    return float(preds[0,0])


def lambda_handler(event, context):
    url = event['url']
    pred = predict(url)
    result = {
        'prediction': pred
    }

    return result
