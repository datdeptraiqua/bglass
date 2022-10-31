import cv2
from flask import Flask, request, jsonify, Response, flash, redirect, url_for
from flask_cors import CORS, cross_origin
import os, time
from PIL import Image
import numpy as np
from gtts import gTTS
from werkzeug.utils import secure_filename

import torch

# Disable Warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# SETUP
IMG_SIZE = 416
# LABELS = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
# 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant',
# 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
# 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle',
# 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
# 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table',  'toilet', 'tv',
# 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator',
# 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

"""
Box order: [ymin, xmin, ymax, xmax]
"""

# Khởi tạo Flask server Backend
app = Flask(__name__)
CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'
model_path = r'D:\Python\Pycharm\Flask\backend_server\Flask_backend_server\assets\yolov7.pt'
UPLOAD_FOLDER = r"static\table-chair.jfif"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


# Load models
def load_pt_model(path_to_model):
    # model = hub.load("https://tfhub.dev/tensorflow/ssd_mobilenet_v2/fpnlite_320x320/1")
    start = time.time()
    model = torch.hub.load("WongKinYiu/yolov7", 'custom', path_to_model)
    ## Set up params
    model.conf = 0.6  # NMS confidence threshold
    model.iou = 0.5  # NMS IoU threshold
    model.classes = [13, 56, 57, 60]  # bench, chair, couch, dining table
    model.max_det = 10  # maximum number of detections per image
    end = time.time()
    print(f"Model loaded in: {end - start} seconds")
    # model
    return model


# Predict
def singular_plural_none(number, object):
    singular = {'13': "bench", '56': "chair", '57': "couch", '60': "dining table"}
    plural = {'13': "benches", '56': "chairs", '57': "couches", '60': "dining tables"}
    if number == 0:
        return None
    elif number == 1:
        return singular[object]
    else:
        return plural[object]


def detect_fn_pt_audio(model, image):
    # print(image)
    # to_tensor = transforms.ToTensor()
    # img = to_tensor(image)
    # img = img.numpy()
    # print(type(img))
    detection = {'13': 0, '56': 0, '57': 0, '60': 0}

    results = model(image, size=416)

    if results.xyxy[0].nelement() != 0:
        for i in range(len(results.xyxy[0])):
            if results.xyxy[0][i][5] == 13:
                detection['13'] += 1
            elif results.xyxy[0][i][5] == 56:
                detection['56'] += 1
            elif results.xyxy[0][i][5] == 57:
                detection['57'] += 1
            elif results.xyxy[0][i][5] == 60:
                detection['60'] += 1

    num_item = sum(detection.values())
    # print(num_item)
    if num_item > 1:
        message_EN = "There are "
        for key in detection:
            item = singular_plural_none(detection[key], key)
            if item is not None:
                message_EN += f"{detection[key]} {item} "
    elif num_item == 1:
        message_EN = "There is "
        for key in detection:
            item = singular_plural_none(detection[key], key)
            if item is not None:
                message_EN += f"{detection[key]} {item} "
    else:
        message_EN = "There are no chairs, benched, couched or dining tables detected"

    message_VN = "Cảnh báo: "
    # if len(diagnosis) == 2:
    #     message_VN += "cây dưa chuột bị cả hai bệnh!"
    # elif diagnosis == [1]:
    #     message_VN += "cây dưa chuột bị bệnh phấn trắng!"
    # elif diagnosis == [2]:
    #     message_VN += "cây dưa chuột bị bệnh sương mai!"
    # elif diagnosis == [0]:
    #     message_VN = "cây dưa chuột khỏe mạnh"
    # else:
    #     message_VN = "Không nhận diện được lá dưa chuột"
    return message_EN, message_VN


# def results_handling(results):
#     labels = []
#     # boxes = []
#     position = []
#     dict = {}
#     for index, element in enumerate(results['detection_scores'][0] > 0.5):
#         if element == True:
#             # print(element)
#             labels.append(LABELS[int(results['detection_classes'][0][index]) - 1])
#             boxes = results["detection_boxes"][0][index].numpy()
#             position.append((boxes[0] + boxes[2]) / 2)
#
#     # for box in boxes:
#     #     xc = (box[0] + box[2]) / 2
#     #     position.append(xc)
#
#     for i, label in enumerate(labels):
#         dict[label] = position[i]
#
#     sorted_dict = sorted(dict.items(), key=lambda x: x[1])
#     message_EN = "From left to right: "
#     # message_VN = "Từ trái sang phải: "
#     for element in sorted_dict:
#         message_EN += f"{element[0]} "
#         # message_VN += f"{element[0]} "
#
#     return message_EN


def generate():
    path = "message.mp3"
    with open(path, 'rb') as fmp3:
        data = fmp3.read(1024)
        while data:
            yield data
            data = fmp3.read(1024)


# model = load_tf_model(model_path)
model = load_pt_model(model_path)

# Download mp3 files
@app.route('/mp3', methods=['GET'])
def get_mp3():
    return Response(generate(), mimetype="audio/mpeg3")


@app.route('/predict', methods=['POST'])
def prediction():
    t0 = time.time()
    file = request.files['image']
    # Read the image via file.stream
    img = Image.open(file.stream)

    # files = {'media': open(path_img, 'rb')}
    # requests.post(url, files=files)
    # global model

    if img is not None:
        message_en, _ = detect_fn_pt_audio(model, img)
        # message = results_handling(results)
        gTTS(text=message_en, lang="en").save("message.mp3")

        # return jsonify({'status': 'success', 'response time': time.time() - t0, "message": message})

        return Response(generate(), mimetype="audio/mpeg3")
    else:
        return "EMPTY"


# Star Backend
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=2000, debug=True)
