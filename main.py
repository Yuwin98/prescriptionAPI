from flask import Flask, jsonify, request
import os
import cv2
import torch
from util import image_resize, save_cropped_images, prepare_image_for_prediction, decode_batch_predictions
import uuid
import tensorflow as tf
import numpy as np
import keras
from pymongo import MongoClient
from correction import correction
import difflib
import json

UPLOAD_FOLDER = './upload'
STATIC_FOLDER = './static'
ALLOWED_EXTENSIONS = {'jpg', 'jpeg'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['STATIC_FOLDER'] = STATIC_FOLDER

frequency_labels = ["BD", "Daily", "Mane", "Nocte", "SOS", "TDS"]

strength_labels = ["0.4mg", "0.5tbs", "0.25tbs", "1", "1.5mg", "1g", "1mg", "1tab", "2.5mg", "2.5ml", "2tsp", "5mg",
                   "5ml", "10mg", "12.5mg", "20mg", "20ml", "25mg", "30mg", "40mg", "50mg", "60mg", "62.mg", "75mg",
                   "80mg", "100mg", "120mg", "150mg", "160mg", "180mg", "200mg", "250mg", "300mg", "375mg", "400mg",
                   "500mg", "600mg", "625mg", "750mg", "LA"]

DRUGLIST = list(open('./drug_dictionary').read().splitlines())

cluster = MongoClient("mongodb+srv://Yms98:Alphagolf212@drugs.ry1tiin.mongodb.net/?retryWrites=true&w=majority")
db = cluster["Drugs"]
collection = db["Drug"]

with open('data.json') as json_file:
    drug_data = dict(json.load(json_file))


@app.route('/api/v1/detect/<size>', methods=['POST'])
def detectroi(size):
    if int(size) > 0:
        data = []
        model = torch.hub.load('./yolov5', "custom", path="./yolov5/best.pt", source='local')
        model.conf = 0.25
        model.iou = 0.9
        for i in range(int(size)):
            image = request.files.get(f'file[{i}]')
            ext = ".jpeg"
            name = f"img_{uuid.uuid1()}"
            filename = name + ext
            path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            image.save(path)

            img = cv2.imread(path)
            img = image_resize(img, width=432, height=72)

            result = model(img)
            save_dir = os.path.join(STATIC_FOLDER, name)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            save_cropped_images(img, result, save_dir)

            prescription_image = os.path.join(save_dir, "prescription.jpeg")
            drug_image = os.path.join(save_dir, "drug.jpeg")
            frequency_image = os.path.join(save_dir, "frequency.jpeg")
            strength_image = os.path.join(save_dir, "strength.jpeg")

            if os.path.exists(drug_image):
                drug_url = drug_image
            else:
                drug_url = "Not Detected"

            if os.path.exists(strength_image):
                strength_url = strength_image
            else:
                strength_url = "Not Detected"

            if os.path.exists(frequency_image):
                frequency_url = frequency_image
            else:
                frequency_url = "Not Detected"

            prescription = {
                "prescription": prescription_image,
                "drug": drug_url,
                "frequency": frequency_url,
                "strength": strength_url
            }

            data.append(prescription)

        return jsonify({
            'data': data,
            'success': True
        })


@app.route("/api/v1/recognize", methods=["POST"])
def recognize():
    frequency_model = tf.keras.models.load_model("./models/frequency.h5")
    trained_model_strength = keras.models.load_model("./models/drugs_v4.h5")
    trained_model_drug = keras.models.load_model("./models/drugs_v2.h5")

    predictor_strength = keras.models.Model(
        trained_model_strength.get_layer(name="image").input, trained_model_strength.get_layer(name="dense2").output
    )
    predictor_drug = keras.models.Model(
        trained_model_drug.get_layer(name="image").input, trained_model_drug.get_layer(name="dense2").output
    )

    prescriptionList = request.get_json(silent=True)['data']
    data = []
    for prescription in prescriptionList:
        if prescription["drugImg"] != "Not Detected":
            drugImg = prepare_image_for_prediction([prescription["drugImg"]], [""])
        else:
            drugImg = None
        if prescription["strengthImg"] != "Not Detected":
            strengthImg = prepare_image_for_prediction([prescription['strengthImg']], [""])
        else:
            strengthImg = None
        if prescription["frequencyImg"] != "Not Detected":
            frequencyImg = cv2.imread(prescription['frequencyImg'])
        else:
            frequencyImg = None

        prescriptionUrl = prescription['prescriptionImg']

        if drugImg is not None:
            predictions = predictor_drug.predict(drugImg)
            pred_drug = decode_batch_predictions(predictions)[0]
            pred_drug = str(pred_drug).replace("[UNK]", "")
            print(f"Before: {pred_drug}")
            print(f"Corrected: {correction(pred_drug)}")
            close_pred = difflib.get_close_matches(pred_drug, DRUGLIST)
            if len(close_pred) > 0:
                pred_drug = close_pred[0]
            else:
                pred_drug = pred_drug
            print(f"After: {pred_drug}")
        else:
            pred_drug = ""

        if strengthImg is not None:
            strength_img = predictor_strength.predict(strengthImg)
            strength_label = decode_batch_predictions(strength_img)[0]
        else:
            strength_label = "N/A"

        if frequencyImg is not None:
            frequency_resized = tf.expand_dims(tf.image.resize(frequencyImg, size=(256, 256)), axis=0)
            frequency_label_prediction = frequency_model.predict(frequency_resized)
            frequency_index = np.argmax(frequency_label_prediction.flatten().tolist())
            frequency_label = frequency_labels[frequency_index]
        else:
            frequency_label = "N/A"

        pred_Drug_lower = str(pred_drug).lower()
        pred_drug_data = drug_data.get(pred_Drug_lower, dict())

        prescription_analysis = {
            "prescription": prescriptionUrl,
            "drug": str(pred_drug).upper(),
            "frequency": frequency_label,
            "strength": strength_label,
            "shortDescription": pred_drug_data.get("description", ""),
            "uses": pred_drug_data.get("uses", []),
            "warnings": pred_drug_data.get("warnings", [])
        }
        data.append(prescription_analysis)

    return jsonify({
        'data': data,
        'success': True
    })


@app.route("/api/v1/recognize-drug/<size>", methods=["POST"])
def recognizeDrug(size):
    if int(size) > 0:

        trained_model_drug = keras.models.load_model("./models/drugs_v2.h5")

        predictor_drug = keras.models.Model(
            trained_model_drug.get_layer(name="image").input, trained_model_drug.get_layer(name="dense2").output
        )

        data = []

        for i in range(int(size)):
            image = request.files.get(f'file[{i}]')
            ext = ".jpeg"
            name = f"img_{uuid.uuid1()}"
            filename = name + ext
            path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            image.save(path)

            drugImg = prepare_image_for_prediction([path], [""])

            drugUrl = path

            if drugImg is not None:
                predictions = predictor_drug.predict(drugImg)
                pred_drug = decode_batch_predictions(predictions)[0]
                pred_drug = str(pred_drug).replace("[UNK]", "")
                print(f"Before: {pred_drug}")
                print(f"Corrected: {correction(pred_drug)}")
                close_pred = difflib.get_close_matches(pred_drug, DRUGLIST)
                if len(close_pred) > 0:
                    pred_drug = close_pred[0] + " or " + correction(pred_drug)
                else:
                    pred_drug = correction(pred_drug)
                print(f"After: {pred_drug}")
            else:
                pred_drug = ""

            pred_Drug_lower = str(pred_drug).lower()
            pred_drug_data = drug_data.get(pred_Drug_lower, dict())

            prescription_analysis = {
                "prescription": drugUrl,
                "drug": str(pred_drug).upper(),
                "shortDescription": pred_drug_data.get("description", ""),
                "uses": pred_drug_data.get("uses", []),
                "warnings": pred_drug_data.get("warnings", [])
            }
            data.append(prescription_analysis)

        return jsonify({
            'data': data,
            'success': True
        })


@app.route("/", methods=["GET"])
def index():
    return jsonify({
        'status': 'OK2'
    })


if __name__ == '__main__':
    if not os.path.exists("./upload"):
        os.mkdir("./upload")

    if not os.path.exists("./static"):
        os.mkdir("./static")

    app.run(debug=True)
