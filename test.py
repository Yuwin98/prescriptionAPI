import tensorflow as tf
from util import image_resize
import cv2
import numpy as np
import os
import torch
import pandas as pd


def test_detection():
    global drug
    model = torch.hub.load('./yolov5', "custom", path="./yolov5/best.pt", source='local')
    model.conf = 0.25
    model.iou = 0.9

    image = cv2.imread(r"C:\Users\Yuwin\PycharmProjects\ReaderAPI\upload\img_3a192291-2e57-11ed-b292-2016b9895a04.jpeg")
    img_1 = image_resize(image, width=432, height=72)
    result = model(img_1)
    df = result.pandas().xyxy[0]
    df = pd.DataFrame(df)
    df = df.loc[df.groupby(['class'])['confidence'].idxmax()].reset_index(drop=True)
    df = pd.DataFrame(df)

    # if 0 in df.index:
    #     drug = df.loc[0, ['xmin', 'ymin', 'xmax', 'ymax']].tolist()
    #     drug = [int(item) for item in drug]
    #
    # if 1 in df.index:
    #     frequency = df.loc[1, ['xmin', 'ymin', 'xmax', 'ymax']].tolist()
    #     frequency = [int(item) for item in frequency]
    #
    # if 2 in df.index:
    #     strength = df.loc[2, ['xmin', 'ymin', 'xmax', 'ymax']].tolist()
    #     strength = [int(item) for item in strength]


# test_detection()


def test_frequency():
    path = r"C:\Users\Yuwin\Desktop\Freq\a-data\data"
    dirs = os.listdir(path)
    frequency_model = tf.keras.models.load_model("./models/frequency.h5")
    img = cv2.imread(r"C:\Users\Yuwin\PycharmProjects\ReaderAPI\static\freq.jpeg")
    img = tf.expand_dims(img, axis=0)
    img = tf.image.resize(img, size=(256, 256))
    result = frequency_model.predict(img).flatten().tolist()
    result = np.argmax(result)
    print(dirs[result])


# test_frequency()


def test_strength():
    path = r"C:\Users\Yuwin\Desktop\Strength\data"
    dirs = os.listdir(path)
    strength_model = tf.keras.models.load_model("./models/strength.h5")
    img = cv2.imread(r"C:\Users\Yuwin\PycharmProjects\ReaderAPI\static\strength.jpeg")
    img = tf.expand_dims(img, axis=0)
    img = tf.image.resize(img, size=(128, 128))
    result = strength_model.predict(img).flatten().tolist()
    result = np.argmax(result)
    print(dirs[result])

# test_strength()


def test_drug():
    frequency_model = tf.keras.models.load_model("./models/lstm-weights-epoch24-val_loss0.242.h5")
    img = cv2.imread(r"E:\Research\Freq\data\bd\36.jpeg")
    img = tf.expand_dims(img, axis=0)
    img = tf.image.resize(img, size=(64, 128))
    result = frequency_model.predict(img).flatten().tolist()
    print(result)


test_drug()
