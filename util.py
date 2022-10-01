import numpy as np
import cv2
import pandas as pd
import os
import keras
import tensorflow as tf
from keras.layers import StringLookup

characters = ['!',
              '"',
              '#',
              '&',
              "'",
              '(',
              ')',
              '*',
              '+',
              ',',
              '-',
              '.',
              '/',
              '0',
              '1',
              '2',
              '3',
              '4',
              '5',
              '6',
              '7',
              '8',
              '9',
              ':',
              ';',
              '?',
              'A',
              'B',
              'C',
              'D',
              'E',
              'F',
              'G',
              'H',
              'I',
              'J',
              'K',
              'L',
              'M',
              'N',
              'O',
              'P',
              'Q',
              'R',
              'S',
              'T',
              'U',
              'V',
              'W',
              'X',
              'Y',
              'Z',
              'a',
              'b',
              'c',
              'd',
              'e',
              'f',
              'g',
              'h',
              'i',
              'j',
              'k',
              'l',
              'm',
              'n',
              'o',
              'p',
              'q',
              'r',
              's',
              't',
              'u',
              'v',
              'w',
              'x',
              'y',
              'z']

max_len = 21
batch_size = 64
padding_token = 99
image_width = 128
image_height = 32

AUTOTUNE = tf.data.AUTOTUNE

# Mapping characters to integers.
char_to_num = StringLookup(vocabulary=list(characters), mask_token=None)

# Mapping integers back to original characters.
num_to_char = StringLookup(
    vocabulary=char_to_num.get_vocabulary(), mask_token=None, invert=True
)


def distortion_free_resize(image, img_size):
    w, h = img_size
    image = tf.image.resize(image, size=(h, w), preserve_aspect_ratio=True)

    # Check tha amount of padding needed to be done.
    pad_height = h - tf.shape(image)[0]
    pad_width = w - tf.shape(image)[1]

    # Only necessary if you want to do same amount of padding on both sides.
    if pad_height % 2 != 0:
        height = pad_height // 2
        pad_height_top = height + 1
        pad_height_bottom = height
    else:
        pad_height_top = pad_height_bottom = pad_height // 2

    if pad_width % 2 != 0:
        width = pad_width // 2
        pad_width_left = width + 1
        pad_width_right = width
    else:
        pad_width_left = pad_width_right = pad_width // 2

    image = tf.pad(
        image,
        paddings=[
            [pad_height_top, pad_height_bottom],
            [pad_width_left, pad_width_right],
            [0, 0],
        ],
    )

    image = tf.transpose(image, perm=[1, 0, 2])
    image = tf.image.flip_left_right(image)
    return image


def preprocess_image(image_path, img_size=(image_width, image_height)):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_png(image, 1)
    image = distortion_free_resize(image, img_size)
    image = tf.cast(image, tf.float32) / 255.0
    return image


def vectorize_label(label):
    label = char_to_num(tf.strings.unicode_split(label, input_encoding="UTF-8"))
    length = tf.shape(label)[0]
    pad_amount = max_len - length
    label = tf.pad(label, paddings=[[0, pad_amount]], constant_values=padding_token)
    return label


def process_images_labels(image_path, label):
    image = preprocess_image(image_path)
    label = vectorize_label(label)
    return {"image": image, "label": label}


def prepare_dataset(image_paths, labels):
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels)).map(
        process_images_labels, num_parallel_calls=AUTOTUNE
    )
    return dataset.batch(batch_size).cache().prefetch(AUTOTUNE)


def prepare_image_for_prediction(image_paths, labels="", bs=1):
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels)).map(
        process_images_labels, num_parallel_calls=AUTOTUNE
    )
    return dataset.batch(bs).cache().prefetch(AUTOTUNE)


def decode_batch_predictions(pred):
    input_len = np.ones(pred.shape[0]) * pred.shape[1]
    # Use greedy search. For complex tasks, you can use beam search.
    results = keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)[0][0][
              :, :max_len
              ]
    # Iterate over the results and get back the text.
    output_text = []
    for res in results:
        res = tf.gather(res, tf.where(tf.math.not_equal(res, -1)))
        res = tf.strings.reduce_join(num_to_char(res)).numpy().decode("utf-8")
        output_text.append(res)
    return output_text


def resize_with_padding(img, width, height):
    new_image_width = width
    new_image_height = height
    color = (0, 0, 0)
    result = np.full((new_image_height, new_image_width, 3), color, dtype=np.uint8)
    old_image_height, old_image_width, channels = img.shape
    x_center = (new_image_width - old_image_width) // 2
    y_center = (new_image_height - old_image_height) // 2
    result[y_center:y_center + old_image_height, x_center:x_center + old_image_width] = img
    return result


def save_cropped_images(img, result, save_dir=""):
    df = result.pandas().xyxy[0]
    df = pd.DataFrame(df)
    df = df.loc[df.groupby(['class'])['confidence'].idxmax()].reset_index(drop=True)
    df = pd.DataFrame(df)
    print(df)

    drug = []
    strength = []
    frequency = []

    if 0 in df.index:
        drug = df.loc[0, ['xmin', 'ymin', 'xmax', 'ymax']].tolist()
        drug = [int(item) for item in drug]

    if 1 in df.index:
        frequency = df.loc[1, ['xmin', 'ymin', 'xmax', 'ymax']].tolist()
        frequency = [int(item) for item in frequency]

    if 2 in df.index:
        strength = df.loc[2, ['xmin', 'ymin', 'xmax', 'ymax']].tolist()
        strength = [int(item) for item in strength]

    # prescription_path = f"C:\\Users\\Yuwin\\PycharmProjects\\ReaderAPI\\static\\{name}\\prescription.jpeg"
    prescription_path = os.path.join(save_dir, "prescription.jpeg")
    cv2.imwrite(prescription_path, img)

    if len(drug) == 4:
        img_cropped = img[drug[1]:drug[3], drug[0]:drug[2]]
        # drug_path = f"C:\\Users\\Yuwin\\PycharmProjects\\ReaderAPI\\static\\{name}\\drug.jpeg"
        drug_path = os.path.join(save_dir, "drug.jpeg")
        cv2.imwrite(drug_path, img_cropped)

    if len(strength) == 4:
        img_cropped = img[strength[1]:strength[3], strength[0]:strength[2]]
        # strength_path = f"C:\\Users\\Yuwin\\PycharmProjects\\ReaderAPI\\static\\{name}\\strength.jpeg"
        strength_path = os.path.join(save_dir, "strength.jpeg")
        cv2.imwrite(strength_path, img_cropped)

    if len(frequency) == 4:
        img_cropped = img[frequency[1]:frequency[3], frequency[0]:frequency[2]]
        # frequency_path = f"C:\\Users\\Yuwin\\PycharmProjects\\ReaderAPI\\static\\{name}\\frequency.jpeg"
        frequency_path = os.path.join(save_dir, "frequency.jpeg")
        cv2.imwrite(frequency_path, img_cropped)


def image_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation=inter)

    # return the resized image
    return resized


def readDrugDict():
    drugs = []
    with open("./drug_dictionary", "r") as f:
        lines = f.readlines()
        for line in lines:
            drugs.append(line.rstrip())
    return drugs
