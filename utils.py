import os
import cv2
import uuid
import pickle

import numpy as np
import pandas as pd
import tensorflow as tf

from PIL import ImageDraw, Image
from tqdm import tqdm

category_index = pickle.load(
    open("/home/dumbmachine/code/SVMWSN/.data/category_index.pkl", "rb"))


def create_xml(images, model=None, train=True):
    """WIll create the xml annotation for the image passed

    Arguments:
        images {list} -- list of images
        model {tf.model} -- the model used to create prediction
    """

    if model is None:
        model = load_model()

    output_dicts = []
    images = np.array(images)
    input_tensor = tf.convert_to_tensor(images)
    output_dict = model(input_tensor)
    detections = output_dict.pop("num_detections").numpy()

    for i in range(len(images)):
        temp_dict = {}
        for key in output_dict.keys():
            temp_dict[key] = output_dict[key][i]

        # getting rid of the trash predictions
        num_detections = detections[i]
        temp_dict = {key: value[:int(num_detections)].numpy()
                     for key, value in temp_dict.items()}
        temp_dict['num_detections'] = num_detections
        temp_dict['detection_classes'] = temp_dict['detection_classes'].astype(
            np.int64)

        # sending the predictions for saving
        drawmage = save_bboxes(
            image=images[i],
            output_dict=temp_dict,
            train=train
        )
    return model


def load_model():
    """
    Loading a simple Tensorflow ObjectDetction Api model if the user doesn't supply a model
    """
    model_name = "ssd_mobilenet_v1_coco_2018_01_28"
    path = os.path.expanduser("~")
    base_url = 'http://download.tensorflow.org/models/object_detection/'
    model_file = model_name + '.tar.gz'
    model_dir = tf.keras.utils.get_file(
        fname=model_name,
        origin=base_url + model_file,
        untar=True)

    model_dir = os.path.join(model_dir, "saved_model")

    model = tf.saved_model.load(str(model_dir))
    model = model.signatures['serving_default']

    return model


def create_screenshots_from_video(video_path, nos_frames=1, verbose=1):
    """Will create the dataset from the video frames present in the video clip

    Arguments:
        video_path {str} -- filepath of the video clip

    Keyword Arguments:
        nos_frames {int} -- The number of franos_framesmes to be taken from each interval (default: {2})
        fps {int} -- requried to distributre the video frames properly and avoid redundant frames (default: {24})
    """
    # creating the variables to take care of the video
    breakpoint_loop = 500
    start = 0
    frames = []
    model = None
    video = cv2.VideoCapture(video_path)
    fps = int(video.get(cv2.CAP_PROP_FPS))
    end = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    progress = tqdm(total=int(end/fps))

    for _ in range(0, end, fps):
        progress.set_description(f"Batch number {int(_/fps)}")
        frames = [video.read()[1] for _ in range(fps)]
        idxs = [int(i) for i in np.random.uniform(0, fps, nos_frames)]
        images = [frames[i] for i in idxs]
        model = make_prediction(images, model)

        progress.update(1)
        if breakpoint_loop <= 0:
            break
        else:
            breakpoint_loop -= 1


def make_prediction(images, model=None, train=True):
    """Generate the predictions

    Arguments:
        images {list} -- list of images

    Keyword Arguments:
        model {[type]} --  The model in serving mode (default: {None})
    """
    if model is None:
        model = load_model()

    output_dicts = []
    images = np.array(images)
    input_tensor = tf.convert_to_tensor(images)
    output_dict = model(input_tensor)
    detections = output_dict.pop("num_detections").numpy()

    for i in range(len(images)):
        temp_dict = {}
        for key in output_dict.keys():
            temp_dict[key] = output_dict[key][i]

        # getting rid of the trash predictions
        num_detections = detections[i]
        temp_dict = {key: value[:int(num_detections)].numpy()
                     for key, value in temp_dict.items()}
        temp_dict['num_detections'] = num_detections
        temp_dict['detection_classes'] = temp_dict['detection_classes'].astype(
            np.int64)

        # sending the predictions for saving
        drawmage = save_bboxes(
            image=images[i],
            output_dict=temp_dict,
        )
    return model


def save_bboxes(image, output_dict, draw=False):
    """Will save the image and the bboxes

    Arguments:
        image {[type]} -- [description]
        output_dict {[type]} -- [description]

    Keyword Arguments:
        draw {bool} -- [description] (default: {False})
    """
    # the information to be saved here
    filename = str(uuid.uuid4())
    xml_path = f"datasets/humans/annotations/{filename}.xml"
    image_path  = f"datasets/humans/images/{filename}.jpg"

    ret = []
    image = cv2.resize(image, (800, 600))

    copy_image = image.copy()
    copy_image = Image.fromarray(copy_image)
    draw = ImageDraw.Draw(copy_image)
    im_width, im_height = copy_image.size

    for cat, bbox, score in zip(
            output_dict['detection_classes'],
            output_dict['detection_boxes'],
            output_dict['detection_scores']):
        # For the test run we will only save the person
        if score > 0.45 and cat == 1:

            ymin, xmin, ymax, xmax = bbox
            (left, right, top, bottom) = (xmin * im_width, xmax * im_width,
                                          ymin * im_height, ymax * im_height)


            if draw:
                draw.line([(left, top), (left, bottom), (right, bottom),
                           (right, top), (left, top)], width=4, fill="red")

            template = open("template", "r").read()
            template = template.replace("CLASS", "person")
            template = template.replace("WIDTH", str(600))
            template = template.replace("HEIGHT",str(800))
            template = template.replace("FILENAME", filename)
            template = template.replace("XMIN", str(left))
            template = template.replace("XMAX", str(right))
            template = template.replace("YMIN", str(top))
            template = template.replace("YMAX", str(bottom))

            open(xml_path, "w").write(template)
            cv2.imwrite(image_path, image)

            # Taking only one object from each image
            break


"""



images = glob("/home/dumbmachine/demos/detectron2-humans/datasets/humans/images/*")

image = random.choice(images)
print(image)

xmax=471.4616298675537
xmin=337.357759475708
ymax=540.5477285385132
ymin=387.92381286621094

(left, right, top, bottom) = (xmin , xmax ,
                                ymin , ymax )

image = Image.open("/home/dumbmachine/demos/detectron2-humans/datasets/humans/images/3a7e668c-0ff5-435d-8493-b42d86ce8dfc.jpg")

copy_image = image.copy()
draw = ImageDraw.Draw(copy_image)
im_width, im_height = copy_image.size


draw.line([(left, top), (left, bottom), (right, bottom),
            (right, top), (left, top)], width=4, fill="red")
copy_image.show()
"""