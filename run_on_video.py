
# import keras and tensorflow
import keras
import tensorflow as tf

# import keras_retinanet
from keras_retinanet import models
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from keras_retinanet.utils.visualization import draw_box, draw_caption
from keras_retinanet.utils.colors import label_color

# import miscellaneous modules
import matplotlib.pyplot as plt
import cv2
import os
import numpy as np
import time
import argparse, json

def get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)

def detect_objects(image_np, model):
    image_np_preprocess, scale = resize_image(image_np)
    image_np_preprocess = preprocess_image(image_np_preprocess)

    image_np_preprocess = np.expand_dims(image_np_preprocess, axis=0)
    boxes, scores, labels = model.predict_on_batch(image_np_preprocess)
    boxes /= scale

    return [boxes[0], scores[0], labels[0]]

def draw_on_image(image_np, img_boxes, img_scores, img_labels):
    for box, score, label in zip(img_boxes, img_scores, img_labels):
        # scores are sorted so we can break
        if score < 0.5:           
            break
        # copy to draw on
        color = label_color(label)

        b = box.astype(int)
        draw_box(image_np, b, color=color)

        caption = "{} {:.3f}".format(labels_to_names[label], score)
        draw_caption(image_np, b, caption)
    return image_np
        
    
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-vid', '--video_path', dest='video_path', type=str,
                            help='Path to input video')
    parser.add_argument('-fps', '--fps', dest='fps', type=int,
                        default=0, help='number of frames to process every second')
    parser.add_argument('-convert', '--convert', dest='convert', type=bool,
                    default=False, help='weither to convert the model to inference model or not')
    parser.add_argument('-mdl', '--model', dest='model_path', type=str,
                    default=os.path.join('..', 'training', 'inference_model_17.h5'), help='path to inference model')
    parser.add_argument('-out', '--output', dest='output', type=str,
                default=None, help='path to output video')
    parser.add_argument('-json', '--json_labels', dest='json_labels', type=str,
                default='hand.json', help='path to json file for label mapping')
    args = parser.parse_args()
    
    labels_to_names = {0:'hand'}
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    keras.backend.tensorflow_backend.set_session(get_session())
    model = models.load_model(args.model_path, backbone_name='resnet50', convert=args.convert)
    

    capture = cv2.VideoCapture(args.video_path)    
    video_fps = capture.get(cv2.CAP_PROP_FPS)

    if args.output != None:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(args.output,
                            fourcc, video_fps,
                            (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
                            int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))))
    
    if args.fps == 0:
        args.fps = video_fps

    frame_count, prediction_fps, stime = 0, 0, time.time()
  
    while True: 
        ret, frame = capture.read()

        if not ret:
            break

        detections = detect_objects(frame, model)
        
        draw_on_image(frame, *detections)
        cv2.imshow('hands', frame)

        if args.output != None:
            out.write(frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_count += 1
        
        prediction_fps += 1
        if  (time.time() - stime) >= 1:
            print(f'fps = {prediction_fps/(time.time() - stime):2.02f}', end='\r')
            prediction_fps, stime =  0, time.time()

print('number of processed frames = ', frame_count)
