import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import numpy as np
import cv2
import sys

class Classifier:
    def __init__(self, lite_path, classes_path):
        interpreter = tf.lite.Interpreter(model_path=lite_path)
        self.classify_lite = interpreter.get_signature_runner('serving_default')
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        # check the type of the input tensor
        self.floating_model = input_details[0]['dtype'] == np.float32

        classes_f = open(classes_path, mode="r")
        self.classes = []

        line = classes_f.readline()
        while line != "":
            self.classes.append(line)
            line = classes_f.readline().strip()

        classes_f.close()

    def predict(self, img_path):
        img = cv2.imread(img_path)
        img = cv2.resize(img, (224, 224))
        if(self.floating_model):
            img = np.float32(img)

        predictions_lite = self.classify_lite(input_2=np.float32(np.expand_dims(img, 0)))['dense']
        score_lite = tf.nn.softmax(predictions_lite)

        return self.classes[np.argmax(score_lite)], 100 * np.max(score_lite)
    
    def string_predict(self, img_path):
        img = cv2.imread(img_path)
        img = cv2.resize(img, (224, 224))
        if(self.floating_model):
            img = np.float32(img)

        predictions_lite = self.classify_lite(input_2=np.float32(np.expand_dims(img, 0)))['dense']
        score_lite = tf.nn.softmax(predictions_lite)

        recyclable = "not recyclable" if self.classes[np.argmax(score_lite)] == "trash" else "recyclable"

        return "This image most likely belongs to {} with a {:.2f} percent confidence. Anyhow, {}!"\
            .format(self.classes[np.argmax(score_lite)], 100 * np.max(score_lite), recyclable)
    
    def recyclable_predict(self, img_path):
        img = cv2.imread(img_path)
        img = cv2.resize(img, (224, 224))
        if(self.floating_model):
            img = np.float32(img)

        predictions_lite = self.classify_lite(input_2=np.float32(np.expand_dims(img, 0)))['dense']
        score_lite = tf.nn.softmax(predictions_lite)

        return "Not recyclable :(" if self.classes[np.argmax(score_lite)] == "trash" else "Recyclable :)"

    def print_predict(self, img_path):
        img = cv2.imread(img_path)
        img = cv2.resize(img, (224, 224))
        if(self.floating_model):
            img = np.float32(img)

        predictions_lite = self.classify_lite(input_2=np.float32(np.expand_dims(img, 0)))['dense']
        score_lite = tf.nn.softmax(predictions_lite)

        print(
            "This image most likely belongs to {} with a {:.2f} percent confidence."
            .format(self.classes[np.argmax(score_lite)], 100 * np.max(score_lite))
        )

classifier = Classifier("model_96.tflite", "classes.txt")

classifier.print_predict("garbage_classification/shoes/shoes1.jpg")