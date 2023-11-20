# Script to run custom TFLite model on a single test image to detect objects
# Source: https://github.com/EdjeElectronics/TensorFlow-Lite-Object-Detection-on-Android-and-Raspberry-Pi/blob/master/TFLite_detection_image.py

# Import packages
import cv2
import numpy as np
import glob
import random
from tensorflow.lite.python.interpreter import Interpreter
import matplotlib
import matplotlib.pyplot as plt

### Define function for inferencing with TFLite model and displaying results
def tflite_detect_image(modelpath, imgpath, lblpath, min_conf=0.5):
    # Load the label map into memory
    with open(lblpath, 'r') as f:
        labels = [line.strip() for line in f.readlines()]

    # Load the Tensorflow Lite model into memory
    interpreter = Interpreter(model_path=modelpath)
    interpreter.allocate_tensors()

    # Get model details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    height = input_details[0]['shape'][1]
    width = input_details[0]['shape'][2]

    float_input = (input_details[0]['dtype'] == np.float32)

    # Load and resize the image to the expected shape [1xHxWx3]
    image = cv2.imread(imgpath)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    imH, imW, _ = image.shape
    image_resized = cv2.resize(image_rgb, (width, height))
    input_data = np.expand_dims(image_resized, axis=0)

    # Normalize pixel values if using a floating model (i.e., if the model is non-quantized)
    if float_input:
        input_data = (np.float32(input_data) - 127.5) / 127.5

    # Perform the actual detection by running the model with the image as input
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    # Retrieve detection results
    boxes = interpreter.get_tensor(output_details[1]['index'])[0]  # Bounding box coordinates of detected objects
    classes = interpreter.get_tensor(output_details[3]['index'])[0]  # Class index of detected objects
    scores = interpreter.get_tensor(output_details[0]['index'])[0]  # Confidence of detected objects

    detections = []

    # Loop over all detections and draw detection box if confidence is above the minimum threshold
    for i in range(len(scores)):
        if ((scores[i] > min_conf) and (scores[i] <= 1.0)):
            # Get bounding box coordinates
            ymin = int(max(1, (boxes[i][0] * imH)))
            xmin = int(max(1, (boxes[i][1] * imW)))
            ymax = int(min(imH, (boxes[i][2] * imH)))
            xmax = int(min(imW, (boxes[i][3] * imW)))

            # Crop the original image to the detected region
            cropped_image = image_rgb[ymin:ymax, xmin:xmax]
            detections.append([scores[i], xmin, ymin, xmax, ymax])
     # Show the original image
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(image_rgb)
    plt.title('Original Image')

    # Show the cropped image
    plt.subplot(1, 2, 2)
    plt.imshow(cropped_image)
    plt.title('Cropped Image')
    
    plt.show()
    return cropped_image


imgpath = 'ISIC_0024300.JPG'  # Receive the image
PATH_TO_MODEL = 'detect.tflite'
PATH_TO_LABELS = 'labelmap.txt'
min_conf_threshold = 0.5

cropped_image = tflite_detect_image(PATH_TO_MODEL, imgpath, PATH_TO_LABELS, min_conf_threshold)
# cropped_image is the cropped image based on the detection
