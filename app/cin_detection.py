import base64
import os
import cv2
import numpy as np
import tensorflow as tf

# Paths to model
MODEL_NAME = 'model'
PATH_TO_CKPT = os.path.join(MODEL_NAME, 'frozen_inference_graph.pb')

# Load the TensorFlow model
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.compat.v1.GraphDef()
    with tf.io.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

sess = tf.compat.v1.Session(graph=detection_graph)
image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
num_detections = detection_graph.get_tensor_by_name('num_detections:0')


def detect_cin(image_base64):
    """
    Detect CIN card in the image and return the cropped CIN card as a Base64-encoded string.
    
    :param image_base64: Base64-encoded input image
    :return: Base64-encoded cropped CIN card
    """
    # Decode the Base64 image
    image_bytes = base64.b64decode(image_base64)
    image_array = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    # Convert image to RGB and expand dimensions for the model
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_expanded = np.expand_dims(image_rgb, axis=0)

    # Perform object detection
    (boxes, scores, _, _) = sess.run(
        [detection_boxes, detection_scores, detection_classes, num_detections],
        feed_dict={image_tensor: image_expanded}
    )

    # Process detection results to find the largest bounding box
    max_area = 0
    largest_box = None
    image_height, image_width, _ = image.shape
    for i in range(int(num_detections[0])):
        if scores[0][i] > 0.6:  # Detection threshold
            box = boxes[0][i]
            ymin, xmin, ymax, xmax = box
            left, right, top, bottom = (
                int(xmin * image_width),
                int(xmax * image_width),
                int(ymin * image_height),
                int(ymax * image_height),
            )
            area = (right - left) * (bottom - top)
            if area > max_area:
                max_area = area
                largest_box = (left, right, top, bottom)

    # Crop the detected CIN card
    if largest_box:
        left, right, top, bottom = largest_box
        cropped_cin = image_rgb[top:bottom, left:right]

        # Convert the cropped image to Base64
        _, buffer = cv2.imencode(".jpg", cropped_cin)
        cropped_cin_base64 = base64.b64encode(buffer).decode("utf-8")
        return cropped_cin_base64
    else:
        raise ValueError("No CIN card detected.")
