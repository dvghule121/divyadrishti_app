import cv2
import numpy as np
import base64
from flask import Flask, Response, request, jsonify
import os
import time

app = Flask(__name__)

# Define the YOLO model and labels
yolo_path = "yolo"
labels_path = os.path.sep.join([yolo_path, "coco.names"])
labels = open(labels_path).read().strip().split("\n")

weights_path = os.path.sep.join([yolo_path, "yolov3.weights"])
config_path = os.path.sep.join([yolo_path, "yolov3.cfg"])

net = cv2.dnn.readNetFromDarknet(config_path, weights_path)
layer_names = net.getLayerNames()
output_layers_indices = net.getUnconnectedOutLayers()
output_layers = [layer_names[i - 1] for i in output_layers_indices]

# Global variables
start_time = time.time()
prev_description = None


def detect_objects(frame):
    (H, W) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    layer_outputs = net.forward(output_layers)

    boxes = []
    confidences = []
    class_ids = []

    for output in layer_outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.8:  # Adjust confidence threshold if needed
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    return boxes, confidences, class_ids

def generate_description(boxes, class_ids, W, H):
    descriptions = []

    for box, class_id in zip(boxes, class_ids):
        (x, y, w, h) = box
        centerX = round((2 * x + w) / 2)
        centerY = round((2 * y + h) / 2)

        if centerX <= W / 3:
            W_pos = "left "
        elif centerX <= (W / 3 * 2):
            W_pos = "center "
        else:
            W_pos = "right "

        if centerY <= H / 3:
            H_pos = "top "
        elif centerY <= (H / 3 * 2):
            H_pos = "mid "
        else:
            H_pos = "bottom "

        descriptions.append(H_pos + W_pos + labels[class_id])

    return ', '.join(descriptions)


def write_output(frame):
    global start_time, prev_description

    (H, W) = frame.shape[:2]
    boxes, confidences, class_ids = detect_objects(frame)

    objects_info = []

    for box, class_id in zip(boxes, class_ids):
        (x, y, w, h) = box
        centerX = round((2 * x + w) / 2)
        centerY = round((2 * y + h) / 2)

        if centerX <= W / 3:
            W_pos = "left"
        elif centerX <= (W / 3 * 2):
            W_pos = "center"
        else:
            W_pos = "right"

        if centerY <= H / 3:
            H_pos = "top"
        elif centerY <= (H / 3 * 2):
            H_pos = "mid"
        else:
            H_pos = "bottom"

        objects_info.append({
            "label": labels[class_id],
            "position": f"{H_pos}-{W_pos}",
            "bbox": [x, y, w, h]  # Bounding box coordinates
        })

    return objects_info




@app.route('/send_frame', methods=['POST'])
def receive_frame():
    try:
        # Log that a frame is being received
        print("Received a frame")

        frame = request.files['frame'].read()

        # Check if the received frame is in JPEG format
        if frame[:2] != b'\xff\xd8':
            print("Received frame is not in JPEG format.")
            return jsonify({"message": "Received frame is not in JPEG format."}), 400

        nparr = np.frombuffer(frame, np.uint8)

        # Specify the desired color scheme (e.g., RGB)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)

        # Check if the image decoding was successful
        if img is None:
            print("Failed to decode the image.")
            return jsonify({"message": "Failed to decode the image."}), 500

        # Convert the color scheme if necessary (e.g., from BGR to RGB)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        objects_info = write_output(img_rgb)

        if objects_info:
            # Log the detected objects
            print("Detected objects:", objects_info)

            return jsonify({"objects": objects_info})
        else:
            # Log that no object was detected
            print("No object detected")

            return jsonify({"message": "No object detected"}), 204  # No content response
    except Exception as e:
        # Log any exceptions that occur
        print("An error occurred:", e)
        return jsonify({"message": "An error occurred"}), 500  # Internal server error


@app.route('/video_feed')
def video_feed():
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(debug=True, port=8000, host="0.0.0.0")