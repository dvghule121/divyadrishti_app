import numpy as np
import cv2
import os
import time
from gtts import gTTS

# Define the arguments
args = {
    "video": "test.mp4",
    "yolo": "yolo",
    "confidence": 0.5,
    "threshold": 0.3
}

# Define global variables
start_time = time.time()

def load_yolo(yolo_path):
    labels_path = os.path.sep.join([yolo_path, "coco.names"])
    labels = open(labels_path).read().strip().split("\n")

    weights_path = os.path.sep.join([yolo_path, "yolov3.weights"])
    config_path = os.path.sep.join([yolo_path, "yolov3.cfg"])

    net = cv2.dnn.readNetFromDarknet(config_path, weights_path)

    if net is None:
        print("[ERROR] Failed to load YOLO model.")
        exit()

    layer_names = net.getLayerNames()
    output_layers_indices = net.getUnconnectedOutLayers()
    output_layers = [layer_names[i - 1] for i in output_layers_indices]

    return net, output_layers, labels

def process_frame(frame, net, output_layers, confidence_threshold):
    (H, W) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    layer_outputs = net.forward(output_layers)

    return (H, W), layer_outputs

def detect_objects(layer_outputs, W, H, confidence_threshold):
    boxes = []
    confidences = []
    class_ids = []

    for output in layer_outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > confidence_threshold:
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    return boxes, confidences, class_ids

def generate_description(boxes, class_ids, labels, W, H):
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

def write_output(frame, output_path, out, frame_count, fps, prev_description, net, output_layers, labels):
    global start_time

    out.write(frame)
    description = prev_description

    if frame_count % fps == 0:  # Process one frame per second
        (H, W), layer_outputs = process_frame(frame, net, output_layers, args["confidence"])
        boxes, confidences, class_ids = detect_objects(layer_outputs, W, H, args["confidence"])
        description = generate_description(boxes, class_ids, labels, W, H)

        if description != prev_description:
            myobj = gTTS(text=description, lang="en", slow=False)
            myobj.save(f"object_detection_frame{frame_count}.mp3")
            print(description)

    frame_count += 1
    if frame_count % 30 == 0:  # Calculate FPS every 30 frames
        end_time = time.time()
        elapsed_time = end_time - start_time
        fps = frame_count / elapsed_time
        print("FPS:", fps)
        start_time = time.time()
        frame_count = 0

    return frame_count, description

def main():
    net, output_layers, labels = load_yolo(args["yolo"])
    video = cv2.VideoCapture(args["video"])

    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(video.get(cv2.CAP_PROP_FPS))

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output.avi', fourcc, fps, (width, height))

    frame_count = 0
    prev_description = None

    while True:
        ret, frame = video.read()
        if not ret:
            break

        frame_count, prev_description = write_output(frame, 'output.avi', out, frame_count, fps, prev_description, net, output_layers, labels)

    video.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
