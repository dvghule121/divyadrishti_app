import cv2
from flask import Flask, Response, request, jsonify, render_template
import requests
import numpy as np
import base64
import json

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/send_frame', methods=['POST'])
def receive_frame():
    data = request.get_json()
    frame_base64 = data.get('frame')
    frame_bytes = base64.b64decode(frame_base64)
    response = send_frame(frame_bytes)
    # Process the frame here
    return jsonify({'message': response.get('description')})


# Function to send a frame to the Flask server
def send_frame(frame):
    _, img_encoded = cv2.imencode('.jpg', frame)
    response = requests.post('http://127.0.0.1:8000/send_frame', files={'frame': img_encoded.tobytes()})
    print("Response status code:", response.status_code)
    print("Response content:", response.content)
    return response

# Function to capture video from the webcam
def capture_video():
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Send frame to the Flask server
        response = send_frame(frame)
        if response is None:
            print("Error: Received empty response from server.")
            continue

        if response.status_code == 200:
            try:
                # Decode the JSON response
                response_data = response.json()
                frame_base64 = response_data.get('frame')
                description = response_data.get('description')

                # Decode the frame from Base64
                frame_bytes = base64.b64decode(frame_base64)
                decoded_frame = cv2.imdecode(np.frombuffer(frame_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)

                # Display the frame with bounding boxes and labels
                cv2.imshow('Frame with Bounding Boxes', decoded_frame)
                print("Object Detection Results:", description)
            except json.JSONDecodeError as e:
                print("Error: Failed to decode JSON response.")
                print(e)
        else:
            print("Error: Received status code", response.status_code)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    capture_video()
    # app.run()