

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

