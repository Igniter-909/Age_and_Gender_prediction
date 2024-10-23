from flask import Flask, render_template, Response, jsonify
import cv2
import numpy as np
from src.predict_pipeline.predict_pipeline import predictPipeline
import io
from PIL import Image

app = Flask(__name__)

# Initialize the video capture and the prediction pipeline
cap = cv2.VideoCapture(0)
pipeline = predictPipeline()

# Get the frame size dynamically from the video capture
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define the bounding box dimensions (centered box)
box_w, box_h = 250, 300  # You can change these values
rect_x = int((frame_width - box_w) / 2)
rect_y = int((frame_height - box_h) / 3)

def gen_frames():
    """Generate camera frames and show the bounding box."""
    global frame
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            # Draw the rectangular bounding box on the frame (centered)
            cv2.rectangle(frame, (rect_x, rect_y), (rect_x + box_w, rect_y + box_h), (0, 255, 0), 2)
            
            # Convert the frame to JPEG
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            # Return the frame as bytes
            yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    """Home page rendering the index.html."""
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """Video streaming route."""
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/predictions')
def predictions():
    """Route to return the predicted gender and age."""
    global frame

    if frame is None:
        return jsonify(pred_gender= 0,pred_age=0)
    # Decode the frame from the global `frame` variable
    img_array = np.frombuffer(frame, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    # Crop the image based on the bounding box (rect_x, rect_y, box_w, box_h)
    cropped_img = img[rect_y:rect_y + box_h, rect_x:rect_x + box_w]

    # Convert the cropped image to grayscale
    pil_img = Image.fromarray(cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY))
    
    # Save the cropped image into a byte stream
    img_byte_arr = io.BytesIO()
    pil_img.save(img_byte_arr, format='JPEG')
    img_byte_arr.seek(0)

    # Call the prediction pipeline with the cropped image
    pred_gender, pred_age = pipeline.predict([img_byte_arr])  # Pass as list to match the predict function

    # Return the predictions as a JSON response
    return jsonify(pred_gender=pred_gender, pred_age=pred_age)

if __name__ == '__main__':
    app.run(debug=True)
    cap.release()
    cv2.destroyAllWindows()
