import cv2
import tensorflow as tf
import numpy as np
from flask import Flask, render_template, Response

app = Flask(__name__)
# Load the Teachable Machine TensorFlow model
model = tf.keras.models.load_model('keras_model1.h5')

# Load the label file
with open('labels1.txt', 'r') as f:
    labels = f.read().splitlines()

# Initialize the video capture
cap = cv2.VideoCapture(0)

# Set the desired dimensions for the zoomed-out frame
zoomed_out_width = 400
zoomed_out_height = 300


def generate_frames():
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Preprocess the frame
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224))
        img = img / 255.0
        img = np.expand_dims(img, axis=0)

        # Perform the prediction
        predictions = model.predict(img)
        prediction_label = labels[np.argmax(predictions)]

        # Create separate regions for displaying frames
        height, width, _ = frame.shape
        box_height = height // 2
        box_width = width // 2

        # Display the zoomed-out frame in the top left quadrant
        zoomed_out_frame = cv2.resize(frame, (zoomed_out_width, zoomed_out_height))
        frame[:zoomed_out_height, :zoomed_out_width] = zoomed_out_frame

        # Display top right text box
        # Calculate the maximum width and height for the top right text box
        max_text_width = width - box_width - 20
        max_text_height = box_height - 20

        # Limit the width and height of the text box
        text_box_width = min(max_text_width,
                             cv2.getTextSize(prediction_label, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)[0][0] + 20)
        text_box_height = min(max_text_height,
                              cv2.getTextSize(prediction_label, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)[0][1] + 20)

        # Adjust the coordinates of the top right text box
        text_box_top_left = (box_width, 0)
        text_box_bottom_right = (box_width + text_box_width, text_box_height)

        # Display the top right text box
        cv2.rectangle(frame, text_box_top_left, text_box_bottom_right, (0, 0, 0), cv2.FILLED)
        cv2.putText(frame, prediction_label, (box_width + 10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                    (255, 255, 255), 2)

        # Adjust the coordinates of the top right quadrant
        top_right_quadrant_top_left = (box_width, 0)
        top_right_quadrant_bottom_right = (width, box_height)

        # Fill the background of the top right quadrant with (50, 50, 50) RGB color
        cv2.rectangle(frame, top_right_quadrant_top_left, top_right_quadrant_bottom_right, (50, 50, 50), cv2.FILLED)

        # Display the prediction label in the top right text box
        cv2.putText(frame, prediction_label, (box_width + 10, box_height // 2), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                    (255, 255, 255), 2)



        if prediction_label == "Pushup Form Good":
            top_text = "Your push ups have good form and aren't harming you body."
        elif prediction_label == "Pushup Form OK":
            top_text = "You push ups could have better form. To get better form try holding you body striaght as you go up and down."
        else:
            top_text = "Your push ups are damaging to your body watch this video:https://www.youtube.com/watch?v=IODxDxX7oi4&ab_channel=Calisthenicmovement"
        top_text = text_wrap(top_text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, box_width - 20)
        text_height = box_height // 2 + 20
        for line in top_text:
            text_x = box_width +10
            cv2.putText(frame, line, (text_x, text_height), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            text_height += 18
            # Display bottom left text box with wrapped text and black background
        bottom_left_text = """Harder variation: See gains faster by doing the harder variation:Decline Push-ups: Use an elevated surface like a box or bench. Place feet on the surface, hands on the ground. Keep body straight. Bend elbows to lower chest towards the ground, then push back up. Increasing height challenges upper body strength.
Spiderman Push-ups: Start in a regular push-up position. As you lower, bring one knee to the corresponding elbow on the same side. Push back up and repeat on the other side. Adds rotation and core challenge.
"""
        bottom_left_text_lines = text_wrap(bottom_left_text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, box_width - 20)
        text_height = box_height + 40 - 20  # Adjusted text_height
        cv2.rectangle(frame, (0, box_height), (box_width, height), (50, 50, 50), cv2.FILLED)
        for line in bottom_left_text_lines:
            text_width = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0][0]
            text_x = 10
            cv2.putText(frame, line, (text_x, text_height), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            text_height += 18  # Adjusted line spacing

        # Display bottom right text box with wrapped text and black background
        bottom_right_text = """Easier variation: If you can't consistently do a lot of normal reps, do more reps of the easier variation.
Incline Push-ups: Use an elevated surface, like a bench or table. Place hands slightly wider than shoulder-width apart. Angle your body with feet on the ground. Bend elbows to lower chest towards surface, then push back up. Decrease angle over time for more difficulty.
"""
        bottom_right_text_lines = text_wrap(bottom_right_text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, width - box_width - 20)
        text_height = box_height + 40 - 20  # Adjusted text_height
        cv2.rectangle(frame, (box_width, box_height), (width, height), (50, 50, 50), cv2.FILLED)
        for line in bottom_right_text_lines:
            text_width = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0][0]
            text_x = box_width + 10
            cv2.putText(frame, line, (text_x, text_height), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            text_height += 18  # Adjusted line spacing

        # Add orange grid between quadrants
        cv2.line(frame, (box_width, 0), (box_width, height), (0, 165, 255), 2)
        cv2.line(frame, (0, box_height), (width, box_height), (0, 165, 255), 2)

        # Encode the frame as JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        # Yield the frame in byte format
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


def text_wrap(text, font, font_scale, max_width):
    lines = []
    words = text.split()
    current_line = words[0]
    for word in words[1:]:
        if cv2.getTextSize(current_line + ' ' + word, font, font_scale, 1)[0][0] <= max_width:
            current_line += ' ' + word
        else:
            lines.append(current_line)
            current_line = word
    lines.append(current_line)
    return lines


@app.route('/')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(debug=False, port=1000)
