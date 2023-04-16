from flask import Flask, Response, render_template
import cv2
from deepface import DeepFace

app = Flask(__name__)

def generate_frames():
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise IOError("Cannot open webcam")

    while True:
        ret, frame = cap.read()

        if not ret:
            break

       
        try:
           
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            
            result = DeepFace.analyze(rgb_frame, actions=['emotion'],enforce_detection=False)

            
            
            dominant_emotion = result[0]['dominant_emotion']

        except Exception as e:
            dominant_emotion = e

        
        cv2.putText(frame, dominant_emotion, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

        
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

@app.route('/')
def display():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
