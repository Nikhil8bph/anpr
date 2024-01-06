# import base64
# from flask import Flask, render_template
# from flask_socketio import SocketIO, emit
# import cv2
# import numpy as np

# app = Flask(__name__)
# socketio = SocketIO(app)

# @socketio.on('frame')
# def process_frame(data):
#     # Decode frame from base64
#     frame = cv2.imdecode(np.frombuffer(base64.b64decode(data), dtype=np.uint8), -1)
#     # Convert to grayscale
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     # Emit processed frame back to frontend
#     emit('processed_frame', gray.tobytes())

# if __name__ == '__main__':
#     socketio.run(app)

from main import ANPR

anpr = ANPR('./test-data/sample.mp4')
anpr.video_implementation_on_ip()