import time
import pickle
import numpy as np
import cv2
from mtcnn.mtcnn import MTCNN
from tensorflow.keras.models import load_model
detector = MTCNN()
model = load_model('model.h5')
# Video from webcam
cap = cv2.VideoCapture("../video4.mp4")

start_time = time.time()
frame_count = 0

img_size = (128, 128)
colors = { 0: (0,255,0), 1:(0,0,255)}
category2label ={0: 'With mask',1: 'Without mask'}
while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break

    frame_count += 1  # for fps
    frame = cv2.flip(frame, 1)  # Mirror the image
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = detector.detect_faces(rgb)
    for face in faces:
        try:
            x, y, w, h = face['box']
            # Predict
            roi = rgb[y: y + h, x: x + w]
            data = cv2.resize(roi, img_size)
            data = data / 255.
            data = data.reshape((1,) + data.shape)
            scores = model.predict(data)
            target = np.argmax(scores, axis=1)[0]
            # Draw bounding boxes
            cv2.rectangle(img=frame, pt1=(x, y), pt2=(x + w, y + h), color=colors[target], thickness=2)
            cv2.putText(frame, "", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        except Exception as e:
            print(e)
            print(roi.shape)

    elapsed_time = time.time() - start_time
    fps = frame_count / elapsed_time
    cv2.putText(img=frame, text='FPS : ' + str(round(fps, 2)), org=(10, 20), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.5, color=(255, 255, 255), thickness=1)

    # Show the frame
    cv2.imshow('Face Mask Detection', frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()