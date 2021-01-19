import numpy as np
import cv2
from mtcnn.mtcnn import MTCNN
from tensorflow.keras.models import load_model
detector = MTCNN()
model = load_model('../model.h5')

img_size = (200, 200)
colors = { 0: (0,255,0), 1:(0,0,255)}
category2label ={0: 'With mask',1: 'Without mask'}

image = cv2.imread('../image.jpeg')
rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

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
        text = "{}: {:.2f}".format(category2label[target], scores[0][target])
        cv2.rectangle(img=image, pt1=(x, y), pt2=(x + w, y + h), color=colors[target], thickness=2)
        cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)

    except Exception as e:
        print(e)
        print(roi.shape)

cv2.imwrite('../result/detected_image.jpg',image)
print('done')