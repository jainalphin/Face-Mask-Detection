# Computer Vision & Internet of Things
## Task 4 : Detection of Face Mask

### Face Mask Detection
An approach to detecting face masks in
crowded places built using RetinaNet Face for face mask detection 
and Xception network for classification.<br><br>
The mask detection model can be said to be a combination of classification and
face detection model.<br><br>
I have used transfer learning with an 
**Xception model** trained on the **ImageNet 
dataset** with a modified final fully connected layer<br><br>

While using the face detection model, 
several different approaches were tried upon based on existing literature,
and the one which worked the best was a **Multi-Task Cascaded Convolutional Neural Network (MTCNN)**
Face pre-trained model which gave the highest measures of 
recall while experimenting on different use-cases and
 testing images of people in a crowded setting.

Proposed Model for Face Mask Detection:<br>
* Apply the Multi-Task Cascaded Convolutional Neural Network (MTCNN)
 Face model for face detection 
to generate detect face from the input image. 
* Xception model for classification into **mask and no-mask** 
categories for the detected face. 
The final output of these two would be 
the faces detected by MTCNN along with the predicted category
 for each face, that is whether the subject is wearing a mask or not.
 
I have applied this trained model on :
   1. Face Mask Detection on Image:  
   2. Face Mask Detection on Video
   3. Face Mask Detection on WebCam 

Links: <br>
 [Presnetation](https://docs.google.com/presentation/d/1ILqdXf1o2KFGCkYIvN9AytfDcuSH3kmk/edit?usp=sharing&ouid=113685146900593867572&rtpof=true&sd=true&usp=embed_facebook) <br>
 [Youtube](https://lnkd.in/ePXgXsm)
