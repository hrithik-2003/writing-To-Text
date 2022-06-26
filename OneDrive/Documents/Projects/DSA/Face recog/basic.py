import cv2
import numpy as np
import face_recognition

#recognising face in the image and convertion color from BGR to RGB
vijayimg = face_recognition.load_image_file("C:/Users/Hrithik/OneDrive/Documents/Projects/DSA/Face recog/images/vijay.jpg")
vijayimg = cv2.cvtColor(vijayimg,cv2.COLOR_BGR2RGB)

#recognising face in the test image and convertion color from BGR to RGB
testimg = face_recognition.load_image_file('C:/Users/Hrithik/OneDrive/Documents/Projects/DSA/Face recog/images/surya.jpg')
testimg = cv2.cvtColor(testimg,cv2.COLOR_BGR2RGB)

#getting the face locations in the image and test image and making a rectangle around the located faces 
face_loc = face_recognition.face_locations(vijayimg)[0]
encodevijay = face_recognition.face_encodings(vijayimg)[0]
cv2.rectangle(vijayimg,(face_loc[3],face_loc[0]),(face_loc[1],face_loc[2]),(0,255,0),2)

face_loc_test = face_recognition.face_locations(testimg)[0]
encodetest = face_recognition.face_encodings(testimg)[0]
cv2.rectangle(testimg,(face_loc_test[3],face_loc_test[0]),(face_loc_test[1],face_loc_test[2]),(0,255,0),2)

#comparing the face with the test face (in this case answer will be true)
result = face_recognition.compare_faces([encodevijay],encodetest)
face_dist = face_recognition.face_distance([encodevijay],encodetest)
print(result)
print(face_dist)

cv2.putText(testimg,f'{result}{round(face_dist[0],2)}',(10,20),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,(0,255,0),1)

cv2.imshow('vijay',vijayimg)
cv2.imshow('vijay-test',testimg)
cv2.waitKey(0)

