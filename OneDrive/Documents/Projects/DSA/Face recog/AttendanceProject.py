import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime
#import pyttsx3

#engine = pyttsx3.init()
#engine.setProperty('rate',120)
#voices = engine.getProperty('voices')
#engine.setProperty('voice',voices[1].id)

path = 'C:/Users/Hrithik/OneDrive/Documents/Projects/DSA/Face recog/Attendance_images'
images = []
classnames = []



myList = os.listdir(path)
print(myList)

for i in myList:
    cur = cv2.imread(f'{path}/{i}')
    images.append(cur)
    classnames.append(os.path.splitext(i)[0])

print(classnames)

def find_encodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)

    return encodeList

def markAttendance(name):
    with open('Attendance.csv','r+') as f:
        datalist = f.readlines()
        nameList = []
        print(datalist)
        for line in datalist:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now = datetime.now()
            datestr = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{datestr}')



markAttendance('Elon')

encodings = find_encodings(images)

print("Encoding Complete...")

capture = cv2.VideoCapture(0)

while True:
    success, img = capture.read()
    imgS = cv2.resize(img, (0,0), None,0.25,0.25)
    imgS = cv2.cvtColor(imgS,cv2.COLOR_BGR2RGB)


    faces = face_recognition.face_locations(imgS)
    encode = face_recognition.face_encodings(imgS, faces) 

    for encodeface, faceloc in zip(encode,faces):
        matching = face_recognition.compare_faces(encodings, encodeface)
        facedist = face_recognition.face_distance(encodings, encodeface)
        index = np.argmin(facedist)
        #print(max(facedist),min(facedist),max(facedist)-min(facedist))

        if matching[index] and ((max(facedist)-min(facedist))>0.14):
            name = classnames[index].upper()
            #print(name)
            y1,x2,y2,x1 = faceloc
            y1,x2,y2,x1 = y1*4,x2*4,y2*4,x1*4
            cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,0),1)
            cv2.rectangle(img,(x1,y2-35),(x2,y2),(255,0,0),cv2.FILLED)
            cv2.putText(img, name,(x1+6,y2-6),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),1)


                
        else:
            y1,x2,y2,x1 = faceloc
            y1,x2,y2,x1 = y1*4,x2*4,y2*4,x1*4
            cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,0),1)
            cv2.rectangle(img,(x1,y2-35),(x2,y2),(255,0,0),cv2.FILLED)
            cv2.putText(img, "UNKNOWN",(x1+6,y2-6),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),1)

    cv2.imshow('Webcam', img)
    cv2.waitKey(1)


