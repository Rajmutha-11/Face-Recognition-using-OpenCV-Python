import cv2
import numpy as np
import face_recognition
import os
from datetime import *

path = 'BasicImages'
images = []
classname=[]
mylist=os.listdir(path)

for x in mylist:
    currentImage = cv2.imread(f'{path}/{x}')
    images.append(currentImage)
    classname.append(os.path.splitext(x)[0])


def findencodings(imagess):
    encodelist =[]
    for img in imagess:
        img= cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        encode=face_recognition.face_encodings(img)[0]
        encodelist.append(encode)
    return encodelist


def markAttendance(name):
    with open('Attendance.csv', 'r+') as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now = datetime.now()

            datetoday = now.strftime("%x")
            weekday = now.strftime("%A")
            dtString = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{weekday},{dtString}')

encodeListKnown = findencodings(images)
print("Encoding Complete")

cap=cv2.VideoCapture(0)

while True:
    success,img = cap.read()
    rgb_img = img[:, :, ::-1]


    facesOfCurrentFrame = face_recognition.face_locations(rgb_img)
    encodesOfCurrentFrame = face_recognition.face_encodings(rgb_img,facesOfCurrentFrame)

    for encodeFace,(y1,x2,y2,x1) in zip(encodesOfCurrentFrame,facesOfCurrentFrame):
        matches = face_recognition.compare_faces(encodeListKnown,encodeFace)
        name = "Unknown"

        faceDistance=face_recognition.face_distance(encodeListKnown,encodeFace)
        matchindex = np.argmin(faceDistance)

        if matches[matchindex]:
            name =classname[matchindex].upper()


        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
        cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
        markAttendance(name)



    cv2.imshow("WebCam",img)

    if cv2.waitKey(1) ==27:
        break

cap.release()
cv2.destroyAllWindows()


