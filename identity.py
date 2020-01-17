import cv2
import pickle
import pickle
import datetime
import json
import time

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainer100.yml")

labels = {"person":1}
with open("labels.pickle", "rb") as f:
    
    og_labels = pickle.load(f)
    labels = {v:k for k,v in og_labels.items()}
cap = cv2.VideoCapture(0)

def write_json(NAME,time1):
    x = {"NAME" : NAME,
         "TIME" : time1,
         }
    y = json.dumps(x)
    print(y)

def iden():
    # Capture frame-by-frame
    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.03, minNeighbors=4)
    for(x,y,w,h) in faces:
        #print(x,y,w,h)
        roi_gray = gray[y:y+h, x:x+w]
        
        roi_color = frame[y:y+h, x:x+w]

        id_, conf = recognizer.predict(roi_gray)
        
        if conf <= 100:
            time1 =  datetime.datetime.now().strftime("%H:%M:%S")
            print(conf)
            font = cv2.FONT_HERSHEY_SIMPLEX
            NAME = labels[id_]
            color = (0, 0, 255)
            stroke = 2
            cv2.putText(frame,NAME, (x,y), font, 1, color, stroke,cv2.LINE_AA)
            write_json(NAME,time1)
        

        color = (255, 0, 0)
        stroke = 2
        width = x + w
        height = y + h
        cv2.rectangle(frame, (x,y), (width, height), color, stroke)

    # Display the resulting frame
    cv2.imshow('frame', frame)


while (True):
    time.sleep(1)
    iden()
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break    
