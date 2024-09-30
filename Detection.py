import cv2
import numpy as np
from os import listdir
from os.path import isfile, join, isdir
import pandas as pd
from datetime import datetime


model = cv2.face.LBPHFaceRecognizer_create()
model.read('D:/face_detection/model/face_trained_model.yml')
print("Trained Model Loaded")

data_path = 'D:/face_detection/data/'
names = {}

label_count = 0
for person_folder in listdir(data_path):
    if isdir(join(data_path, person_folder)):
        names[label_count] = person_folder
        label_count += 1


face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def face_detector(img, size=0.5):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)
    
    if len(faces) == 0:
        return img, None, None, None, None, None

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        roi = img[y:y + h, x:x + w]
        roi = cv2.resize(roi, (200, 200))
        return img, roi, x, y, w, h

def is_already_attended(name):
    file_path = 'D:/face_detection/attendance.xlsx'
    current_date = datetime.now().strftime("%Y-%m-%d")
    
    try:
        df = pd.read_excel(file_path)
        return ((df['Name'] == name) & (df['Date'] == current_date)).any()
    except FileNotFoundError:
        return False

def log_attendance(name):
    file_path = 'D:/face_detection/attendance.xlsx'
    current_time = datetime.now().strftime("%H:%M:%S")
    current_date = datetime.now().strftime("%Y-%m-%d")
    data = {'Name': [name], 'Date': [current_date], 'Time': [current_time]}
    df = pd.DataFrame(data)
    
    try:
        existing_df = pd.read_excel(file_path)
        df = pd.concat([existing_df, df], ignore_index=True)
        df.to_excel(file_path, index=False)
    except FileNotFoundError:
        df.to_excel(file_path, index=False)


cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    image, face, x, y, w, h = face_detector(frame)

    if face is not None:
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        result = model.predict(face)

        if result[1] < 500:
            confidence = int(100 * (1 - (result[1]) / 300))
        else:
            confidence = 0

        if confidence > 75:
            person_name = names[result[0]]
            cv2.putText(image, person_name, (x, y + h + 20), cv2.FONT_HERSHEY_COMPLEX, 1, (42, 235, 35), 2)
            
            if not is_already_attended(person_name):
                log_attendance(person_name)
            else:
                print(f"Attendance already marked for {person_name} today.")
        else:
            cv2.putText(image, "Unknown", (x, y + h + 20), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
    else:
        cv2.putText(image, "Face Not Found", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)

    cv2.imshow('Face Detector & Recognizer', image)

    if cv2.waitKey(1) == 13:  
        break

cap.release()
cv2.destroyAllWindows()
