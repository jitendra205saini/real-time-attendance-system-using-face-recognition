import cv2
import numpy as np
import os


face_classifier = cv2.CascadeClassifier('C:/Users/jiten/AppData/Local/Programs/Python/Python312/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml')

def face_extractor(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)

    if len(faces) == 0:  
        return None

    faces_cropped = []
    for (x, y, w, h) in faces:
        cropped_face = img[y:y+h, x:x+w]
        faces_cropped.append(cropped_face)

    return faces_cropped    



cap = cv2.VideoCapture(0)
count = 0


person_name = "jitendra"  
save_path = f'D:/face_detection/data/{person_name}/'

os.makedirs(save_path, exist_ok=True) 

while True:
    ret, frame = cap.read()
    faces = face_extractor(frame)
    if faces is not None:
        for face in faces:
            count += 1
            face = cv2.resize(face, (200, 200))  
            face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)  

            
            file_name_path = save_path + str(count) + '.jpg'
            cv2.imwrite(file_name_path, face)

            
            cv2.putText(face, str(count), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow('Face Cropper', face)

            if count == 100:  
                break
    else:
        print("Face not found")
        pass

    if cv2.waitKey(1) == 13 or count == 100:  
        break

cap.release()
cv2.destroyAllWindows()
print('Sample Collection Completed')
