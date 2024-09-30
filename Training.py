import cv2
import numpy as np
from os import listdir
from os.path import isfile, join, isdir

# Define the data path where images are stored
data_path = 'D:/face_detection/data/'

Training_Data = []
Labels = []
label_dict = {}  
label_count = 0  

# Iterate over each folder in the data path
for person_folder in listdir(data_path):
    person_folder_path = join(data_path, person_folder)
    
    # Check if the current path is a directory
    if not isdir(person_folder_path):
        continue

    # Iterate over each image file in the person's folder
    for file in listdir(person_folder_path):
        image_path = join(person_folder_path, file)
        
        # Read the image in grayscale
        images = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        
        if images is None:
            print(f"Error loading image {image_path}. Skipping this file.")
            continue  
        
        # Append the image to the training data
        Training_Data.append(np.asarray(images, dtype=np.uint8))
        
        # Assign a label to the person if not already assigned
        if person_folder not in label_dict:
            label_dict[person_folder] = label_count
            label_count += 1
        
        Labels.append(label_dict[person_folder])

# Print out the number of training images and their labels
print(f"Number of training images: {len(Training_Data)}")
print(f"Labels: {Labels}")

# Check if there is enough data to train the model
if len(Training_Data) > 1:
    Labels = np.asarray(Labels, dtype=np.int32)
    
    # Attempt to create and train the LBPH face recognizer
    try:
        # Check if the face module is available
        if hasattr(cv2, 'face') and hasattr(cv2.face, 'LBPHFaceRecognizer_create'):
            model = cv2.face.LBPHFaceRecognizer_create()
            model.train(np.asarray(Training_Data), np.asarray(Labels))
            print("Dataset Model Training Completed")
            
            # Save the trained model
            model.save('D:/face_detection/model/face_trained_model.yml')
            print("Model saved successfully.")
        else:
            raise AttributeError("Face recognition module not available. Make sure opencv-contrib-python is installed.")

    except cv2.error as e:
        print("OpenCV error: ", e)
    except AttributeError as e:
        print("Error: ", e)
        
        # Here you could switch to a different method, like face_recognition
        print("Switching to face_recognition module.")
        # Optionally implement a fallback using face_recognition here

else:
    print("Not enough training data to train the model.")
