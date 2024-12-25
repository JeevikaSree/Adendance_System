from collections.abc import Iterable  # Updated import to avoid DeprecationWarning
import numpy as np
import imutils
import pickle
import time
import cv2
import csv
import os

# Flatten function to handle nested lists
def flatten(lis):
    for item in lis:
        if isinstance(item, Iterable) and not isinstance(item, str):
            for x in flatten(item):
                yield x
        else:
            yield item

# File paths
embeddingFile = "output/embeddings.pickle"
embeddingModel = "openface_nn4.small2.v1.t7"
recognizerFile = "output/recognizer.pickle"
labelEncFile = "output/le.pickle"
studentCSV = "student.csv"  # CSV file for students
attendanceCSV = "output/attendance.csv"  # CSV for marking attendance

# Confidence threshold
conf = 0.5

# Initialize attendance with all as 'A' (Absent)
if not os.path.exists(attendanceCSV):
    with open(studentCSV, 'r') as csvFile, open(attendanceCSV, 'w', newline='') as outFile:
        reader = csv.reader(csvFile)
        writer = csv.writer(outFile)
        for row in reader:
            if len(row) >= 2:  # Ensure the row has at least two columns
                writer.writerow([row[0], row[1], 'A'])  # Name, Roll Number, 'A' for Absent

# Load models
print("[INFO] loading face detector...")
prototxt = "C:/Users/Advpa/OneDrive/Desktop/Adendance_System/model/deploy.prototxt"
model = "C:/Users/Advpa/OneDrive/Desktop/Adendance_System/model/res10_300x300_ssd_iter_140000.caffemodel"
detector = cv2.dnn.readNetFromCaffe(prototxt, model)

print("[INFO] loading face recognizer...")
embedder = cv2.dnn.readNetFromTorch(embeddingModel)
recognizer = pickle.loads(open(recognizerFile, "rb").read())
le = pickle.loads(open(labelEncFile, "rb").read())

# Start video stream
print("[INFO] starting video stream...")
cam = cv2.VideoCapture(0)
time.sleep(1.0)

while True:
    _, frame = cam.read()
    frame = imutils.resize(frame, width=600)
    (h, w) = frame.shape[:2]
    imageBlob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0), swapRB=False, crop=False)
    detector.setInput(imageBlob)
    detections = detector.forward()

    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            face = frame[startY:endY, startX:endX]
            (fH, fW) = face.shape[:2]

            if fW < 20 or fH < 20:
                continue

            # Face embedding
            faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255, (96, 96), (0, 0, 0), swapRB=True, crop=False)
            embedder.setInput(faceBlob)
            vec = embedder.forward()

            # Recognize face
            preds = recognizer.predict_proba(vec)[0]
            j = np.argmax(preds)
            proba = preds[j]
            name = le.classes_[j]

            # Update attendance if confidence is high enough
            if proba * 100 > 70:  # Threshold for marking attendance
                with open(attendanceCSV, 'r') as file:
                    reader = list(csv.reader(file))
                with open(attendanceCSV, 'w', newline='') as file:
                    writer = csv.writer(file)
                    for row in reader:
                        if row[0] == name and row[2] == 'A':  # Mark as Present if Absent
                            row[2] = 'P'
                        writer.writerow(row)

                # Display name and confidence
                text = "{} : {:.2f}%".format(name, proba * 100)
                y = startY - 10 if startY - 10 > 10 else startY + 10
                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 2)
                cv2.putText(frame, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # Press 'ESC' to exit
        break

cam.release()
cv2.destroyAllWindows()
