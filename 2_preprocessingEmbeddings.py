from imutils import paths
import numpy as np
import imutils
import pickle
import cv2
import os

# Directory paths
dataset = "dataset"
output_dir = "C:/Users/Advpa/OneDrive/Desktop/Adendance_System/output"
embeddingFile = os.path.join(output_dir, "embeddings.pickle")
embeddingModel = "openface_nn4.small2.v1.t7"

# Paths for the face detection model
prototxt = "C://Users//Advpa//OneDrive//Desktop//Adendance_System//model//deploy.prototxt"
model = "C://Users//Advpa//OneDrive//Desktop//Adendance_System//model//res10_300x300_ssd_iter_140000.caffemodel"

# Check if paths exist
if not os.path.exists(prototxt):
    print(f"Error: deploy.prototxt file not found at {prototxt}")
if not os.path.exists(model):
    print(f"Error: Caffe model file not found at {model}")
if not os.path.exists(output_dir):
    os.makedirs(output_dir)  # Create output directory if it doesn't exist

# Load face detection and embedding models
detector = cv2.dnn.readNetFromCaffe(prototxt, model)
embedder = cv2.dnn.readNetFromTorch(embeddingModel)

# Get image paths
imagePaths = list(paths.list_images(dataset))

# Initialize data structures
knownEmbeddings = []
knownNames = []
total = 0
conf = 0.5

# Process each image
for (i, imagePath) in enumerate(imagePaths):
    print(f"Processing image {i + 1}/{len(imagePaths)}")
    name = imagePath.split(os.path.sep)[-2]
    image = cv2.imread(imagePath)

    if image is None:
        print(f"Warning: Could not read image {imagePath}. Skipping.")
        continue

    image = imutils.resize(image, width=600)
    (h, w) = image.shape[:2]

    # Convert image to blob for DNN face detection
    imageBlob = cv2.dnn.blobFromImage(
        cv2.resize(image, (300, 300)), 1.0, (300, 300),
        (104.0, 177.0, 123.0), swapRB=False, crop=False
    )

    # Set the input and detect faces
    detector.setInput(imageBlob)
    detections = detector.forward()

    if len(detections) > 0:
        i = np.argmax(detections[0, 0, :, 2])
        confidence = detections[0, 0, i, 2]

        if confidence > conf:
            # Get coordinates for face ROI
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            face = image[startY:endY, startX:endX]
            (fH, fW) = face.shape[:2]

            # Skip small detections
            if fW < 20 or fH < 20:
                continue

            # Convert face to blob for embedding
            faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255, (96, 96),
                                             (0, 0, 0), swapRB=True, crop=False)
            embedder.setInput(faceBlob)
            vec = embedder.forward()

            # Append name and embedding
            knownNames.append(name)
            knownEmbeddings.append(vec.flatten())
            total += 1

print(f"Total embeddings: {total}")

# Serialize embeddings and names to disk
data = {"embeddings": knownEmbeddings, "names": knownNames}
with open(embeddingFile, "wb") as f:
    f.write(pickle.dumps(data))

print("Process Completed")
