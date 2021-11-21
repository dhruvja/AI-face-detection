from imutils import paths
import cv2
import pickle
import face_recognition
import os

imagePaths = list(paths.list_images('images'))
knownEncodings = []
knownName = []

for(i,imagePath) in enumerate(imagePaths):
    # Get the name of the person from the path. The os.path.sep is used to seperate the path of the file in seperate elements of a list. Since the folder name contains the name of the person, we are extracting that.
    name = imagePath.split(os.path.sep)[-2]

    image = cv2.imread(imagePath)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    boxes = face_recognition.face_locations(rgb,model='fog')

    encodings = face_recognition.face_encodings(rgb,boxes)

    for encoding in encodings:
        knownEncodings.append(encoding)
        knownName.append(name)
    

data = {"encodings": knownEncodings, "name": knownName}

f = open("face_enc","wb")

f.write(pickle.dumps(data))

f.close()





