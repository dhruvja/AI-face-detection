import pickle 
import cv2 
import time 
import os
import imutils
import face_recognition


cascPathface = os.path.dirname(cv2.__file__) + "/data/haarcascade_frontalface_alt2.xml"
faceCascade = cv2.CascadeClassifier(cascPathface)
data = pickle.loads(open('face_enc', "rb").read())

image = cv2.imread("testing_images/1200px-Donald_Trump_official_portrait.jpg")

rgb = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

gray = cv2.cvtColor(rgb,cv2.COLOR_BGR2GRAY)
faces = faceCascade.detectMultiScale(gray,
                                     scaleFactor=1.1,
                                     minNeighbors=5,
                                     minSize=(60, 60),
                                     flags=cv2.CASCADE_SCALE_IMAGE)


encodings = face_recognition.face_encodings(rgb)
names = []

for encoding in encodings:

    matches = face_recognition.compare_faces(data['encodings'],encoding)

    name = "Unknown"

    if True in matches:
        matchedIds = [i for(i,b) in enumerate(matches) if b]
        counts = {}

        for i in matchedIds:
            name = data['name']
            counts[name] = counts.get(name,0) + 1
            name = max(counts,key = counts.get)

    names.append(name)
    print(names)


    # for ((x, y, w, h), name) in zip(faces, names):
    #         # rescale the face coordinates
    #         # draw the predicted face name on the image
    #         cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    #         cv2.putText(image, name, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
    #          0.75, (0, 255, 0), 2)
    
    # cv2.imshow("Frame", image)
    # cv2.waitKey(0)
