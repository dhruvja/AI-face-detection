import pickle 
import cv2 
import time 
import os
import imutils
import face_recognition
import random


cascPathface = os.path.dirname(cv2.__file__) + "/data/haarcascade_frontalface_alt2.xml"
faceCascade = cv2.CascadeClassifier(cascPathface)
data = pickle.loads(open('face_enc', "rb").read())

image = cv2.imread("testing_images/us-intelligence-believes-that-russia-meddles-in-the-2016-election-which-brought-donald-trump-to-power-1620443823343-2.jpg")

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
            name = data['name'][i]
            counts[name] = counts.get(name,0) + 1
            name = max(counts,key = counts.get)

    if name == "Unknown":
        person_name = input("The person could not be recognised, Please enter the name of the new person to add him to our database. If not press 0 to skip")
        if person_name != "0":
            boxes = face_recognition.face_locations(rgb,model="fog")
            person_encodings = face_recognition.face_encodings(rgb,boxes)

            for person_encoding in person_encodings:
                data['encodings'].append(person_encoding)
                data['name'].append(person_name)
            
            path = "/Users/Deepak/Documents/AI face detection/images/" + person_name

            os.mkdir(path)

            filename = person_name + str(random.randint(10000,1000000)) + ".jpg"

            cv2.imwrite(os.path.join(path,filename),image)
        
            file = open('face_enc','wb')
            file.write(pickle.dumps(data))
            file.close()

            
        
        else:
            print(name)
    
    else:
        names.append(name)
        print(names)


        for ((x, y, w, h), name) in zip(faces, names):
                # rescale the face coordinates
                # draw the predicted face name on the image
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(image, name, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                0.75, (0, 255, 0), 2)
        
        # cv2.imshow("Frame", image)
        path = "/Users/Deepak/Documents/AI face detection/testing_images"
        cv2.imwrite(os.path.join(path,'updatedImage.jpg'),image)
        # cv2.waitKey(0)
