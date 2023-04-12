import cv2
import face_recognition
import pickle
import os

# Importing student images

folderPath = 'Images'
pathList = os.listdir(folderPath)
imgList = []
visitorID = []
for path in pathList:
    imgList.append(cv2.imread(os.path.join(folderPath,path)))
    visitorID.append(os.path.splitext(path)[0])


def findEncodings(imagesList):
    encodeList = []
    for img in imagesList:
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)

    return encodeList


print("Encoding Started... ")
encodeListKnown = findEncodings(imgList)
encodeListWithIDs = [encodeListKnown, visitorID]
print("Encoding Completed")

file = open("EncodeFile.p", 'wb')
pickle.dump(encodeListWithIDs, file)
file.close()
print("File Saved")