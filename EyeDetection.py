import numpy as np
import dlib
import cv2

EYES = list(range(36,48))

frame_width = 640
frame_height = 480

raw_image = cv2.imread('./face.jpg') #-- 이미지 경로

face_cascade_name = './haarcascade_frontalface_alt.xml' #-- 본인 환경에 맞게 변경할 것
face_cascade = cv2.CascadeClassifier()
if not face_cascade.load(cv2.samples.findFile(face_cascade_name)):
    print('--(!)Error loading face cascade')
    exit(0)

predictor_file = './shape_predictor_68_face_landmarks.dat' #-- 본인 환경에 맞게 변경할 것
predictor = dlib.shape_predictor(predictor_file)

image = cv2.resize(raw_image, (frame_width, frame_height))
faces = face_cascade.detectMultiScale(image)

#- detect face area
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
    rect = dlib.rectangle(int(x), int(y), int(x+w), int(y+h))
        
    #- detect eye area
    points = np.matrix([[p.x, p.y] for p in predictor(image, rect).parts()])
    show_parts = points[EYES]

    for (i, point) in enumerate(show_parts):
        x = point[0,0]
        y = point[0,1]
        cv2.circle(image, (x, y), 1, (0, 255, 255), -1)

cv2.imshow('Face & Eye Area Detection', image)
#cv2.imwrite('output.png',image) #- image output 저장용
cv2.waitKey()
cv2.destroyAllWindows()