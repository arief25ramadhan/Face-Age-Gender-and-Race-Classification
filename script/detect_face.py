from retinaface import RetinaFace
import cv2
import matplotlib.pyplot as plt

def detect_face(img_path):
    obj = RetinaFace.detect_faces(img_path)
    img = cv2.imread(img_path)

    for key in obj.keys():
        identity = obj[key]
        facial_area = identity["facial_area"]
        cv2.rectangle(img, (facial_area[2], facial_area[3]), (facial_area[0], facial_area[1]), (255, 0, 0), 2)

    cv2.imwrite('/home/nodeflux/Desktop/pytorch_course/Face-Age-Gender-and-Race-Classification/script/result.jpg', img)

img_path = '/home/nodeflux/Desktop/pytorch_course/Face-Age-Gender-and-Race-Classification/script/example.jpg'
detect_face(img_path)