#get_face_chip is only supported if you compiled dlib with numpy installed!
import sys
import dlib
import cv2
import numpy as np

def show_jittered_images(jittered_images):
    '''
        Shows the specified jittered images one by one
    '''
    for img in jittered_images:
        cv_bgr_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imshow('image',cv_bgr_img)
        cv2.waitKey(0)

print(
    "Call this program like this:\n"
    "   ./face_jitter.py shape_predictor_5_face_landmarks.dat\n"
    "You can download a trained facial shape predictor from:\n"
    "    http://dlib.net/files/shape_predictor_5_face_landmarks.dat.bz2\n")

predictor_path = "/home/jps/Downloads/shape_predictor_5_face_landmarks.dat"
face_file_path = "/home/jps/LIBRARIES/dlib/dlib/examples/faces/Tom_Cruise_avp_2014_4.jpg"

# Load all the models we need: a detector to find the faces, a shape predictor
# to find face landmarks so we can precisely localize the face
detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor(predictor_path)

# Load the image using OpenCV
bgr_img = cv2.imread(face_file_path)
if bgr_img is None:
    print("Sorry, we could not load '{}' as an image".format(face_file_path))
    exit()

# Convert to RGB since dlib uses RGB images
img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)

# Ask the detector to find the bounding boxes of each face.
dets = detector(img)

num_faces = len(dets)

# Find the 5 face landmarks we need to do the alignment.
faces = dlib.full_object_detections()
for detection in dets:
    faces.append(sp(img, detection))

# Get the aligned face image and show it
image = dlib.get_face_chip(img, faces[0], size=320)
cv_bgr_img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
cv2.imshow('image',cv_bgr_img)
cv2.waitKey(0)

# Show 5 jittered images without data augmentation
jittered_images = dlib.jitter_image(image, num_jitters=5)
show_jittered_images(jittered_images)

# Show 5 jittered images with data augmentation
jittered_images = dlib.jitter_image(image, num_jitters=5, disturb_colors=True)
show_jittered_images(jittered_images)
cv2.destroyAllWindows()