import sys
import dlib
from skimage import io

print(
    "Call this program like this:\n"
    "   ./cnn_face_detector.py mmod_human_face_detector.dat ../examples/faces/*.jpg\n"
    "You can get the mmod_human_face_detector.dat file from:\n"
    "    http://dlib.net/files/mmod_human_face_detector.dat.bz2")

cnn_face_detector = dlib.cnn_face_detection_model_v1('/home/jps/Downloads/mmod_human_face_detector.dat')
win = dlib.image_window()

print("Processing file: {}".format('/home/jps/LIBRARIES/dlib/dlib/examples/faces/2007_007763.jpg'))
img = io.imread('/home/jps/LIBRARIES/dlib/dlib/examples/faces/2007_007763.jpg')
# The 1 in the second argument indicates that we should upsample the image
# 1 time.  This will make everything bigger and allow us to detect more
# faces.
dets = cnn_face_detector(img, 1)
print("Number of faces detected: {}".format(len(dets)))
for i, d in enumerate(dets):
    print("Detection {}: Left: {} Top: {} Right: {} Bottom: {} Confidence: {}".format(
        i, d.rect.left(), d.rect.top(), d.rect.right(), d.rect.bottom(), d.confidence))

rects = dlib.rectangles()
rects.extend([d.rect for d in dets])

win.clear_overlay()
win.set_image(img)
win.add_overlay(rects)
dlib.hit_enter_to_continue()