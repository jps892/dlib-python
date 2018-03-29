import sys
import cv2
import glob
import dlib
from skimage import io

print(
    "Call this program like this:\n"
    "   ./cnn_face_detector.py mmod_human_face_detector.dat ../examples/faces/*.jpg\n"
    "You can get the mmod_human_face_detector.dat file from:\n"
    "    http://dlib.net/files/mmod_human_face_detector.dat.bz2")

cnn_face_detector = dlib.cnn_face_detection_model_v1('/home/jps/Downloads/mmod_human_face_detector.dat')
win = dlib.image_window()
image_list = []
path = "/home/jps/LIBRARIES/dlib/dlib/examples/faces/*(copy).jpg"
for f in glob.glob(path):
    print("Processing file: {}".format(f))
    img = io.imread(f)
    image_list.append(img)
    # The 1 in the second argument indicates that we should upsample the image
    # 1 time.  This will make everything bigger and allow us to detect more
    # faces.
dets = cnn_face_detector(image_list, 1)
'''
This detector returns a mmod_rectangles object. This object contains a list of mmod_rectangle objects.
These objects can be accessed by simply iterating over the mmod_rectangles object
The mmod_rectangle object has two member variables, a dlib.rectangle object, and a confidence score.

It is also possible to pass a list of images to the detector.
    - like this: dets = cnn_face_detector([image_list], upsample_num, batch_size = 128)
    
In this case it will return a mmod_rectangless object.
This object behaves just like a list of lists and can be iterated over.
'''
ii = 0
for det in dets:
    print("Number of faces detected: {}".format(len(det)))
    for i, d in enumerate(det):
        print("Detection {}: Left: {} Top: {} Right: {} Bottom: {} Confidence: {}".format(
            i, d.rect.left(), d.rect.top(), d.rect.right(), d.rect.bottom(), d.confidence))

    rects = dlib.rectangles()
    rects.extend([d.rect for d in det])

    win.clear_overlay()
    win.set_image(image_list[ii])
    win.add_overlay(rects)
    ii = ii+1
    dlib.hit_enter_to_continue()