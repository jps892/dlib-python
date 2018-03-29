import dlib
from skimage import io

image_file = '/home/jps/LIBRARIES/dlib/dlib/examples/faces/2009_004587.jpg'
img = io.imread(image_file)

# Locations of candidate objects will be saved into rects
rects = []
dlib.find_candidate_object_locations(img, rects, min_size=500)

print("number of rectangles found {}".format(len(rects)))
for k, d in enumerate(rects):
    print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
        k, d.left(), d.top(), d.right(), d.bottom()))