import numpy as np
import cv2
import os
import random
import pickle

from semantic.imports import CLASS_IDS, CLASS_COLORS, RELATIONSHIP_TYPES
from semantic.imports import ViewObject, SceneGraph, SceneGraphObject

from semantic.viewobjects import create_view_objects
from semantic.scenegraphs import create_scenegraph_from_viewobjects

labels=cv2.imread('data/testim/gt_label.png', cv2.IMREAD_UNCHANGED)[:,:,2]
depth=cv2.imread('data/testim/depth.png', cv2.IMREAD_UNCHANGED)
rgb=cv2.imread('data/testim/000000.png')

view_objects=create_view_objects(rgb, labels, depth)
print(len(view_objects))

sg, sgd=create_scenegraph_from_viewobjects(view_objects, return_debug_sg=True)

sgd.draw_on_image(rgb)

cv2.imshow("",rgb)
cv2.waitKey()