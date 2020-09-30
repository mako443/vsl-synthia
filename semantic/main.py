import numpy as np
import cv2
import os
import random
import pickle

from semantic.imports import CLASS_IDS, CLASS_COLORS, RELATIONSHIP_TYPES, DIRECTIONS
from semantic.imports import ViewObject, SceneGraph, SceneGraphObject, draw_scenegraph_on_image

from semantic.viewobjects import create_view_objects
from semantic.scenegraphs import create_scenegraph_from_viewobjects, score_scenegraph_to_viewobjects



base_dir='data/SYNTHIA-SEQS-04-SUMMER/selection'
file_names=os.listdir( os.path.join(base_dir,'RGB', 'Stereo_Left', DIRECTIONS[0]) )
direction='Omni_R'
file_name=np.random.choice(file_names)
#file_name='000269.png'
print(file_name)

rgb=   cv2.imread( os.path.join(base_dir,'RGB','Stereo_Left', direction, file_name) )
labels=cv2.imread( os.path.join(base_dir,'GT/LABELS','Stereo_Left', direction, file_name), cv2.IMREAD_UNCHANGED)[:,:,2]
depth= cv2.imread( os.path.join(base_dir,'Depth','Stereo_Left', direction, file_name), cv2.IMREAD_UNCHANGED)

view_objects=create_view_objects(rgb, labels, depth)
sg, sgd=create_scenegraph_from_viewobjects(view_objects, return_debug_sg=True)
score,groundings=score_scenegraph_to_viewobjects(sg, view_objects, unused_factor=0.5)
print('S',score)

img=rgb.copy()
sgd.draw_on_image(img)
cv2.imshow("",img)
cv2.waitKey()

quit()

'''
Module to create View-Objects and Scene-Graphs for the data
'''
base_dir='data/SYNTHIA-SEQS-04-SUMMER/selection'
file_names=os.listdir( os.path.join(base_dir,'RGB', 'Stereo_Left', DIRECTIONS[0]) )

#Create View-Objects
view_objects={ direction:{} for direction in DIRECTIONS} # {direction: {filename: view-objects} }
for direction in DIRECTIONS:
    print(f'Direction: {direction}...')
    view_object_counts=[]
    for file_name in file_names:
        rgb=   cv2.imread( os.path.join(base_dir,'RGB','Stereo_Left', direction, file_name) )
        labels=cv2.imread( os.path.join(base_dir,'GT/LABELS','Stereo_Left', direction, file_name), cv2.IMREAD_UNCHANGED)[:,:,2]
        depth= cv2.imread( os.path.join(base_dir,'Depth','Stereo_Left', direction, file_name), cv2.IMREAD_UNCHANGED)
        assert rgb is not None and labels is not None and depth is not None

        v=create_view_objects(rgb, labels, depth)
        view_objects[direction][file_name]=v
        view_object_counts.append(len(v))

    print(f'Done, {np.mean(view_object_counts)} view-objects on average, saving')

pickle.dump(view_objects, open(os.path.join(base_dir,'view_objects.pkl'),'wb'))

# labels=cv2.imread('data/testim/gt_label.png', cv2.IMREAD_UNCHANGED)[:,:,2]
# depth=cv2.imread('data/testim/depth.png', cv2.IMREAD_UNCHANGED)
# rgb=cv2.imread('data/testim/000000.png')

# view_objects=create_view_objects(rgb, labels, depth)
# print(len(view_objects))

# sg, sgd=create_scenegraph_from_viewobjects(view_objects, return_debug_sg=True)

# score,groundings=score_scenegraph_to_viewobjects(sg, view_objects, unused_factor=0.5)
# print('S',score)

# img=rgb.copy()
# sgd.draw_on_image(img)
# cv2.imshow("",img)
# cv2.waitKey()


# img=rgb.copy()
# draw_scenegraph_on_image(img, groundings)
# cv2.imshow("",img)
# cv2.waitKey()