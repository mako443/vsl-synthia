import numpy as np
import cv2
import os
import random
import pickle

from semantic.imports import CLASS_IDS, CLASS_COLORS, RELATIONSHIP_TYPES, DIRECTIONS
from semantic.imports import ViewObject, SceneGraph, SceneGraphObject, draw_scenegraph_on_image

from semantic.viewobjects import create_view_objects
from semantic.scenegraphs import create_scenegraph_from_viewobjects, score_scenegraph_to_viewobjects

# base_dir='data/SYNTHIA-SEQS-04-SUMMER/selection'
# file_names=os.listdir( os.path.join(base_dir,'RGB', 'Stereo_Left', DIRECTIONS[0]) )
# direction='Omni_R'
# file_name=np.random.choice(file_names)
# #file_name='000269.png'
# print(file_name)

# rgb=   cv2.imread( os.path.join(base_dir,'RGB','Stereo_Left', direction, file_name) )
# labels=cv2.imread( os.path.join(base_dir,'GT/LABELS','Stereo_Left', direction, file_name), cv2.IMREAD_UNCHANGED)[:,:,2]
# depth= cv2.imread( os.path.join(base_dir,'Depth','Stereo_Left', direction, file_name), cv2.IMREAD_UNCHANGED)

# view_objects=create_view_objects(rgb, labels, depth)
# sg, sgd=create_scenegraph_from_viewobjects(view_objects, return_debug_sg=True)
# score,groundings=score_scenegraph_to_viewobjects(sg, view_objects, unused_factor=0.5)
# print('S',score)

# img=rgb.copy()
# sgd.draw_on_image(img)
# cv2.imshow("",img)
# cv2.waitKey()

# quit()

'''
Module to create View-Objects and Scene-Graphs for the data
'''
#for base_dir in ('data/SYNTHIA-SEQS-04-SUMMER/', 'data/SYNTHIA-SEQS-04-DAWN/', 'data/SYNTHIA-SEQS-04-WINTER/'):
for base_dir in ('data/SYNTHIA-SEQS-04-WINTER/', ):
    file_names=os.listdir( os.path.join(base_dir,'RGB', 'Stereo_Left', DIRECTIONS[0]) )
    print(f'{len(file_names)} positions for {base_dir}...')

    #Create View-Objects and Scene-Graphs
    view_objects={ direction:{} for direction in DIRECTIONS} # {direction: {filename: view-objects} }
    scene_graphs={ direction:{} for direction in DIRECTIONS} # {direction: {filename: Scene-Graph } }
    for direction in DIRECTIONS:
        print(f'Direction: {direction}...')
        view_object_counts=[]
        for file_name in file_names:
            rgb=   cv2.imread( os.path.join(base_dir,'RGB','Stereo_Left', direction, file_name) )
            labels=cv2.imread( os.path.join(base_dir,'GT/LABELS','Stereo_Left', direction, file_name), cv2.IMREAD_UNCHANGED)[:,:,2]
            depth= cv2.imread( os.path.join(base_dir,'Depth','Stereo_Left', direction, file_name), cv2.IMREAD_UNCHANGED)
            assert rgb is not None and labels is not None and depth is not None

            vo=create_view_objects(rgb, labels, depth)
            view_objects[direction][file_name]=vo
            view_object_counts.append(len(vo))

            sg=create_scenegraph_from_viewobjects(vo)
            scene_graphs[direction][file_name]=sg

        print(f'Done, {np.mean(view_object_counts)} view-objects on average, saving')

    pickle.dump(view_objects, open(os.path.join(base_dir,'view_objects.pkl'),'wb'))
    pickle.dump(scene_graphs, open(os.path.join(base_dir,'scene_graphs.pkl'),'wb'))
