import numpy as np
import cv2
import os
import random
import pickle

from semantic.imports import CLASS_IDS, CLASS_COLORS, RELATIONSHIP_TYPES
from semantic.imports import ViewObject, SceneGraph, SceneGraphObject

'''
Strategies:
-simply compare centroids (depth-centroid)
-compare rotated rectangles
-compare convex-hull points (exhaustive compare necessary?)
'''

### Strategy: centroids
def get_distances(sub,obj):
    d=obj.centroid_c-sub.centroid_c
    return (d[0], -d[0], d[1], -d[1], d[2], -d[2])

def get_relationship_type(sub,obj):
    dists=get_distances(sub,obj)
    return RELATIONSHIP_TYPES[ np.argmax(dists) ]

def score_relationship_type(sub, rel_type, obj):
    distances=get_distances(sub,obj)
    return np.clip(distances[RELATIONSHIP_TYPES.index(rel_type)] / np.max(distances), 0,1)

def score_color(v: ViewObject, color_name):
    assert color_name in COLOR_NAMES
    color_distances= np.linalg.norm( COLORS-v.color, axis=1 )

    score= np.min(color_distances) / color_distances[COLOR_NAMES.index(color_name)]
    return np.clip(score,0,1)

'''
Strategy: Create a relationship for each object w/ its nearest neighbor (no doublicates)
'''
def create_scenegraph_from_viewobjects(view_objects, return_debug_sg=False):
    scene_graph=SceneGraph()
    if return_debug_sg: scene_graph_debug=SceneGraph()
    blocked_subjects=[]

    if len(view_objects)<2:
        #print('scenegraph_for_view_cluster3d_nnRels(): returning Empty Scene Graph, not enough view objects')
        return scene_graph    

    for sub in view_objects:
        if sub in blocked_subjects: continue    
    
        #Find closest object
        min_dist=np.inf
        min_obj=None
        for obj in view_objects:
            if sub is obj: continue

            dist=np.linalg.norm(sub.centroid_c - obj.centroid_c)
            if dist<min_dist:
                min_dist=dist
                min_obj=obj    

        obj=min_obj
        assert obj is not None
        rel_type=get_relationship_type(sub, obj)    

        blocked_subjects.append(obj) #No doublicate relations, CARE: do before flip

        scene_graph.add_relationship( SceneGraphObject.from_viewobject(sub), rel_type,  SceneGraphObject.from_viewobject(obj) )        
        if return_debug_sg: scene_graph_debug.add_relationship( sub, rel_type, obj )   

    if return_debug_sg: return scene_graph, scene_graph_debug
    else: return scene_graph



def score_scenegraph_to_viewobjects():
    pass

if __name__=='__main__':
    labels=cv2.imread('data/testim/gt_label.png', cv2.IMREAD_UNCHANGED)[:,:,2]
    depth=cv2.imread('data/testim/depth.png', cv2.IMREAD_UNCHANGED)
    rgb=cv2.imread('data/testim/000000.png')

    view_objects=create_view_objects(rgb, labels, depth)
    print(len(view_objects))


