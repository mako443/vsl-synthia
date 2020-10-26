import numpy as np
import cv2
import os
import random
import pickle

from semantic.imports import CLASS_IDS, CLASS_COLORS, RELATIONSHIP_TYPES, COLORS, COLOR_NAMES, CORNERS, CORNER_NAMES, IMAGE_WIDTH, IMAGE_HEIGHT
from semantic.imports import ViewObject, SceneGraph, SceneGraphObject

'''
Strategies:
-simply compare centroids (depth-centroid)
-compare rotated rectangles
-compare convex-hull points (exhaustive compare necessary?)
'''

### Strategy: centroids ###
def get_distances(sub,obj):
    d=obj.centroid_c-sub.centroid_c
    return (d[0], -d[0], d[1], -d[1], d[2], -d[2])

def get_relationship_type(sub,obj):
    dists=get_distances(sub,obj)
    return RELATIONSHIP_TYPES[ np.argmax(dists) ]

def score_relationship_type(sub, rel_type, obj):
    distances=get_distances(sub,obj)
    return np.clip(distances[RELATIONSHIP_TYPES.index(rel_type)] / np.max(distances), 0,1)

# def score_color(v: ViewObject, color_name):
#     assert color_name in COLOR_NAMES
#     color_distances= np.linalg.norm( COLORS-v.color, axis=1 )

#     score= np.min(color_distances) / color_distances[COLOR_NAMES.index(color_name)]
#     return np.clip(score,0,1)

# def score_corner(v: ViewObject, corner_name):
#     distances= np.linalg.norm( CORNERS - (v.centroid_i[0:2]/(IMAGE_WIDTH, IMAGE_HEIGHT) ), axis=1)
#     return np.clip( np.min(distances)/distances[CORNER_NAMES.index(corner_name)] ,0,1)
### Strategy: centroids ###

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

# def create_scenegraph_from_viewobjects(view_objects, return_debug_sg=False):
#     scene_graph=SceneGraph()
#     if return_debug_sg: scene_graph_debug=SceneGraph()
#     #blocked_subjects=[]
#     blocked_pairs=[]

#     if len(view_objects)<2:
#         #print('scenegraph_for_view_cluster3d_nnRels(): returning Empty Scene Graph, not enough view objects')
#         return scene_graph    

#     for sub in view_objects:
#         #if sub in blocked_subjects: continue   
    
#         #Find closest object
#         sub_dists= np.array([ np.linalg.norm(sub.centroid_c - v.centroid_c) for v in view_objects if v!=sub ])
#         obj_candidates= np.array([ v for v in view_objects if v!=sub ])
#         sorted_indices=np.argsort(sub_dists)

#         #Pick the two closest objects
#         for obj in obj_candidates[sorted_indices[0:1]]:
#             #check the inverse rel. doesn't exists yet
#             if (sub, obj) in blocked_pairs or (obj,sub) in blocked_pairs:
#                 continue

#             assert obj is not None
#             rel_type=get_relationship_type(sub, obj)    

#             #blocked_subjects.append(obj) #No doublicate relations, CARE: do before flip
#             blocked_pairs.append( (sub,obj) )

#             scene_graph.add_relationship( SceneGraphObject.from_viewobject(sub), rel_type,  SceneGraphObject.from_viewobject(obj) )        
#             if return_debug_sg: scene_graph_debug.add_relationship( sub, rel_type, obj )   

#     if return_debug_sg: return scene_graph, scene_graph_debug
#     else: return scene_graph

def score_scenegraph_to_viewobjects(scene_graph, view_objects, unused_factor=0.5, verbose=False):
    MIN_SCORE=0.01 #OPTION: hardest penalty for relationship not found | TODO: re-adjust based on best metric combination
    best_groundings=[None for i in range(len(scene_graph.relationships))]
    best_scores=[MIN_SCORE for i in range(len(scene_graph.relationships))] 

    #Can't score empty graphs 1.0 then apply unused_factor because the factor is not enough to compensate
    if scene_graph.is_empty() or len(view_objects)<2:
        return 0.0, None    

    for i_relation, relation in enumerate(scene_graph.relationships):
        assert type(relation[0] is SceneGraphObject)

        sub_sgo, rel_type, obj_sgo= relation

        for sub_vo in [v for v in view_objects if v.label==sub_sgo.label]:
            sub_nn_dists= [ np.linalg.norm(sub_vo.centroid_c - v.centroid_c) for v in view_objects if v!=sub_vo ] #TODO: objects of correct class or all?! -> of all, just like assignment

            sub_vo_color_dists=np.linalg.norm(sub_vo.color - COLORS, axis=1)
            s_color_sub= 1.0- ( np.linalg.norm(COLORS[COLOR_NAMES.index(sub_sgo.color)] - sub_vo.color) - np.min(sub_vo_color_dists) ) / np.max(sub_vo_color_dists)

            sub_vo_corner_dists=np.linalg.norm(sub_vo.centroid_i[0:2]/(IMAGE_WIDTH, IMAGE_HEIGHT) - CORNERS, axis=1)
            s_corner_sub= 1.0- ( np.linalg.norm(CORNERS[CORNER_NAMES.index(sub_sgo.corner)] - sub_vo.centroid_i[0:2]/(IMAGE_WIDTH, IMAGE_HEIGHT)) - np.min(sub_vo_corner_dists) ) / np.max(sub_vo_corner_dists)

            for obj_vo in [v for v in view_objects if v.label==obj_sgo.label and v!=sub_vo]:
                s_reltype=score_relationship_type(sub_vo, rel_type, obj_vo)

                obj_vo_color_dists=np.linalg.norm(obj_vo.color - COLORS, axis=1)
                s_color_obj= 1.0- ( np.linalg.norm(COLORS[COLOR_NAMES.index(obj_sgo.color)] - obj_vo.color) - np.min(obj_vo_color_dists) ) / np.max(obj_vo_color_dists)

                obj_vo_corner_dists=np.linalg.norm(obj_vo.centroid_i[0:2]/(IMAGE_WIDTH, IMAGE_HEIGHT) - CORNERS, axis=1)
                s_corner_obj= 1.0- ( np.linalg.norm(CORNERS[CORNER_NAMES.index(obj_sgo.corner)] - obj_vo.centroid_i[0:2]/(IMAGE_WIDTH, IMAGE_HEIGHT)) - np.min(obj_vo_corner_dists) ) / np.max(obj_vo_corner_dists)

                s_nn= 1.0 - ( np.linalg.norm(sub_vo.centroid_c - obj_vo.centroid_c) - np.min(sub_nn_dists) ) / np.max(sub_nn_dists)

                
                s_ground= s_reltype * s_color_sub * s_color_obj * s_corner_sub * s_corner_obj * s_nn #TODO: reformulate rel-type in same fashion?
                
                if s_ground>best_scores[i_relation]:
                    best_groundings[i_relation]=(sub_vo,rel_type,obj_vo)
                    best_scores[i_relation]=s_ground

    if verbose:
        for i in range(len(best_groundings)):
            if best_groundings[i] is None:
                print('Not found:', scene_graph.relationships[i][0], scene_graph.relationships[i][1], scene_graph.relationships[i][2] )

    #if score_combine=='multiply': final_score = np.prod(best_scores)
    #if score_combine=='mean':     final_score = np.mean(best_scores)
    final_score = np.mean(best_scores)    

    if unused_factor is not None:
        unused_view_objects=[v for v in view_objects]
        for grounding in best_groundings:
            if grounding is not None:
                if grounding[0] in unused_view_objects: unused_view_objects.remove(grounding[0])                    
                if grounding[2] in unused_view_objects: unused_view_objects.remove(grounding[2])

        final_score*= unused_factor**(len(unused_view_objects))

    return final_score, best_groundings   

#TODO: score against 10 corners
#TODO: score all "1.0 - relative" as in SG-SG -> maybe even one unified scoring
#TODO: afterwards accept results! Try GE and VGE-CO "blindly"! -> GE sucks ✖
def score_scenegraph_to_viewobjects_old(scene_graph, view_objects, unused_factor=0.5, verbose=False):
    MIN_SCORE=0.1 #OPTION: hardest penalty for relationship not found | TODO: re-adjust based on best metric combination
    best_groundings=[None for i in range(len(scene_graph.relationships))]
    best_scores=[MIN_SCORE for i in range(len(scene_graph.relationships))] 

    #Can't score empty graphs 1.0 then apply unused_factor because the factor is not enough to compensate
    if scene_graph.is_empty() or len(view_objects)<2:
        return 0.0, None    

    for i_relation, relation in enumerate(scene_graph.relationships):
        assert type(relation[0] is SceneGraphObject)

        subject_label, rel_type, object_label = relation[0].label, relation[1], relation[2].label
        subject_color, object_color = relation[0].color,relation[2].color
        subject_corner, object_corner = relation[0].corner,relation[2].corner

        for sub in [obj for obj in view_objects if obj.label==subject_label]: 
            subject_nn_dist= np.min([ np.linalg.norm(sub.centroid_c - obj.centroid_c) for obj in view_objects if obj is not sub]) #TODO: objects of correct class or all?!

            for obj in [obj for obj in view_objects if obj.label==object_label]:
                if sub==obj: continue

                relationship_score= score_relationship_type(sub, rel_type, obj)
                #CARE: currently color ablation!
                color_score_sub= 1.0 #score_color(sub, subject_color)
                color_score_obj= 1.0 #score_color(obj, object_color)

                corner_score_sub= score_corner(sub, subject_corner)
                corner_score_obj= score_corner(obj, object_corner)

                nn_score = np.clip( subject_nn_dist/np.linalg.norm(sub.centroid_c - obj.centroid_c) , 0,1)

                score= relationship_score * color_score_sub * color_score_obj * corner_score_sub * corner_score_obj * nn_score

                if score>best_scores[i_relation]:
                    best_groundings[i_relation]=(sub,rel_type,obj)
                    best_scores[i_relation]=score        

    if verbose:
        for i in range(len(best_groundings)):
            if best_groundings[i] is None:
                print('Not found:', scene_graph.relationships[i][0], scene_graph.relationships[i][1], scene_graph.relationships[i][2] )

    #if score_combine=='multiply': final_score = np.prod(best_scores)
    #if score_combine=='mean':     final_score = np.mean(best_scores)
    final_score = np.prod(best_scores)    

    if unused_factor is not None:
        unused_view_objects=[v for v in view_objects]
        for grounding in best_groundings:
            if grounding is not None:
                if grounding[0] in unused_view_objects: unused_view_objects.remove(grounding[0])                    
                if grounding[2] in unused_view_objects: unused_view_objects.remove(grounding[2])

        final_score*= unused_factor**(len(unused_view_objects))

    return final_score, best_groundings                                  

if __name__=='__main__':
    labels=cv2.imread('data/testim/gt_label.png', cv2.IMREAD_UNCHANGED)[:,:,2]
    depth=cv2.imread('data/testim/depth.png', cv2.IMREAD_UNCHANGED)
    rgb=cv2.imread('data/testim/000000.png')

    view_objects=create_view_objects(rgb, labels, depth)
    print(len(view_objects))


