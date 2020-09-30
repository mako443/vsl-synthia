import numpy as np
import cv2
import os
import random
import pickle

from semantic.imports import ViewObject, CLASS_IDS, CAMERA_I, CAMERA_I_INV
import semantic.scenegraphs

'''
Create the View-Objects from the patches in the image
'''
#TODO: split patches along depth-gaps?
def create_view_objects(rgb_image, label_image, depth_image):
    view_objects=[]
    depth_image=depth_image.astype(np.float32)/100

    for label, label_id in CLASS_IDS.items():
        #if label not in ('Fence','Pole'):continue
        label_mask= np.uint8(label_image==label_id)

        cc_retval, cc_labels, cc_stats, cc_centroids = cv2.connectedComponentsWithStats(label_mask)

        for i in range(1, len(cc_centroids)):
            if cc_stats[i,cv2.CC_STAT_AREA]< 40*40: continue #OPTIONS: min-area

            object_mask= cc_labels==i
            mindepth, maxdepth= depth_image[object_mask].min(), depth_image[object_mask].max()
            object_color=np.mean( rgb_image[object_mask,:], axis=0 )

            centroid_i=np.array((cc_centroids[i,0],cc_centroids[i,1],np.array(np.mean(depth_image[object_mask]))))
            centroid_c= CAMERA_I_INV @ np.array(( centroid_i[0]*centroid_i[2], centroid_i[1]*centroid_i[2], centroid_i[2] ))
            centroid_c[1]*=-1
            #print(centroid_i, centroid_c)

            bbox=( cc_stats[i, 0], cc_stats[i, 1], mindepth, cc_stats[i, 0]+cc_stats[i, 2], cc_stats[i, 1]+cc_stats[i, 3], maxdepth )
            view_objects.append(ViewObject( label, bbox, centroid_i, centroid_c, object_color))

    return view_objects
                

if __name__=='__main__':
    labels=cv2.imread('data/testim/gt_label.png', cv2.IMREAD_UNCHANGED)[:,:,2]
    depth=cv2.imread('data/testim/depth.png', cv2.IMREAD_UNCHANGED)
    rgb=cv2.imread('data/testim/000000.png')

    view_objects=create_view_objects(rgb, labels, depth)
    print(len(view_objects))

    for v in view_objects:
        #if v.label!='Building':continue
        v.draw_on_image(rgb)
        #print(v)
    cv2.imshow("",rgb)
    cv2.waitKey()