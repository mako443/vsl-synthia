import cv2
import numpy as np
import os
import pickle
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from scipy.spatial.transform import Rotation

from semantic.imports import ViewObject, SceneGraph, SceneGraphObject, DIRECTIONS
from visualgeometric.utils import create_scenegraph_data

def wrap_angle(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi

class SynthiaDataset(Dataset):
    def __init__(self, dirpath_main,transform=None, return_graph_data=False, load_netvlad_features=False, image_limit=None):
        assert os.path.isdir(dirpath_main)

        self.dirpath_main=dirpath_main
        self.transform=transform       
        self.return_graph_data=return_graph_data 
        self.transform=transform
        self.image_limit=image_limit

        self.scene_name=dirpath_main.split('-')[-1].replace('/','-')
        assert len(self.scene_name)>0

        self.image_paths=[]
        self.image_positions=[] #CARE: y-axis is up/down
        self.image_omnis=[]
        self.image_orientations=[] 
        self.image_viewobjects=[]
        self.image_scenegraphs=[]

        file_names=sorted(os.listdir( os.path.join(dirpath_main,'RGB', 'Stereo_Left', DIRECTIONS[0]) ))

        vo_dict=pickle.load(open( os.path.join(dirpath_main,'view_objects.pkl'),'rb') )
        sg_dict=pickle.load(open( os.path.join(dirpath_main,'scene_graphs.pkl'),'rb') )

        #TODO: assume naive 90 deg. angles, anchored at F (re-varify across turn and cross-scene)
        idx=0
        for direction in DIRECTIONS:
            for file_name in file_names:
                self.image_paths.append( os.path.join(dirpath_main, 'RGB', 'Stereo_Left', direction, file_name) )
                assert os.path.isfile(self.image_paths[-1])

                #Image position
                camera_E=np.fromfile( os.path.join(dirpath_main, 'CameraParams', 'Stereo_Left', direction, file_name.replace('.png','.txt')), sep=" ").reshape((4,4)).T
                self.image_positions.append(camera_E[0:3,3])

                #Image orientation: extracting from E matrix was incoherent (?), using forward direction as anchor and -90°, +90°, +180° for left, right, backwards respectively
                camera_E=np.fromfile( os.path.join(dirpath_main, 'CameraParams', 'Stereo_Left', 'Omni_B', file_name.replace('.png','.txt')), sep=" ").reshape((4,4)).T #Load from <Omni_F>
                orientation=Rotation.from_matrix(camera_E[0:3,0:3]).as_euler('xyz')[0] #TODO: transpose/order correct?
                
                self.image_orientations.append(orientation) #TODO: wrap?
                if idx in (11,14):
                    print(self.image_paths[idx]); print(self.image_positions[idx]); print(self.image_orientations[idx])

                if   direction=='Omni_L': orientation-= np.pi/2
                elif direction=='Omni_R': orientation+= np.pi/2
                elif direction=='Omni_B': orientation+= np.pi

                #View-Objects and Scene-Graphs
                self.image_viewobjects.append(vo_dict[direction][file_name])
                self.image_scenegraphs.append(sg_dict[direction][file_name])

                self.image_omnis.append(direction)

                idx+=1

        self.image_paths=np.array(self.image_paths)
        self.image_positions=np.array(self.image_positions)
        self.image_omnis=np.array(self.image_omnis)
        self.image_orientations=np.array(self.image_orientations)
        self.image_viewobjects=np.array(self.image_viewobjects, dtype=np.object)
        self.image_scenegraphs=np.array(self.image_scenegraphs, dtype=np.object)

        assert len(self.image_paths)==len(self.image_positions)==len(self.image_omnis)==len(self.image_orientations)==len(self.image_viewobjects)==len(self.image_scenegraphs)

        if load_netvlad_features:
            assert os.path.isfile( os.path.join(dirpath_main,'netvlad_features.pkl') )
            self.image_netvlad_features=pickle.load( open(os.path.join(dirpath_main,'netvlad_features.pkl'), 'rb') )
            assert len(self.image_netvlad_features)==len(self.image_paths)

        if return_graph_data:
            assert os.path.isfile( os.path.join(dirpath_main,'graph_embeddings.pkl') )
            self.node_embeddings, self.edge_embeddings=pickle.load(open(os.path.join(dirpath_main,'graph_embeddings.pkl'), 'rb'))
            self.image_scenegraph_data=[ create_scenegraph_data(sg, self.node_embeddings, self.edge_embeddings) for sg in self.image_scenegraphs ]
            assert len(self.image_scenegraph_data)==len(self.image_scenegraphs)

            empty_graphs=[1 for sg in self.image_scenegraphs if sg.is_empty()]
            #print(f'Empty Graphs: {np.sum(empty_graphs)} of {len(self.image_positions)}')            

        print(f'SynthiaDataset: {self.scene_name}, {len(self.image_paths)} images from {self.dirpath_main}')

    def __len__(self):
        if self.image_limit is not None:
            return min(len(self.image_paths), self.image_limit)
        else:
            return len(self.image_paths)      

    #Returns the image at the current index
    def __getitem__(self,index):       
        image  =  Image.open(self.image_paths[index]).convert('RGB')     

        if self.transform:
            image = self.transform(image)

        if self.return_graph_data:
            return {'images': image, 'graphs': self.image_scenegraph_data[index]}

        return image    

class SynthiaDatasetTriplet(SynthiaDataset):
    def __init__(self, dirpath_main, transform=None, return_graph_data=False, load_netvlad_features=False, image_limit=None): 
        super().__init__(dirpath_main, transform=transform, return_graph_data=return_graph_data,load_netvlad_features=load_netvlad_features, image_limit=image_limit)
        #TODO/better: based on location and angle | needs to resolve angle bugs?
        
        #CARE: Thresholds currently in steps between images, calculated from the full dataset -> independent of the split!
        self.positive_thresh=5 #Maximum of 5 pictures before/after
        self.negative_thresh=60 #Minimum of 60 pictures before/after
        self.image_frame_indices=np.array([ int(path.split("/")[-1].split(".")[0]) for path in self.image_paths ])

    def __getitem__(self,anchor_index):
        anchor_frame_index=self.image_frame_indices[anchor_index]
        anchor_omni=self.image_omnis[anchor_index]

        #Positive is from same omni
        positive_indices= np.argwhere( np.bitwise_and(self.image_omnis==anchor_omni, np.bitwise_and(self.image_frame_indices>anchor_frame_index-self.positive_thresh, self.image_frame_indices<anchor_frame_index+self.positive_thresh)))
        assert len(positive_indices)>0
        positive_index=np.random.choice(positive_indices.flatten())

        #Negative is from same or other omni
        negative_indices= np.argwhere( np.bitwise_or(self.image_frame_indices>anchor_frame_index+self.negative_thresh, self.image_frame_indices<anchor_frame_index-self.negative_thresh))
        assert len(negative_indices)>0
        negative_index=np.random.choice(negative_indices.flatten())    

        #print('Indices: ', anchor_index, positive_index, negative_index)    

        anchor_image  =  Image.open(self.image_paths[anchor_index]).convert('RGB')     
        positive_image=  Image.open(self.image_paths[positive_index]).convert('RGB')     
        negative_image=  Image.open(self.image_paths[negative_index]).convert('RGB')  

        if self.transform:
            anchor_image= self.transform(anchor_image)
            positive_image= self.transform(positive_image)
            negative_image= self.transform(negative_image)

        if self.return_graph_data:
            return {'images_anchor':anchor_image, 'images_positive':positive_image, 'images_negative':negative_image,
                    'graphs_anchor':self.image_scenegraph_data[anchor_index], 'graphs_positive':self.image_scenegraph_data[positive_index], 'graphs_negative':self.image_scenegraph_data[negative_index]}

        return anchor_image, positive_image, negative_image

#TODO
class SynthiaDatasetMulti(SynthiaDataset):
    pass

#TODO
class SynthiaDatasetMultiTriplet(SynthiaDataset):
    pass


if __name__=='__main__':
    summer=SynthiaDataset('data/SYNTHIA-SEQS-04-SUMMER/selection', return_graph_data=False)
    quit()
    idx=11
    print(summer.image_paths[idx]); print(summer.image_positions[idx]); print(summer.image_orientations[idx])
    idx=14
    print(summer.image_paths[idx]); print(summer.image_positions[idx]); print(summer.image_orientations[idx])


    summer_triplet=SynthiaDatasetTriplet('data/SYNTHIA-SEQS-04-SUMMER/full')
    quit()
    dawn  =SynthiaDataset('data/SYNTHIA-SEQS-04-DAWN/')


    min_dists=[]
    for idx_dawn in range(len(dawn)):
        pos=dawn.image_positions[idx_dawn]
        pos_dists=np.linalg.norm( summer.image_positions - pos, axis=1)
        min_dists.append(np.min(pos_dists))

    print('max of mins:',np.max(min_dists))

-5.5249e-01, -9.9198e-02,  8.2759e-01,  9.9645e+00
2.9749e-03,  9.9265e-01,  1.2097e-01,  3.1012e+00
8.3351e-01, -6.9296e-02,  5.4814e-01,  1.1628e+02
0.0000e+00,  0.0000e+00,  0.0000e+00,  1.0000e+00