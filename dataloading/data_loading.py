import cv2
import numpy as np
import os
import pickle
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from scipy.spatial.transform import Rotation

from semantic.imports import ViewObject, SceneGraph, SceneGraphObject, DIRECTIONS

class SynthiaDataset(Dataset):
    def __init__(self, dirpath_main,split_indices=None, transform=None, return_graph_data=False):
        assert os.path.isdir(dirpath_main)

        self.dirpath_main=dirpath_main
        self.transform=transform       
        self.return_graph_data=return_graph_data 
        self.transform=transform

        self.image_paths=[]
        self.image_positions=[] #CARE: y-axis is up/down
        self.image_orientations=[] 
        self.image_viewobjects=[]
        self.image_scenegraphs=[]

        file_names=sorted(os.listdir( os.path.join(dirpath_main,'RGB', 'Stereo_Left', DIRECTIONS[0]) ))

        vo_dict=pickle.load(open( os.path.join(dirpath_main,'view_objects.pkl'),'rb') )
        sg_dict=pickle.load(open( os.path.join(dirpath_main,'scene_graphs.pkl'),'rb') )

        for direction in DIRECTIONS:
            for file_name in file_names:
                self.image_paths.append( os.path.join(dirpath_main, 'RGB', 'Stereo_Left', direction, file_name) )
                assert os.path.isfile(self.image_paths[-1])

                #Position and Orientation
                camera_E=np.fromfile( os.path.join(dirpath_main, 'CameraParams', 'Stereo_Left', direction, file_name.replace('.png','.txt')), sep=" ").reshape((4,4)).T
                self.image_positions.append(camera_E[0:3,3])
                self.image_orientations.append(Rotation.from_matrix(camera_E[0:3,0:3].T).as_euler('xzy')) #TODO/CARE: Orientations faulty!

                #View-Objects and Scene-Graphs
                self.image_viewobjects.append(vo_dict[direction][file_name])
                self.image_scenegraphs.append(sg_dict[direction][file_name])

        self.image_paths=np.array(self.image_paths)
        self.image_positions=np.array(self.image_positions)
        self.image_orientations=np.array(self.image_orientations)
        self.image_viewobjects=np.array(self.image_viewobjects, dtype=np.object)
        self.image_scenegraphs=np.array(self.image_scenegraphs, dtype=np.object)

        if split_indices is not None:
            assert len(split_indices)==len(self.image_paths)
            self.image_paths=self.image_paths[split_indices]
            self.image_positions=self.image_positions[split_indices]
            self.image_orientations=self.image_orientations[split_indices]
            self.image_viewobjects=self.image_viewobjects[split_indices]
            self.image_scenegraphs=self.image_scenegraphs[split_indices]

        assert len(self.image_paths)==len(self.image_positions)==len(self.image_orientations)==len(self.image_viewobjects)==len(self.image_scenegraphs)

        print(f'SynthiaDataset: {len(self.image_paths)} images from {self.dirpath_main}')

    def __len__(self):
        return len(self.image_paths)      

    #Returns the image at the current index
    def __getitem__(self,index):       
        image  =  Image.open(self.image_paths[index]).convert('RGB')     

        if self.transform:
            image = self.transform(image)

        return image          

if __name__=='__main__':
    dawn  =SynthiaDataset('data/SYNTHIA-SEQS-04-DAWN/')
    summer=SynthiaDataset('data/SYNTHIA-SEQS-04-SUMMER/')

    min_dists=[]
    for idx_dawn in range(len(dawn)):
        pos=dawn.image_positions[idx_dawn]
        pos_dists=np.linalg.norm( summer.image_positions - pos, axis=1)
        min_dists.append(np.min(pos_dists))

    print('max of mins:',np.max(min_dists))
