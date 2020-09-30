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
    def __init__(self, dirpath_main,split_indices=None,transform=None, return_graph_data=False):
        assert os.path.isdir(dirpath_main)

        self.dirpath_main=dirpath_main
        self.transform=transform       
        self.return_graph_data=return_graph_data 

        self.image_paths=[]
        self.image_positions=[] #CARE: y-axis is up/down
        self.image_orientations=[] 
        self.image_viewobjects=[]
        self.image_scenegraphs=[]

        file_names=sorted(os.listdir( os.path.join(dirpath_main,'RGB', 'Stereo_Left', DIRECTIONS[0]) ))

        for direction in DIRECTIONS:
            for file_name in file_names:
                self.image_paths.append( os.path.join(dirpath_main, 'RGB', 'Stereo_Left', direction, file_name) )
                assert os.path.isfile(self.image_paths[-1])

                #print('file',os.path.join(dirpath_main, 'CameraParams', 'Stereo_Left', direction, file_name.replace('.png','.txt')))
                camera_E=np.fromfile( os.path.join(dirpath_main, 'CameraParams', 'Stereo_Left', direction, file_name.replace('.png','.txt')), sep=" ").reshape((4,4)).T
                self.image_positions.append(camera_E[0:3,3])
                self.image_orientations.append(Rotation.from_matrix(camera_E[0:3,0:3].T).as_euler('xzy')) #TODO/CARE: Orientations faulty!

        self.image_paths=np.array(self.image_paths)
        self.image_positions=np.array(self.image_positions)
        #self.image_orientations=np.array(self.image_orientations)

        if split_indices is not None:
            assert len(split_indices)==len(self.image_paths)

        #assert len(self.image_paths)==len(self.image_positions)==len(self.image_orientations)

        print(f'SynthiaDataset: {len(self.image_paths)} images from {self.dirpath_main}')

if __name__=='__main__':
    #Top 3x3 is rotation matrix -.-
#     E=np.array([[-1.3745e-04, -1.2382e-01,  9.9230e-01,  4.3268e+01],
#                 [ 9.8214e-04,  9.9230e-01,  1.2382e-01,  3.1057e+00],
#                 [ 1.0000e+00, -9.9161e-04,  1.4784e-05,  1.1755e+02],
#                 [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  1.0000e+00]]
# )
#     quit()

    # E0=np.fromfile( os.path.join('data/SYNTHIA-SEQS-04-SUMMER/selection', 'CameraParams', 'Stereo_Left', 'Omni_F', '000004.txt'), sep=" ").reshape((4,4)).T
    # E1=np.fromfile( os.path.join('data/SYNTHIA-SEQS-04-SUMMER/selection', 'CameraParams', 'Stereo_Left', 'Omni_L', '000004.txt'), sep=" ").reshape((4,4)).T
    # E2=np.fromfile( os.path.join('data/SYNTHIA-SEQS-04-SUMMER/selection', 'CameraParams', 'Stereo_Left', 'Omni_B', '000004.txt'), sep=" ").reshape((4,4)).T

    # quit()

    dawn  =SynthiaDataset('data/SYNTHIA-SEQS-04-DAWN/selection')
    summer=SynthiaDataset('data/SYNTHIA-SEQS-04-SUMMER/selection')
