import os
import cv2
from shutil import copyfile

base_dir='data/SYNTHIA-SEQS-04-DAWN'
out_dir=os.path.join(base_dir, 'selection')

image_names=os.listdir( os.path.join(base_dir,'RGB', 'Stereo_Left', 'Omni_F') )

for target in ('RGB','GT/LABELS', 'Depth'):
    for omni in ('Omni_F', 'Omni_B', 'Omni_R', 'Omni_L'):
        os.makedirs( os.path.join(out_dir,target, 'Stereo_Left', omni), exist_ok=True)

        for name in image_names[0:-1:10]:
            copyfile( os.path.join(base_dir,target, 'Stereo_Left', omni,name),
                      os.path.join(out_dir,target, 'Stereo_Left', omni,name) )
