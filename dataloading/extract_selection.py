import os
import cv2
from shutil import copyfile

#base_dir='data/SYNTHIA-SEQS-04-DAWN'
for base_dir in ('data/SYNTHIA-SEQS-04-SUMMER/', 'data/SYNTHIA-SEQS-04-DAWN/', 'data/SYNTHIA-SEQS-04-WINTER/'):
    out_dir=os.path.join(base_dir, 'selection')
    print(f'from {base_dir} to {out_dir}')

    image_names=sorted(os.listdir( os.path.join(base_dir,'RGB', 'Stereo_Left', 'Omni_F') ))

    for target in ('RGB','GT/LABELS', 'Depth', 'CameraParams'):
        for omni in ('Omni_F', 'Omni_B', 'Omni_R', 'Omni_L'):
            os.makedirs( os.path.join(out_dir,target, 'Stereo_Left', omni), exist_ok=True)

            for name in image_names[0:10] + image_names[10:-1:10]:
                if target=='CameraParams': 
                    name=name.replace('.png','.txt')

                copyfile( os.path.join(base_dir,target, 'Stereo_Left', omni,name),
                          os.path.join(out_dir,target, 'Stereo_Left', omni,name) )
