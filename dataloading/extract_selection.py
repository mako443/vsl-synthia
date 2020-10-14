import os
import cv2
from shutil import copyfile

# #base_dir='data/SYNTHIA-SEQS-04-DAWN'
# for base_dir in ('data/SYNTHIA-SEQS-04-SUMMER/', 'data/SYNTHIA-SEQS-04-DAWN/', 'data/SYNTHIA-SEQS-04-WINTER/'):
#     out_dir=os.path.join(base_dir, 'selection')
#     print(f'from {base_dir} to {out_dir}')

#     image_names=sorted(os.listdir( os.path.join(base_dir,'RGB', 'Stereo_Left', 'Omni_F') ))

#     for target in ('RGB','GT/LABELS', 'Depth', 'CameraParams'):
#         for omni in ('Omni_F', 'Omni_B', 'Omni_R', 'Omni_L'):
#             os.makedirs( os.path.join(out_dir,target, 'Stereo_Left', omni), exist_ok=True)

#             for name in image_names[0:10] + image_names[10:-1:10]:
#                 if target=='CameraParams': 
#                     name=name.replace('.png','.txt')

#                 copyfile( os.path.join(base_dir,target, 'Stereo_Left', omni,name),
#                           os.path.join(out_dir,target, 'Stereo_Left', omni,name) )

def copy_split(dirpath_in, dirpath_out, offset, step):
    assert os.path.isdir( os.path.join(dirpath_in,'RGB') ) and os.path.isdir( os.path.join(dirpath_in,'GT/LABELS') ) and os.path.isdir( os.path.join(dirpath_in,'Depth') ) and os.path.isdir( os.path.join(dirpath_in,'CameraParams') )
    os.makedirs( dirpath_out, exist_ok=True)    

    print(f'from {dirpath_in} to {dirpath_out} offset {offset} step {step}')

    image_names=sorted(os.listdir( os.path.join(dirpath_in,'RGB', 'Stereo_Left', 'Omni_F') ))

    for target in ('RGB','GT/LABELS', 'Depth', 'CameraParams'):
        print(f'target <{target}>')
        for omni in ('Omni_F', 'Omni_B', 'Omni_R', 'Omni_L'):
            os.makedirs( os.path.join(dirpath_out,target, 'Stereo_Left', omni), exist_ok=False)    

            for name in image_names[offset::step]:
                if target=='CameraParams': 
                    name=name.replace('.png','.txt')   

                copyfile( os.path.join(dirpath_in ,target, 'Stereo_Left', omni,name),
                          os.path.join(dirpath_out,target, 'Stereo_Left', omni,name) )  

for base_dir in ('data/SYNTHIA-SEQS-04-SUMMER/', 'data/SYNTHIA-SEQS-04-DAWN/', 'data/SYNTHIA-SEQS-04-WINTER/'):
    dirpath_in = os.path.join(base_dir, 'full')

    dirpath_out= os.path.join(base_dir, 'train')
    offset, step=0, 4
    copy_split(dirpath_in, dirpath_out, offset, step)

    dirpath_out= os.path.join(base_dir, 'test')
    offset, step=2, 8
    copy_split(dirpath_in, dirpath_out, offset, step)    