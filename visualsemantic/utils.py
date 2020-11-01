import os
import pickle
import numpy as np

from semantic.imports import ViewObject, SceneGraph, SceneGraphObject, draw_scenegraph_on_image, DescriptionObject
'''
Create a dictionary of all words.
Copies of dictionary are saved in all dataset dirs.
'''
def create_known_word_dicts(base_directories):
    #Load all Scene Graphs
    scene_graphs=[]
    for base_dir in base_directories:
        assert os.path.isdir(base_dir)
        assert os.path.isfile( os.path.join(base_dir,'scene_graphs.pkl') )
        do_dict=pickle.load(open(os.path.join(base_dir,'scene_graphs.pkl') , 'rb'))
        for omni in ('Omni_F', 'Omni_B', 'Omni_R', 'Omni_L'):
            scene_graphs.extend( do_dict[omni].values() )

    known_words={}
    for sg in scene_graphs:
        cap=DescriptionObject.generate_caption(sg)
        for word in cap.split(' '):
            if word not in known_words:
                known_words[word]=len(known_words)  

    print(known_words)

    #Save a copy in all directories
    for base_dir in base_directories:
        print(f'Saving in {base_dir}')
        pickle.dump( known_words, open(os.path.join(base_dir, 'known_words.pkl'),'wb'))

if __name__=='__main__':
    directories=('data/SYNTHIA-SEQS-04-SUMMER/dense',)

    create_known_word_dicts(directories)
