import os
import pickle
import numpy as np
import torch_geometric
import torch
from semantic.imports import SceneGraph, SceneGraphObject

'''
Create the feature dictionaries for vertices and edges
Copies of dictionary are saved in all dataset dirs.
'''
def create_embedding_dictionaries(base_directories):
    #Load all Scene Graphs
    all_description_objects=[]
    for base_dir in base_directories:
        assert os.path.isdir(base_dir)
        assert os.path.isfile( os.path.join(base_dir,'scene_graphs.pkl') )
        do_dict=pickle.load(open(os.path.join(base_dir,'scene_graphs.pkl') , 'rb'))
        for omni in ('Omni_F', 'Omni_B', 'Omni_R', 'Omni_L'):
            all_description_objects.extend( do_dict[omni].values() )

    #Gather all vertex and edge types
    vertex_classes=[]
    # edge_classes=[]
    for description_objects in all_description_objects:
        for obj in description_objects:
            vertex_classes.extend(( obj.label, obj.color, obj.corner, obj.distance  ))
            # vertex_classes.extend(( rel[2].label, rel[2].color, rel[2].corner ))
            # edge_classes.append( rel[1] )
    vertex_classes.append('empty')
    # edge_classes.append('attribute')
    
    vertex_classes=np.unique(vertex_classes)
    # edge_classes=np.unique(edge_classes)
    
    vertex_embedding_dict={}
    # edge_embedding_dict={}

    for i,v in enumerate(vertex_classes):
        vertex_embedding_dict[v]=i

    # for i,e in enumerate(edge_classes):
    #     edge_embedding_dict[e]=i     

    #Save a copy in all directories
    for base_dir in base_directories:
        print(f'Saving in {base_dir}')
        pickle.dump( vertex_embedding_dict, open(os.path.join(base_dir, 'graph_embeddings.pkl'),'wb'))

def create_description_data(description_objects, node_dict):
    node_features=[]
    edges=[]
    # edge_features=[]

    #Encode empty description as: Empty --> Empty
    if len(description_objects)==0:
        node_features.append(node_dict['empty'])
        node_features.append(node_dict['empty'])
        edges.append((0,1))
        # edge_features.append(edge_dict['attribute'])
        edges=np.array(edges).reshape((2,1))
        return torch_geometric.data.Data(x=torch.from_numpy(np.array(node_features,dtype=np.int64)),
                                        edge_index=torch.from_numpy(np.array(edges,dtype=np.int64)))

    for do in description_objects:
        #Node for object
        node_features.append(node_dict[do.label])
        obj_idx=len(node_features)-1        

        #Node and edge for color
        node_features.append(node_dict[do.color])
        edges.append( (len(node_features)-1, obj_idx) )      

        #Node and edge for corner
        node_features.append(node_dict[do.corner])
        edges.append( (len(node_features)-1, obj_idx) ) 

        #Node and edge for distance
        node_features.append(node_dict[do.distance])
        edges.append( (len(node_features)-1, obj_idx) )                   
     
    node_features=np.array(node_features)
    edges=np.array(edges).T #Transpose for PyG-format
    # edge_features=np.array(edge_features)
    assert edges.shape[1]==len(description_objects)*3
    assert len(node_features)==len(description_objects)*4

    #return np.array(node_features), np.array(edges), np.array(edge_features)
    # return torch_geometric.data.Data(x=torch.from_numpy(node_features.astype(np.int64)),
    #                                  edge_index=torch.from_numpy(edges.astype(np.int64)),
    #                                  edge_attr= torch.from_numpy(edge_features.astype(np.float32)))

    return torch_geometric.data.Data(x=torch.from_numpy(node_features.astype(np.int64)),
                                     edge_index=torch.from_numpy(edges.astype(np.int64)))

if __name__ == "__main__":
    # directories=('data/SYNTHIA-SEQS-04-SUMMER/train', 'data/SYNTHIA-SEQS-04-SUMMER/test/', 'data/SYNTHIA-SEQS-04-SUMMER/full', 'data/SYNTHIA-SEQS-04-SUMMER/dense/',
    #              'data/SYNTHIA-SEQS-04-DAWN/train', 'data/SYNTHIA-SEQS-04-DAWN/test/', 'data/SYNTHIA-SEQS-04-DAWN/full/', 
    #              'data/SYNTHIA-SEQS-04-WINTER/train', 'data/SYNTHIA-SEQS-04-WINTER/test/', 'data/SYNTHIA-SEQS-04-WINTER/full/')
    directories=('data/SYNTHIA-SEQS-04-SUMMER/dense', 'data/SYNTHIA-SEQS-04-SUMMER/test/',
                 'data/SYNTHIA-SEQS-04-DAWN/test')    
    
    create_embedding_dictionaries(directories)