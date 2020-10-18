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
    all_scene_graphs=[]
    for base_dir in base_directories:
        assert os.path.isdir(base_dir)
        assert os.path.isfile( os.path.join(base_dir,'scene_graphs.pkl') )
        graph_dict=pickle.load(open(os.path.join(base_dir,'scene_graphs.pkl') , 'rb'))
        for omni in ('Omni_F', 'Omni_B', 'Omni_R', 'Omni_L'):
            all_scene_graphs.extend( graph_dict[omni].values() )

    #Gather all vertex and edge types
    vertex_classes=[]
    edge_classes=[]
    for scene_graph in all_scene_graphs:
        for rel in scene_graph.relationships:
            vertex_classes.extend(( rel[0].label, rel[0].color, rel[0].corner ))
            vertex_classes.extend(( rel[2].label, rel[2].color, rel[2].corner ))
            edge_classes.append( rel[1] )
    vertex_classes.append('empty')
    edge_classes.append('attribute')
    
    vertex_classes=np.unique(vertex_classes)
    edge_classes=np.unique(edge_classes)
    
    vertex_embedding_dict={}
    edge_embedding_dict={}

    for i,v in enumerate(vertex_classes):
        vertex_embedding_dict[v]=i

    for i,e in enumerate(edge_classes):
        edge_embedding_dict[e]=i     

    #Save a copy in all directories
    for base_dir in base_directories:
        print(f'Saving in {base_dir}')
        pickle.dump( (vertex_embedding_dict, edge_embedding_dict), open(os.path.join(base_dir, 'graph_embeddings.pkl'),'wb'))

def create_scenegraph_data(scene_graph, node_dict, edge_dict):
    node_features=[]
    edges=[]
    edge_features=[]

    #Encode empty SG as: Empty ---attribute---> Empty
    if scene_graph.is_empty():
        node_features.append(node_dict['empty'])
        node_features.append(node_dict['empty'])
        edges.append((0,1))
        edge_features.append(edge_dict['attribute'])
        edges=np.array(edges).reshape((2,1))
        return torch_geometric.data.Data(x=torch.from_numpy(np.array(node_features,dtype=np.int64)),
                                        edge_index=torch.from_numpy(np.array(edges,dtype=np.int64)),
                                        edge_attr= torch.from_numpy(np.array(edge_features,dtype=np.float32)))

    for rel in scene_graph.relationships:
        sub, rel_type, obj=rel

        #Node for the subject
        node_features.append(node_dict[sub.label])
        sub_idx=len(node_features)-1

        #Node for the object
        node_features.append(node_dict[obj.label])
        obj_idx=len(node_features)-1

        #The relationship edge
        edges.append( (sub_idx,obj_idx) ) 
        edge_features.append( edge_dict[rel_type] )

        # Color and Corner for subject
        node_features.append(node_dict[sub.color])
        edges.append( (len(node_features)-1, sub_idx) )
        edge_features.append(edge_dict['attribute'])
        node_features.append(node_dict[sub.corner])
        edges.append( (len(node_features)-1, sub_idx) )
        edge_features.append(edge_dict['attribute'])

        # Color and Corner for object
        node_features.append(node_dict[obj.color])
        edges.append( (len(node_features)-1, obj_idx) )
        edge_features.append(edge_dict['attribute'])
        node_features.append(node_dict[obj.corner])
        edges.append( (len(node_features)-1, obj_idx) )
        edge_features.append(edge_dict['attribute'])        

    node_features=np.array(node_features)
    edges=np.array(edges).T #Transpose for PyG-format
    edge_features=np.array(edge_features)
    assert len(edge_features)==edges.shape[1]== len(scene_graph.relationships) * (2*2+1)
    assert len(node_features)==len(scene_graph.relationships) * (2*3)

    #return np.array(node_features), np.array(edges), np.array(edge_features)
    return torch_geometric.data.Data(x=torch.from_numpy(node_features.astype(np.int64)),
                                     edge_index=torch.from_numpy(edges.astype(np.int64)),
                                     edge_attr= torch.from_numpy(edge_features.astype(np.float32)))

if __name__ == "__main__":
    directories=('data/SYNTHIA-SEQS-04-SUMMER/train', 'data/SYNTHIA-SEQS-04-SUMMER/test/', 
                 'data/SYNTHIA-SEQS-04-DAWN/train', 'data/SYNTHIA-SEQS-04-DAWN/test/', 
                 'data/SYNTHIA-SEQS-04-WINTER/train', 'data/SYNTHIA-SEQS-04-WINTER/test/')
    
    create_embedding_dictionaries(directories)