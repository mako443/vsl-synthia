import numpy as np
import os
import pickle
import sys

import torch
from torchvision import transforms
from torch_geometric.data import DataLoader

from evaluation.evaluation_functions import eval_featureVectors_scoresDict, eval_featureVectors, print_topK
from dataloading.data_loading import SynthiaDataset
from visualgeometric.geometric_embedding import GeometricEmbedding
from retrieval.netvlad import NetvladModel
from visualgeometric.visual_geometric_embedding import VisualGraphEmbeddingCombined, VisualGraphEmbeddingCombinedPT, create_image_model_resnet_18

def gather_GE_vectors(loader, model):
    #Gather all features
    print('Building GE vectors:', loader.dataset.scene_name)
    embed_vectors=torch.tensor([]).cuda()
    with torch.no_grad():
        for i_batch, batch in enumerate(loader):
            a=batch['graphs']
            a_out=model(a.to('cuda'))
            embed_vectors=torch.cat((embed_vectors,a_out))   
    embed_vectors=embed_vectors.cpu().detach().numpy()
    embed_dim=embed_vectors.shape[1]

    pickle.dump(embed_vectors, open(f'features_GE_e{embed_dim}_d{loader.dataset.scene_name}.pkl','wb'))
    print('Saved GE-vectors')

def randomize_graphs(graph_batch):
    t=graph_batch.x.dtype
    graph_batch.x=torch.randint_like(graph_batch.x.type(torch.float), low=0, high=20).type(t)

    t=graph_batch.edge_attr.dtype
    graph_batch.edge_attr=torch.randint_like(graph_batch.edge_attr, low=0, high=4).type(t)

    edge_index_clone=graph_batch.edge_index.clone().detach()
    graph_batch.edge_index[0,:]=edge_index_clone[1,:]
    graph_batch.edge_index[1,:]=edge_index_clone[0,:]    
    return graph_batch

def gather_VGE_CO_vectors(loader, model, model_name, use_random_graphs=False):
    #Gather all features
    print('Building VGE-CO vectors:', loader.dataset.scene_name)
    embed_vectors=torch.tensor([]).cuda()
    with torch.no_grad():
        for i_batch, batch in enumerate(loader):
            images, graphs=batch['images'], batch['graphs']
            if use_random_graphs: 
                graphs=randomize_graphs(graphs)            
            a_out=model(images.to('cuda'), graphs.to('cuda'))
            embed_vectors=torch.cat((embed_vectors,a_out))   
    embed_vectors=embed_vectors.cpu().detach().numpy()
    embed_dim=embed_vectors.shape[1]

    pickle.dump(embed_vectors, open(f'features_VGE-CO_m{model_name}_d{loader.dataset.scene_name}_rg{use_random_graphs}.pkl','wb'))
    print('Saved VGE-CO-vectors')

if __name__ == "__main__":
    transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),])

    # data_summer_train=SynthiaDataset('data/SYNTHIA-SEQS-04-SUMMER/train', transform=transform, return_graph_data=True)
    data_summer_test =SynthiaDataset('data/SYNTHIA-SEQS-04-SUMMER/test', transform=transform, return_graph_data=True)
    data_summer_dense =SynthiaDataset('data/SYNTHIA-SEQS-04-SUMMER/dense', transform=transform, return_graph_data=True)
    # data_dawn_train=  SynthiaDataset('data/SYNTHIA-SEQS-04-DAWN/train', transform=transform, return_graph_data=True)
    data_dawn_test =  SynthiaDataset('data/SYNTHIA-SEQS-04-DAWN/test', transform=transform, return_graph_data=True)
    # data_winter_train=SynthiaDataset('data/SYNTHIA-SEQS-04-WINTER/train', transform=transform, return_graph_data=True)
    # data_winter_test =SynthiaDataset('data/SYNTHIA-SEQS-04-WINTER/test', transform=transform, return_graph_data=True)

    #TODO: sort by model
    if 'gather-summer' in sys.argv:
        pass
        
        #GE
        # EMBED_DIM_GEOMETRIC=512
        # geometric_embedding=GeometricEmbedding(EMBED_DIM_GEOMETRIC)
        # geometric_embedding_model_name='model_GeometricEmbed_lNone_dSUMMER_b12_g0.75_e512_lTML_m0.5_lr0.0005.pth'
        # geometric_embedding.load_state_dict(torch.load('models/'+geometric_embedding_model_name)); print('Using model:',geometric_embedding_model_name)
        # geometric_embedding.eval(); geometric_embedding.cuda()
        
        # loader=DataLoader(data_summer_train, batch_size=BATCH_SIZE, num_workers=2, pin_memory=True, shuffle=False) 
        # gather_GE_vectors(loader, geometric_embedding)

        # loader=DataLoader(data_summer_test , batch_size=BATCH_SIZE, num_workers=2, pin_memory=True, shuffle=False) 
        # gather_GE_vectors(loader, geometric_embedding)  

        # #VGE-CO
        # EMBED_DIM_GEOMETRIC=1024               
        # resnet=create_image_model_resnet_18()
        # vge_co_model=VisualGraphEmbeddingCombined(resnet, EMBED_DIM_GEOMETRIC)
        # vge_co_model_name='model_VGE-NV-CO_lNone_b12_g0.75_e1024_sTrue_m0.5_dsummer_lr0.01.pth'
        # vge_co_model.load_state_dict(torch.load('models/'+vge_co_model_name)); print('Model:',vge_co_model_name)
        # vge_co_model.eval()
        # vge_co_model.cuda()
        
        # loader=DataLoader(data_summer_train, batch_size=BATCH_SIZE, num_workers=2, pin_memory=True, shuffle=False) 
        # gather_VGE_CO_vectors(loader, vge_co_model, "VGE-NV-CO-summer")

        # loader=DataLoader(data_summer_test , batch_size=BATCH_SIZE, num_workers=2, pin_memory=True, shuffle=False)    
        # gather_VGE_CO_vectors(loader, vge_co_model, "VGE-NV-CO-summer")

        # loader=DataLoader(data_summer_dense , batch_size=BATCH_SIZE, num_workers=2, pin_memory=True, shuffle=False)    
        # gather_VGE_CO_vectors(loader, vge_co_model, "VGE-NV-CO-summer")  

    if 'gather-GE' in sys.argv:
        BATCH_SIZE=12
        EMBED_DIM_GEOMETRIC=100
        geometric_embedding=GeometricEmbedding(EMBED_DIM_GEOMETRIC)
        geometric_embedding_model_name='model_GeometricEmbed-TripReal_lNone_dSUMMER_b32_g0.75_e100_m0.5_lr0.001.pth'
        geometric_embedding.load_state_dict(torch.load('models/'+geometric_embedding_model_name)); print('Using model:',geometric_embedding_model_name)
        geometric_embedding.eval(); geometric_embedding.cuda()
        
        loader=DataLoader(data_summer_dense, batch_size=BATCH_SIZE, num_workers=2, pin_memory=True, shuffle=False) 
        gather_GE_vectors(loader, geometric_embedding)

        loader=DataLoader(data_summer_test , batch_size=BATCH_SIZE, num_workers=2, pin_memory=True, shuffle=False) 
        gather_GE_vectors(loader, geometric_embedding)      

        loader=DataLoader(data_dawn_test , batch_size=BATCH_SIZE, num_workers=2, pin_memory=True, shuffle=False) 
        gather_GE_vectors(loader, geometric_embedding)                   

    if 'eval-GE' in sys.argv:
        features_name_db   ='features_GE_e100_dSUMMER-dense.pkl'
        features_name_query='features_GE_e100_dSUMMER-test.pkl'
        features_db, features_query=pickle.load(open('evaluation_res/'+features_name_db, 'rb')), pickle.load(open('evaluation_res/'+features_name_query, 'rb')); print('features:',features_name_db, features_name_query)
        thresh_results=eval_featureVectors(data_summer_dense, data_summer_test, features_db, features_query, similarity='l2')
        print_topK(thresh_results)        

        features_name_db   ='features_GE_e100_dSUMMER-dense.pkl'
        features_name_query='features_GE_e100_dDAWN-test.pkl'
        features_db, features_query=pickle.load(open('evaluation_res/'+features_name_db, 'rb')), pickle.load(open('evaluation_res/'+features_name_query, 'rb')); print('features:',features_name_db, features_name_query)
        thresh_results=eval_featureVectors(data_summer_dense, data_dawn_test, features_db, features_query, similarity='l2')
        print_topK(thresh_results)        

    if 'eval-VGE-CO' in sys.argv:      
        features_name_db   ='features_VGE-CO_mVGE-NV-CO-summer_dSUMMER-dense.pkl'
        features_name_query='features_VGE-CO_mVGE-NV-CO-summer_dSUMMER-test.pkl'
        features_db, features_query=pickle.load(open('evaluation_res/'+features_name_db, 'rb')), pickle.load(open('evaluation_res/'+features_name_query, 'rb')); print('features:',features_name_db, features_name_query)

        pos_results, ori_results=eval_featureVectors(data_summer_dense, data_summer_test, features_db, features_query, similarity='l2')
        print(pos_results, ori_results,'\n')   
                       