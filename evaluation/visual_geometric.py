import numpy as np
import os
import pickle
import sys

import torch
from torchvision import transforms
from torch_geometric.data import DataLoader

from evaluation.evaluation_functions import eval_featureVectors_scoresDict, eval_featureVectors
from dataloading.data_loading import SynthiaDataset
from visualgeometric.geometric_embedding import GeometricEmbedding

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

if __name__ == "__main__":
    transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),])

    data_summer_train=SynthiaDataset('data/SYNTHIA-SEQS-04-SUMMER/train', transform=transform, return_graph_data=True)
    data_summer_test =SynthiaDataset('data/SYNTHIA-SEQS-04-SUMMER/test', transform=transform, return_graph_data=True)
    data_dawn_train=  SynthiaDataset('data/SYNTHIA-SEQS-04-DAWN/train', transform=transform, return_graph_data=True)
    data_dawn_test =  SynthiaDataset('data/SYNTHIA-SEQS-04-DAWN/test', transform=transform, return_graph_data=True)
    data_winter_train=SynthiaDataset('data/SYNTHIA-SEQS-04-WINTER/train', transform=transform, return_graph_data=True)
    data_winter_test =SynthiaDataset('data/SYNTHIA-SEQS-04-WINTER/test', transform=transform, return_graph_data=True)

    if 'gather-summer' in sys.argv:
        BATCH_SIZE=12
        EMBED_DIM_GEOMETRIC=512
        
        geometric_embedding=GeometricEmbedding(EMBED_DIM_GEOMETRIC)
        geometric_embedding_model_name='model_GeometricEmbed_lNone_dSUMMER_b12_g0.75_e512_lTML_m0.5_lr0.0005.pth'
        geometric_embedding.load_state_dict(torch.load('models/'+geometric_embedding_model_name)); print('Using model:',geometric_embedding_model_name)
        geometric_embedding.eval(); geometric_embedding.cuda()
        
        loader=DataLoader(data_summer_train, batch_size=BATCH_SIZE, num_workers=2, pin_memory=True, shuffle=False) 
        gather_GE_vectors(loader, geometric_embedding)

        loader=DataLoader(data_summer_test , batch_size=BATCH_SIZE, num_workers=2, pin_memory=True, shuffle=False) 
        gather_GE_vectors(loader, geometric_embedding)  

    if 'GE-match' in sys.argv:
        features_name_db   ='features_GE_e512_dSUMMER-train.pkl'
        features_name_query='features_GE_e512_dSUMMER-test.pkl'
        features_db, features_query=pickle.load(open('evaluation_res/'+features_name_db, 'rb')), pickle.load(open('evaluation_res/'+features_name_query, 'rb')); print('features:',features_name_db, features_name_query)

        pos_results, ori_results=eval_featureVectors(data_summer_train, data_summer_test, features_db, features_query, similarity='l2')
        print(pos_results, ori_results,'\n')      