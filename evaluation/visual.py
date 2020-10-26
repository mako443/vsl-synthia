import numpy as np
import os
import pickle
import sys

import torch
from torchvision import transforms
from torch_geometric.data import DataLoader #Use the PyG DataLoader

from evaluation.evaluation_functions import eval_featureVectors, get_split_indices
from dataloading.data_loading import SynthiaDataset
from retrieval.netvlad import NetvladModel, NetvladFCModel
from visualgeometric.visual_geometric_embedding import create_image_model_resnet_18

#CARE: NV-Pitts feature-extraction is done in 'pytorch-NetVlad' sub-directory.

def gather_NV_vectors(loader, model, model_name):
    #Gather all features
    print('Building VGE-CO vectors:', loader.dataset.scene_name)
    embed_vectors=torch.tensor([]).cuda()
    with torch.no_grad():
        for i_batch, batch in enumerate(loader):
            a_out=model(batch.to('cuda'))
            embed_vectors=torch.cat((embed_vectors,a_out))   
    embed_vectors=embed_vectors.cpu().detach().numpy()
    embed_dim=embed_vectors.shape[1]

    pickle.dump(embed_vectors, open(f'features_NV_m{model_name}_d{loader.dataset.scene_name}.pkl','wb'))
    print('Saved VGE-CO-vectors')

if __name__=='__main__':
    transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),])

    data_summer_train=SynthiaDataset('data/SYNTHIA-SEQS-04-SUMMER/train', transform=transform)
    data_summer_test =SynthiaDataset('data/SYNTHIA-SEQS-04-SUMMER/test', transform=transform)
    data_summer_dense =SynthiaDataset('data/SYNTHIA-SEQS-04-SUMMER/dense', transform=transform)
    data_dawn_train=  SynthiaDataset('data/SYNTHIA-SEQS-04-DAWN/train', transform=transform)
    data_dawn_test =  SynthiaDataset('data/SYNTHIA-SEQS-04-DAWN/test', transform=transform)
    data_winter_train=SynthiaDataset('data/SYNTHIA-SEQS-04-WINTER/train', transform=transform)
    data_winter_test =SynthiaDataset('data/SYNTHIA-SEQS-04-WINTER/test', transform=transform)

    if 'gather' in sys.argv:
        BATCH_SIZE=12

        # #NV-SYN
        # resnet=create_image_model_resnet_18()
        # netvlad_model=NetvladModel(resnet)
        # netvlad_model_name='model_NV-SYN_lNone_b12_g0.75_sTrue_m0.5_dsummer_lr0.02.pth'
        # netvlad_model.load_state_dict(torch.load('models/'+netvlad_model_name)); print('Model:',netvlad_model_name)
        # netvlad_model.eval()
        # netvlad_model.cuda()  

        # loader=DataLoader(data_summer_train, batch_size=BATCH_SIZE, num_workers=2, pin_memory=True, shuffle=False) 
        # gather_NV_vectors(loader, netvlad_model, "NV-SYN-summer")        

        # loader=DataLoader(data_summer_test, batch_size=BATCH_SIZE, num_workers=2, pin_memory=True, shuffle=False) 
        # gather_NV_vectors(loader, netvlad_model, "NV-SYN-summer")   

        # loader=DataLoader(data_summer_dense, batch_size=BATCH_SIZE, num_workers=2, pin_memory=True, shuffle=False) 
        # gather_NV_vectors(loader, netvlad_model, "NV-SYN-summer")  

        #NV-SYN-FC
        EMBED_DIM_GEOMETRIC=1024
        resnet=create_image_model_resnet_18()
        nv_fc_model=NetvladFCModel(resnet, EMBED_DIM_GEOMETRIC)
        nv_fc_model_name='model_NV-SYN-FC_lNone_b12_g0.75_e1024_sTrue_m0.5_dsummer_lr0.02.pth'
        nv_fc_model.load_state_dict(torch.load('models/'+nv_fc_model_name)); print('Model:',nv_fc_model_name)
        nv_fc_model.eval()
        nv_fc_model.cuda()  

        loader=DataLoader(data_summer_train, batch_size=BATCH_SIZE, num_workers=2, pin_memory=True, shuffle=False) 
        gather_NV_vectors(loader, nv_fc_model, "NV-SYN-FC-summer")        

        loader=DataLoader(data_summer_test, batch_size=BATCH_SIZE, num_workers=2, pin_memory=True, shuffle=False) 
        gather_NV_vectors(loader, nv_fc_model, "NV-SYN-FC-summer")   

        loader=DataLoader(data_summer_dense, batch_size=BATCH_SIZE, num_workers=2, pin_memory=True, shuffle=False) 
        gather_NV_vectors(loader, nv_fc_model, "NV-SYN-FC-summer")  

    if 'NV-SYN' in sys.argv:
        features_name_db   ='features_NV_mNV-SYN-summer_dSUMMER-train.pkl'
        features_name_query='features_NV_mNV-SYN-summer_dSUMMER-test.pkl'
        features_db, features_query=pickle.load(open('evaluation_res/'+features_name_db, 'rb')), pickle.load(open('evaluation_res/'+features_name_query, 'rb')); print('features:',features_name_db, features_name_query)

        pos_results, ori_results=eval_featureVectors(data_summer_train, data_summer_test, features_db, features_query, similarity='l2')
        print(pos_results, ori_results,'\n') 

        features_name_db   ='features_NV_mNV-SYN-summer_dSUMMER-dense.pkl'
        features_name_query='features_NV_mNV-SYN-summer_dSUMMER-test.pkl'
        features_db, features_query=pickle.load(open('evaluation_res/'+features_name_db, 'rb')), pickle.load(open('evaluation_res/'+features_name_query, 'rb')); print('features:',features_name_db, features_name_query)

        pos_results, ori_results=eval_featureVectors(data_summer_dense, data_summer_test, features_db, features_query, similarity='l2')
        print(pos_results, ori_results,'\n')   

    if 'NV-SYN-FC' in sys.argv:
        features_name_db   ='features_NV_mNV-SYN-FC-summer_dSUMMER-train.pkl'
        features_name_query='features_NV_mNV-SYN-FC-summer_dSUMMER-test.pkl'
        features_db, features_query=pickle.load(open('evaluation_res/'+features_name_db, 'rb')), pickle.load(open('evaluation_res/'+features_name_query, 'rb')); print('features:',features_name_db, features_name_query)

        pos_results, ori_results=eval_featureVectors(data_summer_train, data_summer_test, features_db, features_query, similarity='l2')
        print(pos_results, ori_results,'\n') 

        features_name_db   ='features_NV_mNV-SYN-FC-summer_dSUMMER-dense.pkl'
        features_name_query='features_NV_mNV-SYN-FC-summer_dSUMMER-test.pkl'
        features_db, features_query=pickle.load(open('evaluation_res/'+features_name_db, 'rb')), pickle.load(open('evaluation_res/'+features_name_query, 'rb')); print('features:',features_name_db, features_name_query)

        pos_results, ori_results=eval_featureVectors(data_summer_dense, data_summer_test, features_db, features_query, similarity='l2')
        print(pos_results, ori_results,'\n')        


    # #Eval Summer->Summer (sending every 8-th to query, rest database)
    # if 'NV-Summer-Summer' in sys.argv:
    #     train_indices,test_indices=get_split_indices(3604,step=8)
    #     dataset_train=SynthiaDataset('data/SYNTHIA-SEQS-04-SUMMER/', split_indices=train_indices)
    #     dataset_test =SynthiaDataset('data/SYNTHIA-SEQS-04-SUMMER/', split_indices=test_indices)

    #     netvlad_features=pickle.load(open('data/SYNTHIA-SEQS-04-SUMMER/netvlad_features.pkl','rb'))
    #     features_train=netvlad_features[train_indices]
    #     features_test= netvlad_features[test_indices]

    #     print(eval_featureVectors(dataset_train, dataset_test, features_train, features_test))

    # #Eval Dawn->Summer
    # if 'NV-Dawn-Summer' in sys.argv:
    #     netvlad_features_db=   pickle.load(open('data/SYNTHIA-SEQS-04-SUMMER/netvlad_features.pkl','rb'))
    #     netvlad_features_query=pickle.load(open('data/SYNTHIA-SEQS-04-DAWN/netvlad_features.pkl','rb'))
    #     netvlad_features_query=netvlad_features_query[query_indices_dawn]

    #     print(eval_featureVectors(dataset_summer, dataset_dawn, netvlad_features_db, netvlad_features_query))