import numpy as np
import os
import pickle
import sys

from evaluation.evaluation_functions import eval_featureVectors, get_split_indices
from dataloading.data_loading import SynthiaDataset

#CARE: NetVLAD feature-extraction is done in 'pytorch-NetVlad' sub-directory.

if __name__=='__main__':
    dataset_summer=SynthiaDataset('data/SYNTHIA-SEQS-04-SUMMER/') #Len: 3604
    
    _,query_indices_dawn=get_split_indices(3400,step=8)
    dataset_dawn  =SynthiaDataset('data/SYNTHIA-SEQS-04-DAWN/', split_indices=query_indices_dawn) #Len: 3400

    #Eval Summer->Summer (sending every 8-th to query, rest database)
    if 'NV-Summer-Summer' in sys.argv:
        train_indices,test_indices=get_split_indices(3604,step=8)
        dataset_train=SynthiaDataset('data/SYNTHIA-SEQS-04-SUMMER/', split_indices=train_indices)
        dataset_test =SynthiaDataset('data/SYNTHIA-SEQS-04-SUMMER/', split_indices=test_indices)

        netvlad_features=pickle.load(open('data/SYNTHIA-SEQS-04-SUMMER/netvlad_features.pkl','rb'))
        features_train=netvlad_features[train_indices]
        features_test= netvlad_features[test_indices]

        print(eval_featureVectors(dataset_train, dataset_test, features_train, features_test))

    #Eval Dawn->Summer
    if 'NV-Dawn-Summer' in sys.argv:
        netvlad_features_db=   pickle.load(open('data/SYNTHIA-SEQS-04-SUMMER/netvlad_features.pkl','rb'))
        netvlad_features_query=pickle.load(open('data/SYNTHIA-SEQS-04-DAWN/netvlad_features.pkl','rb'))
        netvlad_features_query=netvlad_features_query[query_indices_dawn]

        print(eval_featureVectors(dataset_summer, dataset_dawn, netvlad_features_db, netvlad_features_query))