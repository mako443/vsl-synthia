import numpy as np
import os
import pickle
import sys

from evaluation.evaluation_functions import eval_featureVectors, get_split_indices
from dataloading.data_loading import SynthiaDataset

'''
NetVLAD feature-extraction is done in 'pytorch-NetVlad' sub-directory.
'''

if __name__=='__main__':
    dataset_dawn  =SynthiaDataset('data/SYNTHIA-SEQS-04-DAWN/')
    dataset_summer=SynthiaDataset('data/SYNTHIA-SEQS-04-SUMMER/')

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