import numpy as np
import os
import pickle
import sys

from evaluation.evaluation_functions import eval_scoresDict, eval_featureVectors_with_scoresDict, get_split_indices
from semantic.scenegraphs import score_scenegraph_to_viewobjects

from dataloading.data_loading import SynthiaDataset

#TODO: COLORS!

def gather_scoresDict_sceneGraph2ViewObjects(dataset_db, dataset_query):
    assert len(dataset_db)//2>len(dataset_query)
    print(f'Gathering {dataset_query.scene_name} to {dataset_db.scene_name} ...')

    scores_dict={} # Dict { query-idx: { db-index: score} }

    unused_factor=0.5
    score_combine='multiply'
    score_relationship='sum'
    min_penalty=0.1

    for query_index in range(len(dataset_query)):
        scores_dict[query_index] = {}
        scene_graph_query=dataset_query.image_scenegraphs[query_index]

        for db_index in range(len(dataset_db)):
            score,_=score_scenegraph_to_viewobjects(scene_graph_query, dataset_db.image_viewobjects[db_index],min_penalty=min_penalty, unused_factor=unused_factor, score_combine=score_combine, score_relationship=score_relationship)
            scores_dict[query_index][db_index]=score
        
        assert len(scores_dict[query_index])==len(dataset_db)

    assert len(scores_dict)==len(dataset_query)

    save_name=f'scores_SG2VO_{dataset_query.scene_name}-{dataset_db.scene_name}_u{unused_factor}_mp{min_penalty}_sc{score_combine}_sr{score_relationship}.pkl'
    print('Saving SG-scores...',save_name)
    pickle.dump(scores_dict, open(save_name,'wb'))


if __name__=='__main__':
    dataset_summer=SynthiaDataset('data/SYNTHIA-SEQS-04-SUMMER/') #Len: 3604
    
    _,query_indices_dawn=get_split_indices(3400,step=8)
    dataset_dawn  =SynthiaDataset('data/SYNTHIA-SEQS-04-DAWN/', split_indices=query_indices_dawn) #Len: 3400 

    if 'Gather-Summer-Summer' in sys.argv:
        train_indices,test_indices=get_split_indices(3604,step=8)
        dataset_train=SynthiaDataset('data/SYNTHIA-SEQS-04-SUMMER/', split_indices=train_indices)
        dataset_test =SynthiaDataset('data/SYNTHIA-SEQS-04-SUMMER/', split_indices=test_indices)

        gather_scoresDict_sceneGraph2ViewObjects(dataset_train, dataset_test)

    if 'Gather-Dawn-Summer' in sys.argv:
        gather_scoresDict_sceneGraph2ViewObjects(dataset_summer, dataset_dawn)

    if 'SG2VO-Summer-Summer' in sys.argv:
        train_indices,test_indices=get_split_indices(3604,step=8)
        dataset_train=SynthiaDataset('data/SYNTHIA-SEQS-04-SUMMER/', split_indices=train_indices)
        dataset_test =SynthiaDataset('data/SYNTHIA-SEQS-04-SUMMER/', split_indices=test_indices)

        scores_filename='scores_SG2VO_Summer-Summer.pkl'
        scores_dict=pickle.load( open('evaluation_res/'+scores_filename, 'rb')); print('Using scores:',scores_filename)
        print(eval_scoresDict(dataset_train, dataset_test, scores_dict), '\n')

        scores_filename='scores_SG2VO_Summer-Summer_u0.5_scsum.pkl'
        scores_dict=pickle.load( open('evaluation_res/'+scores_filename, 'rb')); print('Using scores:',scores_filename)
        print(eval_scoresDict(dataset_train, dataset_test, scores_dict), '\n')

        scores_filename='scores_SG2VO_Summer-Summer_uNone_scmultiply.pkl'
        scores_dict=pickle.load( open('evaluation_res/'+scores_filename, 'rb')); print('Using scores:',scores_filename)
        print(eval_scoresDict(dataset_train, dataset_test, scores_dict), '\n')     

        scores_filename='scores_SG2VO_Summer-Summer_u0.5_mp0.01_scmultiply_srmultiply.pkl'
        scores_dict=pickle.load( open('evaluation_res/'+scores_filename, 'rb')); print('Using scores:',scores_filename)
        print(eval_scoresDict(dataset_train, dataset_test, scores_dict), '\n')   

        scores_filename='scores_SG2VO_Summer-Summer_u0.5_mp0.1_scmultiply_srsum.pkl'
        scores_dict=pickle.load( open('evaluation_res/'+scores_filename, 'rb')); print('Using scores:',scores_filename)
        print(eval_scoresDict(dataset_train, dataset_test, scores_dict), '\n')                          

    if 'SG2VO-Dawn-Summer' in sys.argv:
        scores_filename='scores_SG2VO_DAWN-SUMMER.pkl'
        scores_dict=pickle.load( open('evaluation_res/'+scores_filename, 'rb')); print('Using scores:',scores_filename)
        print(eval_scoresDict(dataset_summer, dataset_dawn, scores_dict), '\n')   
