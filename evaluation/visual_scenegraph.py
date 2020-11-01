import numpy as np
import os
import pickle
import sys

from evaluation.evaluation_functions import eval_scoresDict, eval_scoresDict_threshold, eval_featureVectors_scoresDict, print_topK
# from semantic.scenegraphs import score_scenegraph_to_viewobjects
from semantic.scenegraphs2 import score_scenegraph_to_viewobjects, score_scenegraph_to_scenegraph

from dataloading.data_loading import SynthiaDataset

def gather_scoresDict_sceneGraph2ViewObjects(dataset_db, dataset_query, score_params):
    assert len(dataset_db)//1.5>len(dataset_query)
    print(f'Gathering SG-VO {dataset_query.scene_name} to {dataset_db.scene_name}, params: {score_params}')

    scores_dict={} # Dict { query-idx: { db-index: score} }

    for query_index in range(len(dataset_query)):
        scores_dict[query_index] = {}
        sg_query=dataset_query.image_scenegraphs[query_index]

        for db_index in range(len(dataset_db)):
            score=score_scenegraph_to_viewobjects(sg_query, dataset_db.image_viewobjects[db_index], score_params)
            scores_dict[query_index][db_index]=score
        
        assert len(scores_dict[query_index])==len(dataset_db)

    assert len(scores_dict)==len(dataset_query)

    s0,s1,s2,s3,s4=score_params
    save_name=f'scores_SG-VO_{dataset_query.scene_name}-{dataset_db.scene_name}_{s0}_{s1}_{s2}_{s3}_{s4}.pkl'
    print('Saving SG-VO scores...',save_name)
    pickle.dump(scores_dict, open(save_name,'wb'))

def gather_scoresDict_sceneGraph2sceneGraph(dataset_db, dataset_query, score_params):
    assert len(dataset_db)//1.5>len(dataset_query)
    print(f'Gathering SG-SG {dataset_query.scene_name} to {dataset_db.scene_name}, params: {score_params}')

    scores_dict={} # Dict { query-idx: { db-index: score} }

    for query_index in range(len(dataset_query)):
        scores_dict[query_index] = {}
        sg_query=dataset_query.image_scenegraphs[query_index]

        for db_index in range(len(dataset_db)):
            score=score_scenegraph_to_scenegraph(sg_query, dataset_db.image_scenegraphs[db_index], score_params)
            scores_dict[query_index][db_index]=score
        
        assert len(scores_dict[query_index])==len(dataset_db)

    assert len(scores_dict)==len(dataset_query)

    s0,s1,s2,s3,s4=score_params
    save_name=f'scores_SG-SG_{dataset_query.scene_name}-{dataset_db.scene_name}_{s0}_{s1}_{s2}_{s3}_{s4}.pkl'
    print('Saving SG-SG scores...',save_name)
    pickle.dump(scores_dict, open(save_name,'wb'))    

if __name__=='__main__':
    data_summer_train = SynthiaDataset('data/SYNTHIA-SEQS-04-SUMMER/train')
    data_summer_test  = SynthiaDataset('data/SYNTHIA-SEQS-04-SUMMER/test')
    data_summer_dense  = SynthiaDataset('data/SYNTHIA-SEQS-04-SUMMER/dense')
    # data_dawn_train   = SynthiaDataset('data/SYNTHIA-SEQS-04-DAWN/train')
    data_dawn_test    = SynthiaDataset('data/SYNTHIA-SEQS-04-DAWN/test')
    # data_winter_train = SynthiaDataset('data/SYNTHIA-SEQS-04-WINTER/train')
    # data_winter_test  = SynthiaDataset('data/SYNTHIA-SEQS-04-WINTER/test')


    if 'gather-SG-VO-1' in sys.argv:
        gather_scoresDict_sceneGraph2ViewObjects(data_summer_dense, data_summer_test, (1,0,1,1,-0.15))
    if 'gather-SG-VO-2' in sys.argv:
        gather_scoresDict_sceneGraph2ViewObjects(data_summer_dense, data_summer_test, (1,1,0,1,-0.15))
    if 'gather-SG-VO-3' in sys.argv:
        gather_scoresDict_sceneGraph2ViewObjects(data_summer_dense, data_summer_test, (1,1,1,0,-0.15)) 

    if 'gather-SG-VO' in sys.argv:
        #Summer -> Summer
        gather_scoresDict_sceneGraph2ViewObjects(data_summer_dense, data_summer_test, (1,1,1,1,-0.15))
        #Dawn -> Summer
        gather_scoresDict_sceneGraph2ViewObjects(data_summer_dense, data_dawn_test, (1,1,1,1,-0.15))        
        gather_scoresDict_sceneGraph2ViewObjects(data_summer_dense, data_dawn_test, (1,0,1,1,-0.25))                    

    if 'SG-VO' in sys.argv:
        #Ablations
        scores_filename='scores_SG-VO_SUMMER-test-SUMMER-dense_1_1_0_1_-0.15.pkl'
        scores_dict=pickle.load( open('evaluation_res/'+scores_filename, 'rb')); print('Using scores:',scores_filename)        
        thresh_results=eval_scoresDict(data_summer_dense, data_summer_test, scores_dict)
        print_topK(thresh_results)

        scores_filename='scores_SG-VO_SUMMER-test-SUMMER-dense_1_1_1_0_-0.15.pkl'
        scores_dict=pickle.load( open('evaluation_res/'+scores_filename, 'rb')); print('Using scores:',scores_filename)        
        thresh_results=eval_scoresDict(data_summer_dense, data_summer_test, scores_dict)
        print_topK(thresh_results)        

        quit()


        scores_filename='scores_SG2VO_SUMMER-test-SUMMER-dense_1_1_1_1_-0.15.pkl'
        scores_dict=pickle.load( open('evaluation_res/'+scores_filename, 'rb')); print('Using scores:',scores_filename)        
        thresh_results=eval_scoresDict(data_summer_dense, data_summer_test, scores_dict)
        print_topK(thresh_results)

        scores_filename='scores_SG2VO_SUMMER-test-SUMMER-dense_1_0_1_1_-0.25.pkl'
        scores_dict=pickle.load( open('evaluation_res/'+scores_filename, 'rb')); print('Using scores:',scores_filename)        
        thresh_results=eval_scoresDict(data_summer_dense, data_summer_test, scores_dict)
        print_topK(thresh_results)     

        scores_filename='scores_SG2VO_DAWN-test-SUMMER-dense_1_1_1_1_-0.15.pkl'
        scores_dict=pickle.load( open('evaluation_res/'+scores_filename, 'rb')); print('Using scores:',scores_filename)        
        thresh_results=eval_scoresDict(data_summer_dense, data_dawn_test, scores_dict)
        print_topK(thresh_results)    

        scores_filename='scores_SG2VO_DAWN-test-SUMMER-dense_1_0_1_1_-0.25.pkl'
        scores_dict=pickle.load( open('evaluation_res/'+scores_filename, 'rb')); print('Using scores:',scores_filename)        
        thresh_results=eval_scoresDict(data_summer_dense, data_dawn_test, scores_dict)
        print_topK(thresh_results) 

    if 'gather-SG-SG-1' in sys.argv:
        gather_scoresDict_sceneGraph2sceneGraph(data_summer_dense, data_summer_test, (1,1,1,1,-0.15))   
        gather_scoresDict_sceneGraph2sceneGraph(data_summer_dense, data_summer_test, (1,0,1,1,-0.25))                
        gather_scoresDict_sceneGraph2sceneGraph(data_summer_dense, data_dawn_test, (1,1,1,1,-0.15))   
        gather_scoresDict_sceneGraph2sceneGraph(data_summer_dense, data_dawn_test, (1,0,1,1,-0.25))    

    if 'SG-SG' in sys.argv:
        scores_filename='scores_SG-SG_SUMMER-test-SUMMER-dense_1_1_1_1_-0.15.pkl'
        scores_dict=pickle.load( open('evaluation_res/'+scores_filename, 'rb')); print('Using scores:',scores_filename)        
        thresh_results=eval_scoresDict(data_summer_dense, data_summer_test, scores_dict)
        print_topK(thresh_results)

        scores_filename='scores_SG-SG_SUMMER-test-SUMMER-dense_1_0_1_1_-0.25.pkl'
        scores_dict=pickle.load( open('evaluation_res/'+scores_filename, 'rb')); print('Using scores:',scores_filename)        
        thresh_results=eval_scoresDict(data_summer_dense, data_summer_test, scores_dict)
        print_topK(thresh_results) 

        scores_filename='scores_SG-SG_DAWN-test-SUMMER-dense_1_1_1_1_-0.15.pkl'
        scores_dict=pickle.load( open('evaluation_res/'+scores_filename, 'rb')); print('Using scores:',scores_filename)        
        thresh_results=eval_scoresDict(data_summer_dense, data_dawn_test, scores_dict)
        print_topK(thresh_results) 

        scores_filename='scores_SG-SG_DAWN-test-SUMMER-dense_1_0_1_1_-0.25.pkl'
        scores_dict=pickle.load( open('evaluation_res/'+scores_filename, 'rb')); print('Using scores:',scores_filename)        
        thresh_results=eval_scoresDict(data_summer_dense, data_dawn_test, scores_dict)
        print_topK(thresh_results)                        


    if 'SG2VO-thresh' in sys.argv:
        scores_filename='scores_SG2VO_SUMMER-test-SUMMER-dense_1_1_1_1_-0.15.pkl'
        scores_dict=pickle.load( open('evaluation_res/'+scores_filename, 'rb')); print('Using scores:',scores_filename)        
        thresh_results=eval_scoresDict_threshold(data_summer_dense, data_summer_test, scores_dict)
        print(thresh_results,'\n')     

        scores_filename='scores_SG2VO_SUMMER-test-SUMMER-dense_1_0_1_1_-0.25.pkl'
        scores_dict=pickle.load( open('evaluation_res/'+scores_filename, 'rb')); print('Using scores:',scores_filename)        
        thresh_results=eval_scoresDict_threshold(data_summer_dense, data_summer_test, scores_dict)
        print(thresh_results,'\n')            

    if 'netvlad-SG2VO-summer' in sys.argv:
        features_name_db   ='features_NV_mNV-SYN-summer_dSUMMER-dense.pkl'
        features_name_query='features_NV_mNV-SYN-summer_dSUMMER-test.pkl'
        features_db, features_query=pickle.load(open('evaluation_res/'+features_name_db, 'rb')), pickle.load(open('evaluation_res/'+features_name_query, 'rb')); print('features:',features_name_db, features_name_query)
        scores_filename='scores_SG2VO_SUMMER-test-SUMMER-dense_1_1_1_1_-0.15.pkl'
        scores_dict=pickle.load( open('evaluation_res/'+scores_filename, 'rb')); print('Using scores:',scores_filename)
        pos_results, ori_results=eval_featureVectors_scoresDict(data_summer_dense, data_summer_test, features_db, features_query, scores_dict, combine='sum')
        print(pos_results, ori_results,'\n') 
        pos_results, ori_results=eval_featureVectors_scoresDict(data_summer_dense, data_summer_test, features_db, features_query, scores_dict, combine='top-score-voting')
        print(pos_results, ori_results,'\n')

        features_name_db   ='features_NV_mNV-SYN-summer_dSUMMER-dense.pkl'
        features_name_query='features_NV_mNV-SYN-summer_dSUMMER-test.pkl'
        features_db, features_query=pickle.load(open('evaluation_res/'+features_name_db, 'rb')), pickle.load(open('evaluation_res/'+features_name_query, 'rb')); print('features:',features_name_db, features_name_query)
        scores_filename='scores_SG2VO_SUMMER-test-SUMMER-dense_1_0_1_1_-0.25.pkl'
        scores_dict=pickle.load( open('evaluation_res/'+scores_filename, 'rb')); print('Using scores:',scores_filename)
        pos_results, ori_results=eval_featureVectors_scoresDict(data_summer_dense, data_summer_test, features_db, features_query, scores_dict, combine='sum')
        print(pos_results, ori_results,'\n')   

    if 'netvlad-SG2VO-dawn-summer' in sys.argv:
        features_name_db   ='features_NV_mNV-SYN-summer_dSUMMER-dense.pkl'
        features_name_query='features_NV_mNV-SYN-summer_dDAWN-test.pkl'
        features_db, features_query=pickle.load(open('evaluation_res/'+features_name_db, 'rb')), pickle.load(open('evaluation_res/'+features_name_query, 'rb')); print('features:',features_name_db, features_name_query)
        scores_filename='scores_SG2VO_DAWN-test-SUMMER-dense_1_1_1_1_-0.15.pkl'  
        scores_dict=pickle.load( open('evaluation_res/'+scores_filename, 'rb')); print('Using scores:',scores_filename)    
        pos_results, ori_results=eval_featureVectors_scoresDict(data_summer_dense, data_dawn_test, features_db, features_query, scores_dict, combine='sum')
        print(pos_results, ori_results,'\n')    

        features_name_db   ='features_NV_mNV-SYN-summer_dSUMMER-dense.pkl'
        features_name_query='features_NV_mNV-SYN-summer_dDAWN-test.pkl'
        features_db, features_query=pickle.load(open('evaluation_res/'+features_name_db, 'rb')), pickle.load(open('evaluation_res/'+features_name_query, 'rb')); print('features:',features_name_db, features_name_query)
        scores_filename='scores_SG2VO_DAWN-test-SUMMER-dense_1_0_1_1_-0.25.pkl'  
        scores_dict=pickle.load( open('evaluation_res/'+scores_filename, 'rb')); print('Using scores:',scores_filename)    
        pos_results, ori_results=eval_featureVectors_scoresDict(data_summer_dense, data_dawn_test, features_db, features_query, scores_dict, combine='sum')
        print(pos_results, ori_results,'\n')                        

    if 'NV-Pitts_SG2VO' in sys.argv:
        #Summer->Summer
        features_name_db   ='features_NV_mNV-Pitts_dSUMMER-dense.pkl'
        features_name_query='features_NV_mNV-Pitts_dSUMMER-test.pkl'
        features_db, features_query=pickle.load(open('evaluation_res/'+features_name_db, 'rb')), pickle.load(open('evaluation_res/'+features_name_query, 'rb')); print('features:',features_name_db, features_name_query)
        scores_filename='scores_SG2VO_SUMMER-test-SUMMER-dense_1_1_1_1_-0.15.pkl'
        scores_dict=pickle.load( open('evaluation_res/'+scores_filename, 'rb')); print('Using scores:',scores_filename)
        thresh_results=eval_featureVectors_scoresDict(data_summer_dense, data_summer_test, features_db, features_query, scores_dict, combine='sum')
        print_topK(thresh_results) 

        features_name_db   ='features_NV_mNV-Pitts_dSUMMER-dense.pkl'
        features_name_query='features_NV_mNV-Pitts_dDAWN-test.pkl'
        features_db, features_query=pickle.load(open('evaluation_res/'+features_name_db, 'rb')), pickle.load(open('evaluation_res/'+features_name_query, 'rb')); print('features:',features_name_db, features_name_query)
        scores_filename='scores_SG2VO_DAWN-test-SUMMER-dense_1_1_1_1_-0.15.pkl'
        scores_dict=pickle.load( open('evaluation_res/'+scores_filename, 'rb')); print('Using scores:',scores_filename)
        thresh_results=eval_featureVectors_scoresDict(data_summer_dense, data_dawn_test, features_db, features_query, scores_dict, combine='sum')
        print_topK(thresh_results)

        features_name_db   ='features_NV_mNV-Pitts_dSUMMER-dense.pkl'
        features_name_query='features_NV_mNV-Pitts_dDAWN-test.pkl'
        features_db, features_query=pickle.load(open('evaluation_res/'+features_name_db, 'rb')), pickle.load(open('evaluation_res/'+features_name_query, 'rb')); print('features:',features_name_db, features_name_query)
        scores_filename='scores_SG2VO_DAWN-test-SUMMER-dense_1_0_1_1_-0.25.pkl'
        scores_dict=pickle.load( open('evaluation_res/'+scores_filename, 'rb')); print('Using scores:',scores_filename)
        thresh_results=eval_featureVectors_scoresDict(data_summer_dense, data_dawn_test, features_db, features_query, scores_dict, combine='sum')
        print_topK(thresh_results)      

    if 'NV-Pitts_SG-SG' in sys.argv:
        #Summer->Summer
        features_name_db   ='features_NV_mNV-Pitts_dSUMMER-dense.pkl'
        features_name_query='features_NV_mNV-Pitts_dSUMMER-test.pkl'
        features_db, features_query=pickle.load(open('evaluation_res/'+features_name_db, 'rb')), pickle.load(open('evaluation_res/'+features_name_query, 'rb')); print('features:',features_name_db, features_name_query)
        scores_filename='scores_SG-SG_SUMMER-test-SUMMER-dense_1_1_1_1_-0.15.pkl'
        scores_dict=pickle.load( open('evaluation_res/'+scores_filename, 'rb')); print('Using scores:',scores_filename)
        thresh_results=eval_featureVectors_scoresDict(data_summer_dense, data_summer_test, features_db, features_query, scores_dict, combine='sum')
        print_topK(thresh_results) 

        #Dawn -> Summer
        features_name_db   ='features_NV_mNV-Pitts_dSUMMER-dense.pkl'
        features_name_query='features_NV_mNV-Pitts_dDAWN-test.pkl'
        features_db, features_query=pickle.load(open('evaluation_res/'+features_name_db, 'rb')), pickle.load(open('evaluation_res/'+features_name_query, 'rb')); print('features:',features_name_db, features_name_query)
        scores_filename='scores_SG-SG_DAWN-test-SUMMER-dense_1_1_1_1_-0.15.pkl'
        scores_dict=pickle.load( open('evaluation_res/'+scores_filename, 'rb')); print('Using scores:',scores_filename)
        thresh_results=eval_featureVectors_scoresDict(data_summer_dense, data_dawn_test, features_db, features_query, scores_dict, combine='sum')
        print_topK(thresh_results)

        features_name_db   ='features_NV_mNV-Pitts_dSUMMER-dense.pkl'
        features_name_query='features_NV_mNV-Pitts_dDAWN-test.pkl'
        features_db, features_query=pickle.load(open('evaluation_res/'+features_name_db, 'rb')), pickle.load(open('evaluation_res/'+features_name_query, 'rb')); print('features:',features_name_db, features_name_query)
        scores_filename='scores_SG-SG_DAWN-test-SUMMER-dense_1_0_1_1_-0.25.pkl'
        scores_dict=pickle.load( open('evaluation_res/'+scores_filename, 'rb')); print('Using scores:',scores_filename)
        thresh_results=eval_featureVectors_scoresDict(data_summer_dense, data_dawn_test, features_db, features_query, scores_dict, combine='sum')
        print_topK(thresh_results)                         
      
