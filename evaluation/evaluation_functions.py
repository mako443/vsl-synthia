import numpy as np
import os
import pickle

def eval_scoresDict(dataset_db, dataset_query, scores_dict, retrievals_name=None, top_k=(1,3,5,10), thresholds=[(7.5,5), (10.0,30), (20.0,45)]):
    assert len(dataset_query)==len(scores_dict) and len(dataset_db)==len( next(iter(scores_dict.values())))
    
    image_positions_db, image_orientations_db = dataset_db.image_positions, dataset_db.image_orientations
    image_positions_query, image_orientations_query = dataset_query.image_positions, dataset_query.image_orientations

    retrieval_dict={}

    # pos_results={ k:[] for k in top_k }
    # ori_results={ k:[] for k in top_k }    

    thresh_hits  = {t: {k:[] for k in top_k} for t in thresholds }
    retrieval_counts = {k:[] for k in top_k}      

    for query_index in range(len(dataset_query)):
        pos_dists=np.linalg.norm(image_positions_db[:]-image_positions_query[query_index], axis=1)
        ori_dists=np.abs(image_orientations_db[:]-image_orientations_query[query_index])
        ori_dists=np.minimum(ori_dists, 2*np.pi-ori_dists)    

        db_indices=np.arange(len(dataset_db))
        scores = np.array([ scores_dict[query_index][db_index] for db_index in db_indices ])        

        sorted_indices=np.argsort(-1.0*scores) #High->Low
        assert len(pos_dists)==len(ori_dists)==len(scores)==len(sorted_indices)

        retrieval_dict[query_index]=sorted_indices[:np.max(top_k)]     

        for k in top_k:
            # pos_results[k].append( np.mean(pos_dists[sorted_indices[0:k]]) )
            # ori_results[k].append( np.mean(ori_dists[sorted_indices[0:k]]) )
            topk_pos_dists=pos_dists[sorted_indices[0:k]]
            topk_ori_dists=ori_dists[sorted_indices[0:k]]
            
            retrieval_counts[k].append(len(topk_pos_dists))
            for t in thresholds:
                absolute_pos_thresh=t[0]
                absolute_ori_thresh=np.deg2rad(t[1])                
                thresh_hits[t][k].append( np.sum( (topk_pos_dists<=absolute_pos_thresh) & (topk_ori_dists<=absolute_ori_thresh) ) )

    for k in top_k:
        for t in thresholds:
            thresh_hits[t][k]= np.sum(thresh_hits[t][k]) / np.sum(retrieval_counts[k])
    return thresh_hits

    # assert len(pos_results[k])==len(ori_results[k])==len(dataset_query)

    # for k in top_k:
    #     pos_results[k]=np.float16(np.mean(pos_results[k]))
    #     ori_results[k]=np.float16(np.mean(ori_results[k]))

    # if retrievals_name is not None:
    #     print(f'Saving retrieval results {retrievals_name} ...')
    #     pickle.dump(retrieval_dict, open(f'retrievals_{retrievals_name}','wb'))        

    # return pos_results, ori_results

def eval_scoresDict_threshold(dataset_db, dataset_query, scores_dict, top_k=(1,3,5,10)):
    assert len(dataset_query)==len(scores_dict) and len(dataset_db)==len( next(iter(scores_dict.values())))

    thresh_pos=25
    thresh_ori=np.pi/4
    
    image_positions_db, image_orientations_db = dataset_db.image_positions, dataset_db.image_orientations
    image_positions_query, image_orientations_query = dataset_query.image_positions, dataset_query.image_orientations

    thresh_results={ k:[] for k in top_k }

    for query_index in range(len(dataset_query)):
        pos_dists=np.linalg.norm(image_positions_db[:]-image_positions_query[query_index], axis=1)
        ori_dists=np.abs(image_orientations_db[:]-image_orientations_query[query_index])
        ori_dists=np.minimum(ori_dists, 2*np.pi-ori_dists)    

        db_indices=np.arange(len(dataset_db))
        scores = np.array([ scores_dict[query_index][db_index] for db_index in db_indices ])        

        sorted_indices=np.argsort(-1.0*scores) #High->Low
        assert len(pos_dists)==len(ori_dists)==len(scores)==len(sorted_indices)


        for k in top_k:
            thresh_results[k].append( np.mean( np.bitwise_and(pos_dists[sorted_indices[0:k]]<=thresh_pos, ori_dists[sorted_indices[0:k]]<=thresh_ori )))

    assert len(thresh_results[k])==len(dataset_query)

    for k in top_k:
        thresh_results[k]=np.float16(np.mean(thresh_results[k]))  

    return thresh_results    


def eval_featureVectors(dataset_db, dataset_query, features_db, features_query, top_k=(1,3,5,10), thresholds=[(7.5,5), (10.0,30), (20.0,45)], similarity='l2'):
    assert len(dataset_db)==len(features_db) and len(dataset_query)==len(features_query)
    assert len(dataset_db)/1.5 > len(dataset_query)
    assert similarity in ('l2','cosine')
    #assert dataset_db.image_netvlad_features is not None and dataset_query.image_netvlad_features is not None

    print(f'eval_featureVectors: {dataset_query.scene_name} -> {dataset_db.scene_name}')
    # features_db=dataset_db.image_netvlad_features
    # features_query=dataset_query.image_netvlad_features

    image_positions_db, image_orientations_db = dataset_db.image_positions, dataset_db.image_orientations
    image_positions_query, image_orientations_query = dataset_query.image_positions, dataset_query.image_orientations

    # pos_results={ k:[] for k in top_k }
    # ori_results={ k:[] for k in top_k }
    thresh_hits  = {t: {k:[] for k in top_k} for t in thresholds }
    retrieval_counts = {k:[] for k in top_k}     

    query_indices=np.arange(len(dataset_query))
    for query_index in query_indices:
        pos_dists=np.linalg.norm(image_positions_db[:]-image_positions_query[query_index], axis=1)
        ori_dists=np.abs(image_orientations_db[:]-image_orientations_query[query_index])
        ori_dists=np.minimum(ori_dists, 2*np.pi-ori_dists)    

        if similarity=='l2':
            feature_diffs=features_db-features_query[query_index]
            feature_diffs=np.linalg.norm(feature_diffs,axis=1)   
            sorted_indices=np.argsort(feature_diffs) #Low->High

        #CARE: assumes normed features!
        if similarity=='cosine':
            feature_scores=features_db@features_query.T
            sorted_indices=np.argsort(-1.0*feature_scores) #High->Low

        assert len(pos_dists)==len(dataset_db)==len(sorted_indices)

        for k in top_k:
            # pos_results[k].append( np.mean(pos_dists[sorted_indices[0:k]]) )
            # ori_results[k].append( np.mean(ori_dists[sorted_indices[0:k]]) )
            topk_pos_dists=pos_dists[sorted_indices[0:k]]
            topk_ori_dists=ori_dists[sorted_indices[0:k]]    

            retrieval_counts[k].append(len(topk_pos_dists))
            for t in thresholds:
                absolute_pos_thresh=t[0]
                absolute_ori_thresh=np.deg2rad(t[1])                
                thresh_hits[t][k].append( np.sum( (topk_pos_dists<=absolute_pos_thresh) & (topk_ori_dists<=absolute_ori_thresh) ) )

    for k in top_k:
        for t in thresholds:
            thresh_hits[t][k]= np.sum(thresh_hits[t][k]) / np.sum(retrieval_counts[k])
    return thresh_hits        

    # assert len(pos_results[k])==len(ori_results[k])==len(dataset_query)

    # for k in top_k:
    #     pos_results[k]=np.float16(np.mean(pos_results[k]))
    #     ori_results[k]=np.float16(np.mean(ori_results[k]))

    # return pos_results, ori_results


#Retain above 0.75*max
def top_score_voting(indices_netvlad, semantic_scores):
    assert len(indices_netvlad)==len(semantic_scores)
    max_score=np.max(semantic_scores)
    return indices_netvlad[semantic_scores>=0.75*max_score]    

'''
Indices0: indices top-ranked by NetVLAD
Scores0 : *semantic* scores corresponding to Indices0
Indices1: indices top-ranked by semantics
Scores1 : *semantic* scores corresponding to Indices1
'''
def top_score_voting3(indices0, scores0, indices1, scores1):
    assert len(indices0)==len(scores0)==len(indices1)==len(scores1)
    mean_score=np.mean(scores1) #Mean of best semantic scores
    indices0=indices0[scores0>=0.75*mean_score]
    indices1=indices1[scores1>=0.75*mean_score]
    if len(indices0)>0:
        return indices0
    else:
        return indices1

def top_score_voting4(indices0, scores0, indices1, scores1):
    assert len(indices0)==len(scores0)==len(indices1)==len(scores1)
    max_score=np.max(scores1) #Max of best semantic scores
    indices0=indices0[scores0>=0.75*max_score]
    indices1=indices1[scores1>=0.75*max_score]
    if len(indices0)>0:
        return indices0
    else:
        return indices1        

def eval_featureVectors_scoresDict(dataset_db, dataset_query, features_db, features_query, scores_dict, top_k=(1,3,5,10), thresholds=[(7.5,5), (10.0,30), (20.0,45)], combine='sum'):
    assert len(dataset_db)==len(features_db)==len(next(iter(scores_dict.values()))) and len(dataset_query)==len(features_query)==len(scores_dict)
    assert len(features_db)//2>len(features_query)
    assert combine in ('sum','top-score-voting')
    print(f'Eval features+scores: {dataset_query.scene_name} -> {dataset_db.scene_name}, combine: {combine}')

    image_positions_db, image_orientations_db = dataset_db.image_positions, dataset_db.image_orientations
    image_positions_query, image_orientations_query = dataset_query.image_positions, dataset_query.image_orientations

    # pos_results={ k:[] for k in top_k }
    # ori_results={ k:[] for k in top_k }
    thresh_hits  = {t: {k:[] for k in top_k} for t in thresholds }
    retrieval_counts = {k:[] for k in top_k}    

    query_indices=np.arange(len(dataset_query))
    for query_index in query_indices:
        pos_dists=np.linalg.norm(image_positions_db[:]-image_positions_query[query_index], axis=1)
        ori_dists=np.abs(image_orientations_db[:]-image_orientations_query[query_index])
        ori_dists=np.minimum(ori_dists, 2*np.pi-ori_dists)    

        feature_diffs=features_db-features_query[query_index]
        feature_diffs=np.linalg.norm(feature_diffs,axis=1)   

        db_indices=np.arange(len(dataset_db))
        scores = np.array([ scores_dict[query_index][db_index] for db_index in db_indices ])  

        #Norm both for comparability
        feature_diffs = feature_diffs/np.max(np.abs(feature_diffs))
        scores= scores/np.max(np.abs(scores))
        assert len(scores)==len(feature_diffs)==len(dataset_db)                  

        for k in top_k:
            if combine=='sum':
                combined_scores= scores + -1.0*feature_diffs 
                sorted_indices=np.argsort( -1.0*combined_scores)[0:k] #High->Low  
            if combine=='top-score-voting':
                indices_netvlad=np.argsort(feature_diffs)[0:k]
                semantic_scores=scores[indices_netvlad]
                sorted_indices=top_score_voting(indices_netvlad, semantic_scores)

            assert len(sorted_indices)<=k
            topk_pos_dists=pos_dists[sorted_indices]
            topk_ori_dists=ori_dists[sorted_indices]    

            retrieval_counts[k].append(len(topk_pos_dists))
            for t in thresholds:
                absolute_pos_thresh=t[0]
                absolute_ori_thresh=np.deg2rad(t[1])                
                thresh_hits[t][k].append( np.sum( (topk_pos_dists<=absolute_pos_thresh) & (topk_ori_dists<=absolute_ori_thresh) ) )

    for k in top_k:
        for t in thresholds:
            thresh_hits[t][k]= np.sum(thresh_hits[t][k]) / np.sum(retrieval_counts[k])
    return thresh_hits        
                        

    #         # pos_results[k].append( np.mean(pos_dists[sorted_indices[0:k]]) )
    #         # ori_results[k].append( np.mean(ori_dists[sorted_indices[0:k]]) )

    # assert len(pos_results[k])==len(ori_results[k])==len(dataset_query)

    # for k in top_k:
    #     pos_results[k]=np.float16(np.mean(pos_results[k]))
    #     ori_results[k]=np.float16(np.mean(ori_results[k]))

    # return pos_results, ori_results    

#TODO / Optional
def reduce_indices_gridVoting():
    pass

def get_split_indices(size, step=8):
    query_indices=np.zeros(size, dtype=np.bool)
    query_indices[::step]=True
    db_indices= ~query_indices
    return db_indices,query_indices

def print_topK(thresh_results):
    print('<-----')
    top_t=sorted(list(thresh_results.keys()))
    top_k=sorted(list(thresh_results[top_t[0]].keys()))
    for t in top_t: 
        print(f'{t[0]}/{ t[1] :0.0f}', end="")
        for k in top_k:
            print('\t', end="")
    # print('Scene')
    print()
    for t in top_t:
        for k in top_k:
            print(f'{k}\t', end="")
    print('\n------')
    for t in top_t:
        for k in top_k:
            print(f'{thresh_results[t][k]:0.3f}\t', end="")
    # for k in top_k:
    #     print(f'{scene_results[k]:0.3f}\t', end="")   
    print('\n----->\n')      