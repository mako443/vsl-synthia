import numpy as np
import os

def eval_scoresDict(dataset_db, dataset_query, scores_dict, top_k=(1,3,5,10)):
    assert len(dataset_query)==len(scores_dict) and len(dataset_db)==len( next(iter(scores_dict.values())))
    
    image_positions_db, image_orientations_db = dataset_db.image_positions, dataset_db.image_orientations
    image_positions_query, image_orientations_query = dataset_query.image_positions, dataset_query.image_orientations

    pos_results={ k:[] for k in top_k }
    ori_results={ k:[] for k in top_k }    

    for query_index in range(len(dataset_query)):
        pos_dists=np.linalg.norm(image_positions_db[:]-image_positions_query[query_index], axis=1)
        ori_dists=np.abs(image_orientations_db[:]-image_orientations_query[query_index])
        ori_dists=np.minimum(ori_dists, 2*np.pi-ori_dists)    

        db_indices=np.arange(len(dataset_db))
        scores = np.array([ scores_dict[query_index][db_index] for db_index in db_indices ])        

        sorted_indices=np.argsort(-1.0*scores) #High->Low
        assert len(pos_dists)==len(ori_dists)==len(scores)==len(sorted_indices)

        for k in top_k:
            pos_results[k].append( np.mean(pos_dists[sorted_indices[0:k]]) )
            ori_results[k].append( np.mean(ori_dists[sorted_indices[0:k]]) )

    assert len(pos_results[k])==len(ori_results[k])==len(dataset_query)

    for k in top_k:
        pos_results[k]=np.float16(np.mean(pos_results[k]))
        ori_results[k]=np.float16(np.mean(ori_results[k]))

    return pos_results, ori_results


def eval_featureVectors(dataset_db, dataset_query, features_db, features_query, top_k=(1,3,5,10)):
    assert len(dataset_db)==len(features_db) and len(dataset_query)==len(features_query)
    assert len(dataset_db)/1.5 > len(dataset_query)
    #assert dataset_db.image_netvlad_features is not None and dataset_query.image_netvlad_features is not None

    print(f'eval_featureVectors: {dataset_query.scene_name} -> {dataset_db.scene_name}')
    # features_db=dataset_db.image_netvlad_features
    # features_query=dataset_query.image_netvlad_features

    image_positions_db, image_orientations_db = dataset_db.image_positions, dataset_db.image_orientations
    image_positions_query, image_orientations_query = dataset_query.image_positions, dataset_query.image_orientations

    pos_results={ k:[] for k in top_k }
    ori_results={ k:[] for k in top_k }

    query_indices=np.arange(len(dataset_query))
    for query_index in query_indices:
        pos_dists=np.linalg.norm(image_positions_db[:]-image_positions_query[query_index], axis=1)
        ori_dists=np.abs(image_orientations_db[:]-image_orientations_query[query_index])
        ori_dists=np.minimum(ori_dists, 2*np.pi-ori_dists)    

        feature_diffs=features_db-features_query[query_index]
        feature_diffs=np.linalg.norm(feature_diffs,axis=1)   

        sorted_indices=np.argsort(feature_diffs) #Low->High
        assert len(pos_dists)==len(dataset_db)==len(sorted_indices)

        for k in top_k:
            pos_results[k].append( np.mean(pos_dists[sorted_indices[0:k]]) )
            ori_results[k].append( np.mean(ori_dists[sorted_indices[0:k]]) )

    assert len(pos_results[k])==len(ori_results[k])==len(dataset_query)

    for k in top_k:
        pos_results[k]=np.float16(np.mean(pos_results[k]))
        ori_results[k]=np.float16(np.mean(ori_results[k]))

    return pos_results, ori_results

def eval_featureVectors_scoresDict(dataset_db, dataset_query, features_db, features_query, scores_dict, top_k=(1,3,5,10), combine='sum'):
    print(len(dataset_db))
    print(len(features_db))
    print(len(next(iter(scores_dict.values()))))
    print(len(dataset_query))
    print(len(features_query))
    print(len(scores_dict))

    assert len(dataset_db)==len(features_db)==len(next(iter(scores_dict.values()))) and len(dataset_query)==len(features_query)==len(scores_dict)
    assert len(features_db)//2>len(features_query)
    assert combine in ('sum',)

    image_positions_db, image_orientations_db = dataset_db.image_positions, dataset_db.image_orientations
    image_positions_query, image_orientations_query = dataset_query.image_positions, dataset_query.image_orientations

    pos_results={ k:[] for k in top_k }
    ori_results={ k:[] for k in top_k }

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

        if combine=='sum':
            combined_scores= scores + -1.0*feature_diffs 
            sorted_indices=np.argsort( -1.0*combined_scores) #High->Low            

        assert len(pos_dists)==len(dataset_db)==len(sorted_indices)

        for k in top_k:
            pos_results[k].append( np.mean(pos_dists[sorted_indices[0:k]]) )
            ori_results[k].append( np.mean(ori_dists[sorted_indices[0:k]]) )

    assert len(pos_results[k])==len(ori_results[k])==len(dataset_query)

    for k in top_k:
        pos_results[k]=np.float16(np.mean(pos_results[k]))
        ori_results[k]=np.float16(np.mean(ori_results[k]))

    return pos_results, ori_results    

#TODO / Optional
def reduce_indices_gridVoting():
    pass

def get_split_indices(size, step=8):
    query_indices=np.zeros(size, dtype=np.bool)
    query_indices[::step]=True
    db_indices= ~query_indices
    return db_indices,query_indices