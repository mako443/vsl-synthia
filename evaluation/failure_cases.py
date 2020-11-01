import pickle
import numpy as np
import cv2
import os

from dataloading.data_loading import SynthiaDataset

def get_biggest_top3_position_errors(retrieval_dict, data_train, data_test, num_results=10):
    assert len(retrieval_dict)==len(data_test)
    query_indices=np.arange(len(data_train))
    #db_indices=np.arange(len(data_test))

    pos_errors= np.zeros( len(query_indices) )
    for query_idx in query_indices:
        top3_indices=retrieval_dict[query_idx][0:3]
        pos_errors[query_idx] = np.mean([ np.linalg.norm( dataset_test.image_positions[query_idx] - dataset_train.image_positions[db_idx] ) for db_idx in top3_indices ])

    sorted_indices=np.argsort(-1*pos_errors) # High->Low

    results= {query_indices[i]: retrieval_dict[query_indices[i]][0:3]} #TODO: check this is correct!

    print(f'Top {num_results} position errors:', pos_errors[sorted_indices[0:num_results]])
    return results

def view_cases(cases, dataset_train, dataset_test, num_cases=10):
    if len(cases)<=num_cases: query_indices=list(cases.keys())
    else: query_indices= np.random.choice(list(cases.keys()), size=num_cases)
    
    for i,query_idx in enumerate(query_indices):
        db_indices=cases[query_idx]
        print('Showing',query_idx, db_indices)
        img_query=  cv2.cvtColor(np.asarray(dataset_test[query_idx]), cv2.COLOR_RGB2BGR)
        if type(db_indices) in (list, np.ndarray):
            img_db   =  cv2.cvtColor( np.hstack(( [np.asarray(dataset_train[db_idx]) for db_idx in db_indices] )) , cv2.COLOR_RGB2BGR)
            img_query, img_db= cv2.resize(img_query, None, fx=0.5, fy=0.5), cv2.resize(img_db, None, fx=0.5, fy=0.5)
        else:
            img_db   =  cv2.cvtColor(np.asarray(dataset_train[db_indices]), cv2.COLOR_RGB2BGR)
        cv2.imshow("query", img_query)
        cv2.imshow("db",    img_db)
        cv2.imwrite(f'cases_{i}.png', np.hstack(( img_query, img_db )) )
        cv2.waitKey()    

if __name__=='__main__':
    data_train=SynthiaDataset('data/SYNTHIA-SEQS-04-SUMMER/selection/train')
    data_test =SynthiaDataset('data/SYNTHIA-SEQS-04-SUMMER/selection/test')

    retrieval_dict=pickle.load(open('retrievals/'+'retrievals_NV_SG-Match.pkl','rb'))

    cases=get_biggest_top3_position_errors(retrieval_dict, data_train, data_test)
