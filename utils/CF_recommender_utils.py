import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def calculate_hit_rate(left_out_dict, user_ids_lst, top_20_recommended_ids):
    '''
    Claculate hit rate for top 20 using left-one-out set
    '''
    hit_rate = 0
    total_users = len(user_ids_lst)
    
    for user, ids_lst in zip(user_ids_lst, top_20_recommended_ids):
        if left_out_dict[user] in ids_lst:
            hit_rate += 1
    return hit_rate/total_users

def aggregare_vectors(people_lst, artists_vectors_df):
    '''
    return mean of the vectors from people_lst
    '''
    vector_matrix = np.array(artists_vectors_df[artists_vectors_df['person_id'].isin(set(people_lst))]['vector'])
    if len(vector_matrix) > 1:
        return np.mean(vector_matrix, axis=0)
    else: 
        if len(vector_matrix) == 1:
            return vector_matrix[0]
        else:
            return vector_matrix


def topN_similar_by_vector(artists_vectors_df, aggr_vector, N = 20):
    '''
    predict top-N similar artists to the vector aggr_vector
    '''
    sim = cosine_similarity(aggr_vector.reshape(1,-1), list(artists_vectors_df['vector']))[0]
    
    df = artists_vectors_df.copy()
    df['score'] = sim
    
    df = df.sort_values(by = ['score'], ascending = False)
    
    sim_artists = df['person_id'][:N]
    sim_scores = df['score'][:N]
    
    return list(sim_artists), list(sim_scores)


def recommend_by_user(model, left_out_row, user_code_dict, code_person_dict, user_artist_matrix, N=20):
    '''
    Recommend the top N items, removing the users own liked items from
    the results (implicit library do it automatically)
    '''
    user_id = left_out_row['user_id']
    user_code = user_code_dict[user_id]
    rec = model.recommend(user_code, user_artist_matrix, N=N)
    rec_persons = [code_person_dict[code] for (code,score) in rec]
    return rec_persons  

   

def recommend_by_average(test_df_row, artists_vectors_df, N=20):
    '''
    Custom recommender. Recommend most similar vectors from item factors (vectors)
    using cosine similarity
    '''
    persons_lst = test_df_row['persons_lst']

    aggr_v = aggregare_vectors(persons_lst, artists_vectors_df)

    N = len(set(persons_lst))+N

    if len(aggr_v)!=0:
        artists, scores = topN_similar_by_vector(artists_vectors_df, aggr_v, N=N)
        artists_new = []

        for ar in artists:
            if ar not in set(persons_lst):
                artists_new.append(ar)                              
        return(artists_new[:20])
    else:
        return([])

    