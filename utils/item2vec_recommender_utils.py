import numpy as np

def get_similar_artists(model, artist_vector, people_dict, n = 20):
    '''
    Get top-N artists similar by vector to artist_vector
    '''
    
    # extract most similar products for the input vector
    try:
        most_sim = model.similar_by_vector(artist_vector, topn = n+1)[1:]
    except ValueError:
        return [], []

    
    # extract name and similarity score of the similar artists
    most_sim_pairs = []
    
    people_ids = []
    
    for s in most_sim:     
        pair = (people_dict[s[0]], s[1])
        most_sim_pairs.append(pair)
        
        people_ids.append(s[0])
        
    return most_sim_pairs, people_ids  



def aggregate_vectors(model, people_lst):
    '''
    Claculates an average of artists vectors
    for a list of artist that a user listened to 
    '''
    person_vec = []
    for person_id in set(people_lst):
        try:
            person_vec.append(model[person_id])
        except KeyError:
            continue  
            
    if len(person_vec) > 1:
        return np.mean(person_vec, axis=0)
    else:
        return person_vec


def recommend(model, df_grouped, people_dict):
    '''
    Recommends top 20 artists to the user based
    on the previously listened tracks
    '''
    top_20_recommended_names = []
    top_20_recommended_ids = []
    for i, row in df_grouped.iterrows():
        user_id = row['user_id']
        persons_lst = row['persons_lst']
        
        persons_aggr = aggregate_vectors(model, persons_lst)
        
        names, ids = get_similar_artists(model, persons_aggr, people_dict)
        
        top_20_recommended_names.append(names)
        top_20_recommended_ids.append(ids)
        
    return top_20_recommended_names, top_20_recommended_ids  



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


def hit_rate_evaluate(model, df_grouped, left_out_dict, people_dict):
    '''
    hit rate evaluation
    '''
    top_names, top_ids = recommend(model, df_grouped, people_dict)
    
    print('Recommendations calculated')
    
    users_lst = list(df_grouped['user_id'])
    
    print('Starting hit rate calculation')
    
    hit_rate = calculate_hit_rate(left_out_dict, users_lst, top_ids)
    return hit_rate