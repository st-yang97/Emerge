from sklearn.neighbors import NearestNeighbors
import numpy  as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from multiprocessing import Pool
from collections import Counter

  

def pre_label_exp(scrna_com,scrna_test,fish_log_all_bat,scrna_type_raw,layeruse,k_label,
              mruse):    
    scrna_com_re = scrna_com.to_df()
    scrna_type_use = scrna_type_raw
    spot_labels_b = scrna_type_use[layeruse].astype('category').cat.codes.to_numpy()
    spot_labels_b = spot_labels_b.reshape(-1, 1) 
    def _make_ohe():
        try:
            # sklearn >= 1.2
            return OneHotEncoder(sparse_output=False, handle_unknown='ignore', dtype=np.float32)
        except TypeError:
            # sklearn < 1.2
            return OneHotEncoder(sparse=False, handle_unknown='ignore', dtype=np.float32)
    encoder = _make_ohe()
    subspot_clu_n_mat = encoder.fit_transform(spot_labels_b)
    if hasattr(subspot_clu_n_mat, "toarray"):
        subspot_clu_n_mat = subspot_clu_n_mat.toarray()
    label_mapping = dict(enumerate(scrna_type_use[layeruse].astype('category').cat.categories))
    label_df = pd.DataFrame(list(label_mapping.items()), columns=['Label_Num', 'Cell_Type'])
    subspot_clu_n_mat = pd.DataFrame(subspot_clu_n_mat,columns = label_df['Cell_Type'].values)
    knn_model = NearestNeighbors(n_neighbors=k_label, algorithm='auto', metric= mruse).fit(
        scrna_com_re)  
    distances, indices = knn_model.kneighbors(fish_log_all_bat.to_df())                
    max_distance = np.nanmax(distances)
    distances[np.isnan(distances)] = max_distance  
    distances = distances/max_distance                 
    sigma = 0.1
    weights_matrix = np.exp(-distances /  sigma)
    total_weights = np.sum(weights_matrix, axis=1)
    total_weights[total_weights == 0] = 1
    weights_matrix /= total_weights.reshape(-1, 1)
    weighted_average_types = np.sum(subspot_clu_n_mat.values[indices] * weights_matrix[:, :, np.newaxis], axis=1)
    total_weights = np.sum(weighted_average_types, axis=1)
    total_weights[total_weights == 0] = 1
    weighted_average_types /= total_weights.reshape(-1, 1)           
    weighted_average_types = pd.DataFrame(weighted_average_types,
                                          index =fish_log_all_bat.to_df().index,
                                          columns = subspot_clu_n_mat.columns) 
              
    pre_cell_types  = weighted_average_types.idxmax(axis=1)
    predicted_labels = np.array(pre_cell_types)
    sigma = 0.1
    dis_to_simi = np.exp(-distances /  sigma)                  
    scrna_test  = scrna_test.to_df().values
    weighted_sum = np.sum(scrna_test[indices] * dis_to_simi[:, :, np.newaxis], axis=1)       
    sum_of_weights = np.sum(dis_to_simi, axis=1)[:, np.newaxis] 
    per_exp = weighted_sum / sum_of_weights
    return predicted_labels,weighted_average_types,per_exp








def envpca(type_mat,disuse, celldis_mat,the_cut0):     
    neighbor_mask = celldis_mat < disuse
    np.fill_diagonal(neighbor_mask, False) 
    weights = np.where(neighbor_mask, 1 , 0)  
    weights_sum = np.sum(weights, axis=1)
    weights = weights / (weights_sum[:, np.newaxis]+1e-16)
    env_feat = np.dot(weights, np.array(type_mat))
    env_feat = pd.DataFrame(env_feat,index = type_mat.index,columns = type_mat.columns)  
    eva_type_list = list()   
    high_type = type_mat.idxmax(axis=1)      
    type_use = np.unique(high_type.values)
    eva_type_list = list()
    for type_here in type_use: 
        data_this = high_type == type_here   
        env_feat_this = env_feat.loc[env_feat.index.values[data_this.values],:]
        non_zero_ratio = (env_feat_this != 0).mean(axis=0)
        type_throw = non_zero_ratio[non_zero_ratio < the_cut0]
        type_throw = type_throw.index.values
        env_feat_this = env_feat_this.drop(columns=type_throw)
        #print(env_feat_this.shape)
        nor_env_feat_this = env_feat_this.div(env_feat_this.sum(axis=1), axis=0)                                    
        eva_type_list.append(nor_env_feat_this)        
    return type_use,eva_type_list




def tensor_square_loss_adjusted(C1, C2, T):
    C1 = np.asarray(C1, dtype=np.float64)
    C2 = np.asarray(C2, dtype=np.float64)
    T = np.asarray(T, dtype=np.float64)
    def f1(a):
        return (a**2) / 2
    def f2(b):
        return (b**2) / 2
    def h1(a):
        return a
    def h2(b):
        return b
    tens = -np.dot(h1(C1), T).dot(h2(C2).T) 
    tens -= tens.min()
    return tens


def get_neighbors_and_distances_for_cell(args):
    point_idx, tree, points, cell_types, disuse = args
    neighbors_idx = tree.query_radius([points[point_idx]], r=disuse)[0]
    neighbors_idx = neighbors_idx[neighbors_idx != point_idx]    
    if len(neighbors_idx) > 0:
        distances = np.linalg.norm(points[neighbors_idx] - points[point_idx], axis=1)
        neighbor_types = cell_types[neighbors_idx]
        type_counts = Counter(neighbor_types)
        return (point_idx, neighbors_idx, distances, type_counts)
    else:
        return (point_idx, [], [], Counter())
    
    
    

def process_all_cells(tree, points, cell_types, disuse, num_processes=4):
    args = [(i, tree, points, cell_types, disuse) for i in range(len(points))]
    with Pool(num_processes) as pool:
        results = pool.map(get_neighbors_and_distances_for_cell, args)
    return results




def envpcabignome(fish_type_new,tree,fish_loc_val,disuse,the_cut0,num_processes=4):
    
    cell_types = fish_type_new.values
    num_cells = len(fish_type_new)
    results = process_all_cells(tree, fish_loc_val, cell_types, disuse, num_processes)  # Adjust num_processes based on your CPU cores   
    unique_types = sorted(set(cell_types))
    env_feat = pd.DataFrame(0, index=range(num_cells), columns=unique_types)   
    for result in results:
        point_idx, _, _, type_counts = result
        total_neighbors = sum(type_counts.values())
        if total_neighbors > 0:
            for cell_type, count in type_counts.items():
                env_feat.at[point_idx, cell_type] = count / total_neighbors
    
    env_feat = env_feat.rename(index=dict(zip(env_feat.index, fish_type_new.index)))
    high_type = fish_type_new
    type_use = np.unique(high_type.values)
    eva_type_list = list()
    for type_here in type_use: 
        data_this = high_type == type_here   
        env_feat_this = env_feat.loc[env_feat.index.values[data_this.values],:]
        non_zero_ratio = (env_feat_this != 0).mean(axis=0)
        type_throw = non_zero_ratio[non_zero_ratio < the_cut0]
        type_throw = type_throw.index.values
        env_feat_this = env_feat_this.drop(columns=type_throw)
        #print(env_feat_this.shape)
        
        nor_env_feat_this = env_feat_this.div(env_feat_this.sum(axis=1), axis=0)        
        eva_type_list.append(nor_env_feat_this)
        
    return type_use,eva_type_list

           