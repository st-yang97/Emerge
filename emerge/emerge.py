from __future__ import annotations
from typing import Dict, List, Optional, Sequence
import warnings
import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc
from scipy.spatial import distance_matrix
from sklearn.neighbors import BallTree, NearestNeighbors
import ot
from ot.bregman import sinkhorn, sinkhorn2

from .utils.func0905 import (
    pre_label_exp,
    envpca,
    envpcabignome,
    tensor_square_loss_adjusted,
)

warnings.filterwarnings("ignore")


def run_emerge(
    *,
    fish_exp_raw: pd.DataFrame,        # cells × genes
    fish_type: Optional[pd.DataFrame] = None,          # index:cells
    fish_loc: pd.DataFrame,            # index:cells default 'X','Y' 'x_centroid','y_centroid'
    scrna_sc_raw: pd.DataFrame,        # cells × genes
    scrna_type_raw: pd.DataFrame,       
    
    type_col: str = "cell_types",      
    x_col: str = "X", y_col: str = "Y",
    layeruse: str = "Annotation",

    train_genes: Optional[Sequence[str]] = None,
    test_genes: Optional[Sequence[str]] = None,
   

    k_label_list: Sequence[int] = (10, 20, 30, 40),
    k_knn_list: Sequence[int] = (10, 20, 30, 40),
    mruse: Sequence[str] = ("braycurtis","correlation","cosine","chebyshev","canberra","euclidean"),
    type_threod: float = 0.5,          
    disuse_list: Optional[Sequence[float]] = None, 
    alpha_linear: float = 0.5,
    tol: float = 1e-9,
    max_iter: int = 1000,
    aca_use: bool = True,
    the_cut0: float = 0.1,
    epsilon: float = 0.01,
    cell_or_cluster: str = "cell",     # "cell" "cluster"
    env_backend: str = "standard",     # "standard"=envpca "big"=envpcabignome for big data
    big_env_processes: int = 4,
) -> Dict[str, object]:
   
    if disuse_list is None:
        disuse_list = [55, 110, 165, 220, 275]
    
   

    genes_scrna = set(scrna_sc_raw.columns)
    genes_fish  = set(fish_exp_raw.columns)
    common_gene = sorted(list(genes_scrna & genes_fish))
    pre_gene    = sorted(list(genes_scrna - genes_fish))

   
    if train_genes is not None and test_genes is not None:
        train_gene = np.array([g for g in train_genes if g in common_gene])
        test_gene  = np.array([g for g in test_genes  if g in genes_scrna])

    else:
        train_gene = np.array(common_gene)
        test_gene  = np.array(pre_gene)

    # -------------------- 2) normalize_total + log1p --------------------
    #scrna_sc_raw_ann = ad.AnnData(scrna_sc_raw)
    scrna_tr = ad.AnnData(scrna_sc_raw[common_gene]); sc.pp.normalize_total(scrna_tr); sc.pp.log1p(scrna_tr)
    fish_log  = ad.AnnData(fish_exp_raw[common_gene]); sc.pp.normalize_total(fish_log);  sc.pp.log1p(fish_log)
    # 
    scrna_te = None
    if set(test_gene).issubset(set(common_gene)):
        print('cross-validation')
        scrna_te = ad.AnnData(scrna_sc_raw[common_gene]); sc.pp.normalize_total(scrna_te); sc.pp.log1p(scrna_te)
    else:
        scrna_te = ad.AnnData(scrna_sc_raw[pre_gene]); sc.pp.normalize_total(scrna_te); sc.pp.log1p(scrna_te)
        
    if fish_type is None:
        fish_type = pd.DataFrame({"cell_types": ["Unknown"] * fish_exp_raw.shape[0]},  index=fish_exp_raw.index)

    fish_tr = fish_log[:,train_gene]
    fish_type_aligned = fish_type.copy()
    fish_type_aligned.index = fish_tr.to_df().index
    fish_loc_val = fish_loc[[x_col, y_col]].values
    loc_fish = distance_matrix(fish_loc_val, fish_loc_val)

    # 
    pre_sc_test = pd.DataFrame(
        np.zeros((fish_exp_raw.shape[0], len(test_gene))),
        index=fish_exp_raw.index, columns=test_gene
    )

    # 
    type_index_list_cross: List[List[int]] = []
    type_stay_list_cross: List[List[str]] = []
    eva_type_use_list_cross: List[List[pd.DataFrame]] = []
    weighted_average_types_cross: List[pd.DataFrame] = []
    eva_type_use_list_cross_nos: List = []
        
    # ----------------------------------------
    weighted_acc = None
    for mr in mruse:
        for k_label in k_label_list:
            _, weighted0, _ = pre_label_exp(
                scrna_tr[:, train_gene],scrna_tr[:, train_gene],                
                fish_tr,
                scrna_type_raw,
                layeruse,
                k_label,
                mr
            )
            if weighted_acc is None:
                weighted_acc = weighted0.values.astype(float)
                wa_cols = list(weighted0.columns); wa_idx  = list(weighted0.index)
            else:
                weighted_acc += weighted0.values

    weighted_average_types = pd.DataFrame(
        weighted_acc / (len(mruse) * len(k_label_list)),
        index=wa_idx, columns=wa_cols
    )
    weighted_average_types_cross.append(weighted_average_types)

    # ----------------------------------------
    if cell_or_cluster == "cluster":
        pre_cell_types = weighted_average_types.idxmax(axis=1).to_numpy()
        fish_type_new  = fish_type_aligned[type_col].copy()
        for fishtype in np.unique(fish_type_aligned[type_col].values):
            idx = fish_type_aligned.index.values[fish_type_aligned[type_col].values == fishtype]
            mapped = pre_cell_types[fish_type_aligned[type_col].values == fishtype]
            fish_type_new.loc[idx] = pd.Series(mapped).value_counts().idxmax()
    
        fish_type_new_sc = weighted_average_types.max(axis=1)  
        fish_type_old    = fish_type_aligned[type_col].copy()
        
    elif cell_or_cluster == "cell":       
        predicted_labels  = np.array(weighted_average_types.idxmax(axis=1))    
        fish_type_new = fish_type_aligned[type_col].copy()
        fish_type_new[:] = predicted_labels
        fish_type_new_sc = weighted_average_types.max(axis=1)
        fish_type_old = fish_type_aligned[type_col].copy()

    # ----------------------------------------
    eva_type_list_list: List[List[pd.DataFrame]] = []
    if env_backend == "standard":
        inter_score_py = pd.get_dummies(fish_type_new.to_frame()[type_col])
        for disuse in disuse_list:
            type_uset, eva_type_lista = envpca(inter_score_py, disuse, loc_fish, the_cut0)
            eva_type_list_list.append(eva_type_lista)
    elif env_backend == "big":
        tree = BallTree(fish_loc_val, metric="euclidean")
        for disuse in disuse_list:
            type_uset, eva_type_lista = envpcabignome(
                fish_type_new, tree, fish_loc_val, disuse, the_cut0, num_processes=big_env_processes
            )
            eva_type_list_list.append(eva_type_lista)
    else:
        raise ValueError("env_backend must be 'standard' or 'big'")

    # ----------------------------------------
    fish_type_fi = fish_type_new[fish_type_new_sc.values >= type_threod]
    max_index    = fish_type_fi.index.values
    fish_train_high = fish_tr[max_index, :]

    iii = 0
    type_index_list: List[int] = []
    type_stay_list: List[str] = []
    eva_type_use_list: List[pd.DataFrame] = []

    for type_hi in type_uset:
        print(type_hi)
        fish_this = (fish_type_fi == type_hi)
        if fish_this.sum() > 0:
            type_stay_list.append(type_hi)

            rna_this = (scrna_type_raw[layeruse] == type_hi)
            fi_coexp_this = fish_train_high[fish_this, :]
            fish_this_cell = fi_coexp_this.to_df().index

            t_list, loss_list = [], []
            for eva_type_list in eva_type_list_list:
                env_feat_this = eva_type_list[iii].loc[fish_this_cell, :].values

                sc_coexp_his = scrna_tr[rna_this, train_gene]

                sc_hvg_this = ad.AnnData(scrna_sc_raw.loc[scrna_tr.obs_names[rna_this], :])
                sc.pp.normalize_total(sc_hvg_this); sc.pp.log1p(sc_hvg_this)
                sc.pp.highly_variable_genes(sc_hvg_this, n_top_genes=500)
                hvg_mask = sc_hvg_this.var["highly_variable"]
                hvg_exp_this = sc_hvg_this[:, hvg_mask].X
                hvg_exp_use = hvg_exp_this.toarray() if hasattr(hvg_exp_this, "toarray") else hvg_exp_this

                q_sc_knn = np.zeros(sc_coexp_his.shape[0])
                for k_inty in k_knn_list:
                    k_use = min(k_inty, sc_coexp_his.shape[0])
                    for mr in mruse:
                        nbrs = NearestNeighbors(n_neighbors=k_use, algorithm="auto", metric=mr).fit(sc_coexp_his.to_df())
                        _, indices = nbrs.kneighbors(fi_coexp_this.to_df())
                        n_fi = fi_coexp_this.shape[0]
                        n_sc = sc_coexp_his.shape[0]
                        W_m = np.zeros((n_fi, n_sc))
                        row_idx = np.repeat(np.arange(n_fi), k_use)
                        col_idx = indices.flatten()
                        W_m[row_idx, col_idx] = 1
                        q_sc_knn0 = (W_m != 0).sum(0)
                        q_sc_knn0 = q_sc_knn0 / q_sc_knn0.sum()
                        q_sc_knn += q_sc_knn0
                q_sc_knn = q_sc_knn / (len(mruse) * len(k_knn_list))
                q_sc_jun = np.ones_like(q_sc_knn) / len(q_sc_knn)
                q_sc = (q_sc_knn + q_sc_jun) / 2
                p_fish = np.ones(fi_coexp_this.shape[0]) / fi_coexp_this.shape[0]

                C1_dist = np.zeros((env_feat_this.shape[0], env_feat_this.shape[0]))
                for mr in mruse:
                    C1_0 = ot.dist(env_feat_this, env_feat_this, metric=mr)
                    if np.isnan(C1_0).sum() != 0:
                        C1_0[np.isnan(C1_0)] = np.nanmax(C1_0)
                    if C1_0.max() != 0:
                        C1_0 = C1_0 / C1_0.max()
                    C1_dist += C1_0
                C1_dist /= len(mruse)

                C2_dist = np.zeros((hvg_exp_use.shape[0], hvg_exp_use.shape[0]))
                for mr in mruse:
                    C2_0 = ot.dist(hvg_exp_use, hvg_exp_use, metric=mr)
                    if np.isnan(C2_0).sum() != 0:
                        C2_0[np.isnan(C2_0)] = np.nanmax(C2_0)
                    if C2_0.max() != 0:
                        C2_0 = C2_0 / C2_0.max()
                    C2_dist += C2_0
                C2_dist /= len(mruse)

                X_fi = fi_coexp_this.X.toarray() if hasattr(fi_coexp_this.X, "toarray") else fi_coexp_this.X
                X_sc = sc_coexp_his.X.toarray() if hasattr(sc_coexp_his.X, "toarray") else sc_coexp_his.X
                cost_mat = np.zeros((X_fi.shape[0], X_sc.shape[0]))
                for mr in mruse:
                    C0 = ot.dist(X_fi, X_sc, metric=mr)
                    if np.isnan(C0).sum() != 0:
                        C0[np.isnan(C0)] = np.nanmax(C0)
                    C0 = C0 / C0.max()
                    cost_mat += C0
                cost_mat /= len(mruse)

                # Sinkhorn / FGW
                if alpha_linear == 1 or env_feat_this.shape[0] == 1:
                    T = sinkhorn(p_fish, q_sc, cost_mat, epsilon)
                    tls = sinkhorn2(p_fish, q_sc, cost_mat, epsilon)
                else:
                    T = sinkhorn(p_fish, q_sc, cost_mat, epsilon)
                    cpt, err = 0, 1
                    while (err > tol) and (cpt < max_iter):
                        Tprev = T
                        tens = tensor_square_loss_adjusted(C1_dist, C2_dist, T)
                        aca = np.median(tens) / np.median(cost_mat) if aca_use else 1
                        tens_all = (1 - alpha_linear) * tens + alpha_linear * aca * cost_mat
                        T = sinkhorn(p_fish, q_sc, tens_all, epsilon)
                        if cpt % 10 == 0:
                            err = np.linalg.norm(T - Tprev)
                        cpt += 1
                    tls = sinkhorn2(p_fish, q_sc, tens_all, epsilon)

                t_list.append(T); loss_list.append(tls)

            min_index = int(np.argmin(loss_list))
            type_index_list.append(min_index)
            eva_type_use_list.append(eva_type_list_list[min_index][iii])#.loc[fish_this_cell, :])

            T_best = t_list[min_index]
            T_nor  = T_best / T_best.sum(1)[:, None]

            X_sc_te = scrna_te[rna_this, test_gene].X
            if hasattr(X_sc_te, "toarray"): X_sc_te = X_sc_te.toarray()
            pred_block = T_nor.dot(X_sc_te)
            pre_sc_test.loc[fi_coexp_this.obs_names, test_gene] = pred_block

        iii += 1

    type_index_list_cross.append(type_index_list)
    type_stay_list_cross.append(type_stay_list)
    eva_type_use_list_cross.append(eva_type_use_list)

    # ----------------------------------------
    fish_type_oldlow = fish_type_old[fish_type_new_sc.values < type_threod]
    low_index = fish_type_oldlow.index.values
    if len(low_index) > 0:
        print(len(low_index) )

        fish_train_low = fish_tr[low_index, :]
        per_low = np.zeros((len(low_index), len(test_gene)))
        high_data_test = pre_sc_test.loc[fish_train_high.obs_names, test_gene].values

        for k_low in k_label_list:
            for mus in mruse:
                knn_model = NearestNeighbors(n_neighbors=k_low, algorithm="auto", metric=mus).fit(
                    fish_train_high.to_df()
                )
                distances, indices = knn_model.kneighbors(fish_train_low.to_df())
                max_distance = np.nanmax(distances)
                distances[np.isnan(distances)] = max_distance
                distances = distances / max_distance
                sigma = 0.1
                dis_to_simi = np.exp(-distances / sigma)
                weighted_sum = np.sum(high_data_test[indices] * dis_to_simi[:, :, None], axis=1)
                sum_of_weights = dis_to_simi.sum(1)[:, None]
                per_low += (weighted_sum / sum_of_weights)

        per_low = per_low / (len(mruse) * len(k_label_list))
        pre_sc_test.loc[fish_train_low.obs_names, test_gene] = per_low

    # ----------------------------------------
    return {
        "pre_sc_test": pre_sc_test,
        "type_index_list_cross": type_index_list_cross,
        "type_stay_list_cross": type_stay_list_cross,
        "eva_type_use_list_cross": eva_type_use_list_cross,
        "weighted_average_types_cross": weighted_average_types_cross,
        "eva_type_use_list_cross_nos": eva_type_use_list_cross_nos,
    }
