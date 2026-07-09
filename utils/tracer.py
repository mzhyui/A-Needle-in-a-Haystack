import copy
import numpy as np
from sklearn.decomposition import PCA
from scipy.stats import norm
from operator import itemgetter


def processing_updates(local_updated_weights, user_ids):
    layers = [key for key in local_updated_weights[0].keys()]
    client_weights = np.zeros((len(local_updated_weights), 1))
    for layer in layers:
        layer_weights = []
        for user_id in user_ids:
            client_update = local_updated_weights[user_id][layer]
            client_value = client_update.ravel().cpu()
            layer_weights.append(client_value)
        client_weights = np.concatenate(
            (client_weights, np.array(layer_weights)), axis=1)

    client_weights_flattern = client_weights[:, 1:]
    return client_weights_flattern

def detecting_expert_1_2(client_weights_flattern, user_ids, threshold=0.3, top_k=5, detecting_method=None):
    client_weights = copy.deepcopy(client_weights_flattern)
    size_th = 100000
    pca = PCA(n_components=1)
    n, d = client_weights.shape
    if d > size_th:
        idx = np.sort(np.random.choice(d, size_th, replace=False))
        client_weights = client_weights[:, idx]

    if detecting_method == 'sign':
        clients_sort = np.sign(np.array(client_weights))
    else:
        clients_sort = np.argsort(np.argsort(
            np.array(client_weights), axis=0), axis=0)
    
    clients_sort = clients_sort - np.mean(clients_sort, 0)
    sore_list = pca.fit_transform(clients_sort)
    scores = MadScore(sore_list)  # 1.6 sign  10 add noise
    # return set(np.array(user_ids)[(scores > threshold).ravel()])
    return np.array(user_ids)[np.argsort(scores.ravel())[-top_k:][::-1]]

def detecting_expert_3(local_updated_weights, user_ids, threshold=0.5, top_k=5):
    layers = [key for key in local_updated_weights[0].keys()]
    layer_weights = []

    # 以第一个client为基准，识别方差不为0的层
    reference_client = user_ids[0]
    valid_layers = []
    
    for layer_name in layers:
        weight = local_updated_weights[reference_client][layer_name]
        
        if not weight.dtype.is_floating_point:
            weight = weight.float()
        
        if weight.numel() <= 1:
            variance = 0.0  # Set variance to 0 for single-element tensors
        else:
            variance = weight.var().item()
        
        if variance > 1e-8:  # 设置一个小的阈值避免数值精度问题
            valid_layers.append(layer_name)

    for client_id in user_ids:
        tmp = itemgetter(
            *list(valid_layers)[-2:])(local_updated_weights[client_id])
        layer_weights.append(np.concatenate([a.ravel().cpu() for a in tmp]))

    X = np.array(layer_weights)
    pca = PCA(n_components=1)
    pca.fit(X)
    X_new = pca.transform(X)
    # inv_cov_matrix = np.linalg.inv(np.cov(X_new, rowvar=False))
    # mean_distr = X_new.mean(axis=0)
    # values = np.array([mahalanobis(value, mean_distr, inv_cov_matrix) for value in X_new])
    values = X_new.ravel()
    score = MadScore(values)
    # pre_attacker = set(np.array(user_ids)[abs(score) > threshold])
    pre_attacker = np.array(user_ids)[np.argsort(score.ravel())[-top_k:][::-1]]
    return pre_attacker

def MadScore(distance):
    median = np.median(distance)
    abs_dev = distance - median
    mad = np.median(abs(abs_dev))  # *b
    mod_z_score = norm.ppf(0.75) * abs_dev / mad
    # result = mod_z_score>3
    return mod_z_score  # result