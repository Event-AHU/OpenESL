import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
# from torch_geometric.nn import HypergraphConv
# from torch_geometric.utils import dense_to_sparse

from torch.nn.parameter import Parameter


def Eu_dis(x: np.ndarray) -> np.ndarray:
    x = np.mat(x)
    aa = np.sum(np.multiply(x, x), 1)
    ab = x * x.T
    dist_mat = aa + aa.T - 2 * ab
    dist_mat[dist_mat < 0] = 0
    dist_mat = np.sqrt(dist_mat)
    dist_mat = np.maximum(dist_mat, dist_mat.T)
    return np.asarray(dist_mat)


def feature_concat(*F_list, normal_col: bool = False) -> np.ndarray:
    features = None
    for f in F_list:
        if f is None or f == []:
            continue
        if len(f.shape) > 2:
            f = f.reshape(-1, f.shape[-1])
        if normal_col:
            f_max = np.max(np.abs(f), axis=0) + 1e-12
            f = f / f_max
        features = f if features is None else np.hstack((features, f))
    if normal_col and features is not None:
        features_max = np.max(np.abs(features), axis=0) + 1e-12
        features = features / features_max
    return features


def hyperedge_concat(*H_list):
    H = None
    for h in H_list:
        if h is None or len(h) == 0:
            continue
        if H is None:
            H = h
        else:
            if type(h) != list:
                H = np.hstack((H, h))
            else:
                tmp = []
                for a, b in zip(H, h):
                    tmp.append(np.hstack((a, b)))
                H = tmp
    return H


def construct_H_with_KNN_from_distance(
    dis_mat: np.ndarray, k_neig: int, is_probH: bool = False, m_prob: float = 1.0
) -> np.ndarray:
    n_obj = dis_mat.shape[0]
    H = np.zeros((n_obj, n_obj), dtype=np.float32)
    for center_idx in range(n_obj):
        dis_mat[center_idx, center_idx] = 0
        dis_vec = dis_mat[center_idx]
        nearest_idx = np.argsort(dis_vec)
        avg_dis = np.maximum(np.average(dis_vec), 1e-12)
        if center_idx not in nearest_idx[:k_neig]:
            nearest_idx = nearest_idx.copy()
            nearest_idx[k_neig - 1] = center_idx
        for node_idx in nearest_idx[:k_neig]:
            if is_probH:
                H[node_idx, center_idx] = np.exp(
                    - (dis_vec[node_idx] ** 2) / ((m_prob * avg_dis) ** 2)
                )
            else:
                H[node_idx, center_idx] = 1.0
    return H


def construct_H_with_KNN(
    X: np.ndarray,
    K_neigs,
    split_diff_scale: bool = False,
    is_probH: bool = False,
    m_prob: float = 1.0,
):
    if len(X.shape) != 2:
        X = X.reshape(-1, X.shape[-1])
    if isinstance(K_neigs, int):
        K_neigs = [K_neigs]

    dis_mat = Eu_dis(X)
    H_all = []
    for k in K_neigs:
        H_k = construct_H_with_KNN_from_distance(dis_mat, k, is_probH, m_prob)
        if not split_diff_scale:
            H_all = hyperedge_concat(H_all, H_k)
        else:
            H_all.append(H_k)
    return H_all


def generate_G_from_H(H, variable_weight: bool = False):
    if isinstance(H, list):
        return [generate_G_from_H(h, variable_weight) for h in H]
    else:
        return _generate_G_from_H(H, variable_weight)


def _generate_G_from_H(H: np.ndarray, variable_weight: bool = False):
    H = np.array(H, dtype=np.float32)
    n_edge = H.shape[1]
    W = np.ones(n_edge, dtype=np.float32)
    DV = np.sum(H * W, axis=1)
    DE = np.sum(H, axis=0)
    DV = np.maximum(DV, 1e-12)
    DE = np.maximum(DE, 1e-12)
    invDE = np.diag(np.power(DE, -1.0)).astype(np.float32)
    DV2  = np.diag(np.power(DV, -0.5)).astype(np.float32)
    Wmat = np.diag(W).astype(np.float32)
    Hm  = np.asarray(H, dtype=np.float32)
    HTm = Hm.T
    if variable_weight:
        DV2_H = DV2 @ Hm
        invDE_HT_DV2 = invDE @ HTm @ DV2
        return DV2_H, Wmat, invDE_HT_DV2
    else:
        G = DV2 @ Hm @ Wmat @ invDE @ HTm @ DV2
        return torch.tensor(G, dtype=torch.float32)


def build_H_and_G_from_tokens(
    tokens: torch.Tensor,
    is_probH=True,
    m_prob=1.0,
):
    """
    tokens: (B, N, d)
    返回:
      - H_list: list of H numpy
      - G: (B, N, N) torch.Tensor
    """
    B, N, d = tokens.shape
    G_list = []
    H_list = []
    for b in range(B):
        X_np = tokens[b].detach().cpu().numpy()
        H = construct_H_with_KNN(X_np, K_neigs=10, split_diff_scale=False, is_probH=is_probH, m_prob=m_prob)
        G = generate_G_from_H(H, variable_weight=False)
        G_list.append(G.unsqueeze(0))  # (1, N, N)
        H_list.append(H)
    G_batch = torch.cat(G_list, dim=0).to(tokens.device)  # (B, N, N)
    return H_list, G_batch


class HGNN(nn.Module):
    def __init__(self, in_ch, n_class, n_hid, dropout=0.5):
        super(HGNN, self).__init__()
        self.dropout = dropout
        self.hgc1 = HGNN_conv(in_ch, n_hid)
        self.hgc2 = HGNN_conv(n_hid, n_class)

    def forward(self, x, G):
        x = F.relu(self.hgc1(x, G))
        x = F.dropout(x, self.dropout)
        x = self.hgc2(x, G)
        return x


class HGNN_conv(nn.Module):
    def __init__(self, in_ft, out_ft, bias=True):
        super(HGNN_conv, self).__init__()

        self.weight = Parameter(torch.Tensor(in_ft, out_ft))
        if bias:
            self.bias = Parameter(torch.Tensor(out_ft))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x: torch.Tensor, G: torch.Tensor):
        x = x.matmul(self.weight)
        if self.bias is not None:
            x = x + self.bias
        x = G.matmul(x)
        return x