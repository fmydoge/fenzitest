import torch
from torch_scatter import scatter_add


def get_distance(pos, edge_index):
    return (pos[edge_index[0]] - pos[edge_index[1]]).norm(dim=-1)


def eq_transform(score_d, pos, edge_index, edge_length):
    N = pos.size(0)
    edge_length = torch.clamp(edge_length, min=1e-8)
    dd_dr = (1. / edge_length) * (pos[edge_index[0]] - pos[edge_index[1]])  # (E, 3)
    # score_pos = scatter_add(dd_dr * score_d, edge_index[0], dim=0, dim_size=N) \
    #     + scatter_add(- dd_dr * score_d, edge_index[1], dim=0, dim_size=N) # (N, 3)

    # if torch.isnan(dd_dr).any():
    #     print("⚠️ Warning: dd_dr contains NaN values!")
    #     dd_dr = torch.nan_to_num(dd_dr, nan=0.0)  # 修复 NaN

    score_pos = scatter_add(dd_dr * score_d, edge_index[0], dim=0, dim_size=N)

    # if torch.isnan(score_pos).any():
    #     print("⚠️ Warning: scatter_add output contains NaN values!")
    #     score_pos = torch.nan_to_num(score_pos, nan=0.0)  # 修复 NaN

    return score_pos


def convert_cluster_score_d(cluster_score_d, cluster_pos, cluster_edge_index, cluster_edge_length, subgraph_index):
    """
    Args:
        cluster_score_d:    (E_c, 1)
        subgraph_index:     (N, )
    """
    cluster_score_pos = eq_transform(cluster_score_d, cluster_pos, cluster_edge_index, cluster_edge_length)  # (C, 3)
    score_pos = cluster_score_pos[subgraph_index]
    return score_pos

# new func
def get_angle_index(edge_index, local_edge_mask):
    """
    计算 angle_index (3, A)。

    参数:
    - edge_index: (2, E) 图的边索引
    - local_edge_mask: (E,) 是否是局部边的掩码

    返回:
    - angle_index: (3, A) 每个键角的原子索引
    """
    if not isinstance(edge_index, torch.Tensor):
        edge_index = torch.tensor(edge_index, dtype=torch.long)  # 转换为张量

    device = edge_index.device
    local_edges = edge_index[:, local_edge_mask]  # 选择局部边 (2, E_local)

    angle_list = []
    for i in range(local_edges.shape[1]):
        node_i, node_j = local_edges[:, i]  # (i, j)
        # 找到所有与 j 连接的其他节点 k
        mask = (local_edges[0] == node_j) & (local_edges[1] != node_i)
        neighbor_k = local_edges[1, mask]

        for node_k in neighbor_k:
            angle_list.append([node_i.item(), node_j.item(), node_k.item()])

    if angle_list:
        angle_index = torch.tensor(angle_list, dtype=torch.long, device=device).T  # (3, A)
    else:
        angle_index = torch.empty((3, 0), dtype=torch.long, device=device)  # 处理无键角情况

    return angle_index

# new func
def get_dihedral_index(angle_index, edge_index, local_edge_mask):
    """
    计算 dihedral_index (4, D)。

    参数:
    - angle_index: (3, A) 每个键角的原子索引
    - edge_index: (2, E) 图的边索引
    - local_edge_mask: (E,) 是否是局部边的掩码

    返回:
    - dihedral_index: (4, D) 每个二面角的原子索引
    """
    if not isinstance(edge_index, torch.Tensor):
        edge_index = torch.tensor(edge_index, dtype=torch.long)  # 转换为张量

    device = edge_index.device
    local_edges = edge_index[:, local_edge_mask]  # (2, E_local)

    dihedral_list = []
    for i in range(angle_index.shape[1]):
        node_i, node_j, node_k = angle_index[:, i]  # (i, j, k)
        # 找到所有与 k 连接的其他节点 l
        mask = (local_edges[0] == node_k) & (local_edges[1] != node_j)
        neighbor_l = local_edges[1, mask]

        for node_l in neighbor_l:
            dihedral_list.append([node_i.item(), node_j.item(), node_k.item(), node_l.item()])

    if dihedral_list:
        dihedral_index = torch.tensor(dihedral_list, dtype=torch.long, device=device).T  # (4, D)
    else:
        dihedral_index = torch.empty((4, 0), dtype=torch.long, device=device)  # 处理无二面角情况

    return dihedral_index

def get_angle(pos, angle_index):
    """
    Args:
        pos:  (N, 3)
        angle_index:  (3, A), left-center-right.
    """
    n1, ctr, n2 = angle_index  # (A, )
    v1 = pos[n1] - pos[ctr]  # (A, 3)
    v2 = pos[n2] - pos[ctr]
    inner_prod = torch.sum(v1 * v2, dim=-1, keepdim=True)  # (A, 1)
    length_prod = torch.norm(v1, dim=-1, keepdim=True) * torch.norm(v2, dim=-1, keepdim=True)  # (A, 1)
    angle = torch.acos(inner_prod / length_prod)  # (A, 1)
    return angle


def get_dihedral(pos, dihedral_index):
    """
    Args:
        pos:  (N, 3)
        dihedral:  (4, A)
    """
    n1, ctr1, ctr2, n2 = dihedral_index  # (A, )
    v_ctr = pos[ctr2] - pos[ctr1]  # (A, 3)
    v1 = pos[n1] - pos[ctr1]
    v2 = pos[n2] - pos[ctr2]
    n1 = torch.cross(v_ctr, v1, dim=-1)  # Normal vectors of the two planes
    n2 = torch.cross(v_ctr, v2, dim=-1)
    inner_prod = torch.sum(n1 * n2, dim=1, keepdim=True)  # (A, 1)
    length_prod = torch.norm(n1, dim=-1, keepdim=True) * torch.norm(n2, dim=-1, keepdim=True)
    dihedral = torch.acos(inner_prod / length_prod)
    return dihedral
