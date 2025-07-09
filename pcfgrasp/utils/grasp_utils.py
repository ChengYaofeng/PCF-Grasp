import torch
import numpy as np
from utils.pointnet2_utils import index_points
import torch.nn.functional as F

def get_bin_vals(global_config):
    """
    创建抓取的宽度
    
    参数：
        global_config {dict}
    
    返回：
        torch.constant 
    """
    bins_bounds = np.array(global_config['DATA']['labels']['offset_bins'])

    if global_config['TEST']['bin_vals'] == 'max':
        bin_vals = (bins_bounds[1:] + bins_bounds[:-1])/2 
        bin_vals[-1] = bins_bounds[-1]

    elif global_config['TEST']['bin_vals'] == 'mean':
        bin_vals = bins_bounds[1:]
    
    else:
        raise NotImplementedError
    
    if not global_config['TEST']['allow_zero_margin']:
        bin_vals = np.minimum(bin_vals, global_config['DATA']['gripper_width']-global_config['TEST']['extra_opening'])

    tc_bin_vals = torch.tensor(bin_vals, dtype=torch.float32)

    return tc_bin_vals

def build_6d_grasp(approach_dirs, base_dirs, contact_pts, thickness, use_cpu=True, gripper_depth=0.1034):
    """
    根据网络预测结果，建立6d抓取

    参数：
        approach_dirs {np.ndarray/torch.tensor} -- Nx3 approach direction vectors
        base_dirs {np.ndarray/torch.tensor} -- Nx3 base direction vectors
        contact_pts {np.ndarray/torch.tensor} -- Nx3 contact points
        thickness {np.ndarray/torch.tensor} -- Nx1 grasp width

        use_torch {bool} -- whether inputs and outputs are torch tensors (default: {False})
        gripper_depth {float} -- distance from gripper coordinate frame to gripper baseline in m (default: {0.1034})
    
    返回：
        np.ndarray -- Nx4x4 grasp poses in camera coordinates

    """

    if use_cpu:
        grasps_R = torch.stack([base_dirs, torch.cross(approach_dirs, base_dirs), approach_dirs], dim=3)
        grasps_t = contact_pts + torch.unsqueeze(thickness,2)/2 * base_dirs - gripper_depth * approach_dirs
        ones = torch.ones((contact_pts.shape[0], contact_pts.shape[1], 1, 1), dtype=torch.float32)
        zeros = torch.zeros((contact_pts.shape[0], contact_pts.shape[1], 1, 3), dtype=torch.float32)
        homog_vec = torch.cat([zeros, ones], dim=3)#.cuda()
        grasps = torch.cat([torch.cat([grasps_R, torch.unsqueeze(grasps_t, 3)], dim=3), homog_vec], dim=2)
    
    else:
        grasps_R = torch.stack([base_dirs, torch.cross(approach_dirs, base_dirs), approach_dirs], dim=3)
        grasps_t = contact_pts + torch.unsqueeze(thickness,2)/2 * base_dirs - gripper_depth * approach_dirs
        ones = torch.ones((contact_pts.shape[0], contact_pts.shape[1], 1, 1), dtype=torch.float32)
        zeros = torch.zeros((contact_pts.shape[0], contact_pts.shape[1], 1, 3), dtype=torch.float32)
        homog_vec = torch.cat([zeros, ones], dim=3).cuda()
        grasps = torch.cat([torch.cat([grasps_R, torch.unsqueeze(grasps_t, 3)], dim=3), homog_vec], dim=2)
    
    return grasps

def multi_bin_labels(cont_labels, bin_boundaries):
    """
    计算抓取标签

    Arguments:
        cont_labels {torch.Variable} -- continouos labels
        bin_boundaries {list} -- bin boundary values

    Returns:
        torch.Variable -- one/multi hot bin labels
    """ 
    # print(cont_labels.shape) torch.Size([1, 1024, 1])
    bins = []
    for b in range(len(bin_boundaries) - 1):
        bins.append(torch.logical_and(torch.greater_equal(cont_labels, bin_boundaries[b]), torch.less(cont_labels, bin_boundaries[b + 1])))
    
    multi_hot_labels = torch.cat(bins, dim=2)
    multi_hot_labels = multi_hot_labels.type(torch.float32)

    return multi_hot_labels

def compute_labels(pos_contact_pts_mesh, pos_contact_dirs_mesh, pos_contact_approaches_mesh, pos_finger_diffs, pc_cam_pl, camera_pose_pl, global_config):
    """
    在点云mesh上建立的抓取标签，从相机坐标系下通过最大进邻接触点获得的
    所有点如果没有成功抓取接触都被考虑为失败的抓取点

    参数：
        pos_contact_pts_mesh {torch.constant} -- positive contact points on the mesh scene (Mx3)
        pos_contact_dirs_mesh {torch.constant} -- respective contact base directions in the mesh scene (Mx3)
        pos_contact_approaches_mesh {torch.constant} -- respective contact approach directions in the mesh scene (Mx3)
        pos_finger_diffs {torch.constant} -- respective grasp widths in the mesh scene (Mx1)
        pc_cam_pl {torch.placeholder} -- bxNx3 rendered point clouds
        camera_pose_pl {torch.placeholder} -- bx4x4 camera poses(1x4x4)
        global_config {dict} -- global config
    返回：
        [dir_labels_pc_cam, offset_labels_pc, grasp_success_labels_pc, approach_labels_pc_cam] --每个点的成功标签和接触点的成功姿态标签
    """
    # print(pos_finger_diffs.shape)
    # print(torch.sum(pos_contact_dirs_mesh,dim=0))
    # print(torch.sum(pos_contact_approaches_mesh, dim=0))
    # print(pos_contact_dirs_mesh.shape)
    # print(pos_contact_pts_mesh.shape)

    label_config = global_config['DATA']['labels']
    # model_config = global_config['MODEL']

    nsample = label_config['k']
    radius = label_config['max_radius']
    filter_z = label_config['filter_z']
    z_val = label_config['z_val']

    xyz_cam = pc_cam_pl[:,:,:3]

    # pad_homog = torch.ones((xyz_cam.shape[0],xyz_cam.shape[1], 1))
    # print("zhixing")
    # print(camera_pose_pl.shape)

    # pc_mesh = torch.matmul(torch.cat([xyz_cam, pad_homog], 2), torch.transpose(torch.inverse(camera_pose_pl), 2, 1))[:,:,:3]
    #相机坐标系下的点
    pad_homog2 = torch.ones((pos_contact_dirs_mesh.shape[0], pos_contact_dirs_mesh.shape[1], 1))

    # print(torch.transpose(camera_pose_pl[:,:3,:3], 2, 1).shape) #233
    contact_point_dirs_batch_cam = torch.matmul(pos_contact_dirs_mesh, torch.transpose(camera_pose_pl[:,:3,:3], 2, 1))[:,:,:3]
    # 接触点乘R矩阵
    # print(torch.sum(contact_point_dirs_batch_cam, dim=1))

    pos_contact_approaches_batch_cam = torch.matmul(pos_contact_approaches_mesh, torch.transpose(camera_pose_pl[:,:3,:3], 2, 1))[:,:,:3]

    contact_point_batch_cam = torch.matmul(torch.cat([pos_contact_pts_mesh, pad_homog2], 2), torch.transpose(camera_pose_pl, 2, 1))[:,:,:3]

    if filter_z:
        dir_filter_passed = torch.repeat_interleave(torch.greater(contact_point_dirs_batch_cam[:,:,2:3], torch.tensor([z_val])), 3, dim=2)
        pos_contact_pts_mesh = torch.where(dir_filter_passed, pos_contact_pts_mesh, torch.ones_like(pos_contact_pts_mesh)*100000)

    # print(xyz_cam.shape)  #torch.Size([1, 1024, 3])
    # print(contact_point_batch_cam.shape)  #torch.Size([1, 16000, 3])
    squared_dists_all = torch.sum((torch.unsqueeze(contact_point_batch_cam,1)-torch.unsqueeze(xyz_cam,2))**2,dim=3)
    # print(squared_dists_all.shape)  #torch.Size([1, 1024, 16000])
    neg_squared_dists_k, close_contact_pt_idcs = torch.topk(-squared_dists_all, k=nsample, sorted=False)
    # print(close_contact_pt_idcs)
    squared_dists_k = -neg_squared_dists_k


    grasp_success_labels_pc = torch.less(torch.mean(squared_dists_k, dim=2), radius*radius).type(torch.float32) # (batch_size, num_point)
    # print(torch.sum(grasp_success_labels_pc))
    grouped_dirs_pc_cam = index_points(contact_point_dirs_batch_cam, close_contact_pt_idcs)
    grouped_approaches_pc_cam = index_points(pos_contact_approaches_batch_cam, close_contact_pt_idcs)
    grouped_offsets = index_points(torch.unsqueeze(pos_finger_diffs,2), close_contact_pt_idcs)

    dir_labels_pc_cam = F.normalize(torch.mean(grouped_dirs_pc_cam, dim=2),dim=2) # (batch_size, num_point, 3)
    approach_labels_pc_cam = F.normalize(torch.mean(grouped_approaches_pc_cam, dim=2),dim=2) # (batch_size, num_point, 3)
    offset_labels_pc = torch.mean(grouped_offsets, dim=2)


    if global_config['MODEL']['bin_offsets']:
        offset_labels_pc = torch.abs(offset_labels_pc)
        offset_labels_pc = multi_bin_labels(offset_labels_pc, global_config['DATA']['labels']['offset_bins'])
    

    # print(torch.sum(dir_labels_pc_cam,dim=1))

    return dir_labels_pc_cam, offset_labels_pc, grasp_success_labels_pc, approach_labels_pc_cam