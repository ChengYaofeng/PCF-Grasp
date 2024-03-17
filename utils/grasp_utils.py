import torch
import numpy as np
from utils.pointnet2_utils import index_points
import torch.nn.functional as F

def get_bin_vals(global_config):
    """
    parameter:
        global_config {dict}
    
    return:
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

def build_6d_grasp(approach_dirs, base_dirs, contact_pts, thickness, use_gpu=True, gripper_depth=0.1034):
    """
    6d grasp building

    parameter:
        approach_dirs {np.ndarray/torch.tensor} -- Nx3 approach direction vectors
        base_dirs {np.ndarray/torch.tensor} -- Nx3 base direction vectors
        contact_pts {np.ndarray/torch.tensor} -- Nx3 contact points
        thickness {np.ndarray/torch.tensor} -- Nx1 grasp width

        use_torch {bool} -- whether inputs and outputs are torch tensors (default: {False})
        gripper_depth {float} -- distance from gripper coordinate frame to gripper baseline in m (default: {0.1034})
    
    return:
        np.ndarray -- Nx4x4 grasp poses in camera coordinates

    """

    if use_gpu:
        grasps_R = torch.stack([base_dirs, torch.cross(approach_dirs, base_dirs), approach_dirs], dim=3)
        grasps_t = contact_pts + torch.unsqueeze(thickness,2)/2 * base_dirs - gripper_depth * approach_dirs
        ones = torch.ones((contact_pts.shape[0], contact_pts.shape[1], 1, 1), dtype=torch.float32)
        zeros = torch.zeros((contact_pts.shape[0], contact_pts.shape[1], 1, 3), dtype=torch.float32)
        homog_vec = torch.cat([zeros, ones], dim=3).cuda()
        grasps = torch.cat([torch.cat([grasps_R, torch.unsqueeze(grasps_t, 3)], dim=3), homog_vec], dim=2)
        
    else:       
        grasps_R = torch.stack([base_dirs, torch.cross(approach_dirs, base_dirs), approach_dirs], dim=3)
        grasps_t = contact_pts + torch.unsqueeze(thickness,2)/2 * base_dirs - gripper_depth * approach_dirs
        ones = torch.ones((contact_pts.shape[0], contact_pts.shape[1], 1, 1), dtype=torch.float32)
        zeros = torch.zeros((contact_pts.shape[0], contact_pts.shape[1], 1, 3), dtype=torch.float32)
        homog_vec = torch.cat([zeros, ones], dim=3)
        grasps = torch.cat([torch.cat([grasps_R, torch.unsqueeze(grasps_t, 3)], dim=3), homog_vec], dim=2)
    
    return grasps

def multi_bin_labels(cont_labels, bin_boundaries):
    """
    Arguments:
        cont_labels {torch.Variable} -- continouos labels
        bin_boundaries {list} -- bin boundary values

    Returns:
        torch.Variable -- one/multi hot bin labels
    """ 
    bins = []
    for b in range(len(bin_boundaries) - 1):
        bins.append(torch.logical_and(torch.greater_equal(cont_labels, bin_boundaries[b]), torch.less(cont_labels, bin_boundaries[b + 1])))
    
    multi_hot_labels = torch.cat(bins, dim=2)
    multi_hot_labels = multi_hot_labels.type(torch.float32)

    return multi_hot_labels

def compute_labels(args, pos_contact_pts_mesh, pos_contact_dirs_mesh, pos_contact_approaches_mesh, pos_finger_diffs, pc_cam_pl, camera_pose_pl, global_config):
    """
    parameter:
        pos_contact_pts_mesh {torch.constant} -- positive contact points on the mesh scene (Mx3)
        pos_contact_dirs_mesh {torch.constant} -- respective contact base directions in the mesh scene (Mx3)
        pos_contact_approaches_mesh {torch.constant} -- respective contact approach directions in the mesh scene (Mx3)
        pos_finger_diffs {torch.constant} -- respective grasp widths in the mesh scene (Mx1)
        pc_cam_pl {torch.placeholder} -- bxNx3 rendered point clouds
        camera_pose_pl {torch.placeholder} -- bx4x4 camera poses(1x4x4)
        global_config {dict} -- global config
    return:
        [dir_labels_pc_cam, offset_labels_pc, grasp_success_labels_pc, approach_labels_pc_cam] --每个点的成功标签和接触点的成功姿态标签
    """

    label_config = global_config['DATA']['labels']
    # model_config = global_config['MODEL']

    nsample = label_config['k']
    radius = label_config['max_radius']
    filter_z = label_config['filter_z']
    z_val = label_config['z_val']

    xyz_cam = pc_cam_pl[:,:,:3]

    if args.use_gpu:
        pad_homog2 = torch.ones((pos_contact_dirs_mesh.shape[0], pos_contact_dirs_mesh.shape[1], 1)).cuda()
        z_val_tensor = torch.tensor([z_val]).cuda()
    else:
        pad_homog2 = torch.ones((pos_contact_dirs_mesh.shape[0], pos_contact_dirs_mesh.shape[1], 1))
        z_val_tensor = torch.tensor([z_val])
        
    contact_point_dirs_batch_cam = torch.matmul(pos_contact_dirs_mesh, torch.transpose(camera_pose_pl[:,:3,:3], 2, 1))[:,:,:3]

    pos_contact_approaches_batch_cam = torch.matmul(pos_contact_approaches_mesh, torch.transpose(camera_pose_pl[:,:3,:3], 2, 1))[:,:,:3]
    
    contact_point_batch_cam = torch.matmul(torch.cat([pos_contact_pts_mesh, pad_homog2], 2), torch.transpose(camera_pose_pl, 2, 1))[:,:,:3]

    if filter_z:
        dir_filter_passed = torch.repeat_interleave(torch.greater(contact_point_dirs_batch_cam[:,:,2:3], z_val_tensor), 3, dim=2)
        pos_contact_pts_mesh = torch.where(dir_filter_passed, pos_contact_pts_mesh, torch.ones_like(pos_contact_pts_mesh)*100000)

    squared_dists_all = torch.sum((torch.unsqueeze(contact_point_batch_cam,1)-torch.unsqueeze(xyz_cam,2))**2,dim=3)
    neg_squared_dists_k, close_contact_pt_idcs = torch.topk(-squared_dists_all, k=nsample, sorted=False)
    squared_dists_k = -neg_squared_dists_k

    grasp_success_labels_pc = torch.less(torch.mean(squared_dists_k, dim=2), radius*radius).type(torch.float32) # (batch_size, num_point)
    grouped_dirs_pc_cam = index_points(contact_point_dirs_batch_cam, close_contact_pt_idcs)
    grouped_approaches_pc_cam = index_points(pos_contact_approaches_batch_cam, close_contact_pt_idcs)
    grouped_offsets = index_points(torch.unsqueeze(pos_finger_diffs,2), close_contact_pt_idcs)

    dir_labels_pc_cam = F.normalize(torch.mean(grouped_dirs_pc_cam, dim=2),dim=2) # (batch_size, num_point, 3)
    approach_labels_pc_cam = F.normalize(torch.mean(grouped_approaches_pc_cam, dim=2),dim=2) # (batch_size, num_point, 3)
    offset_labels_pc = torch.mean(grouped_offsets, dim=2)


    if global_config['MODEL']['bin_offsets']:
        offset_labels_pc = torch.abs(offset_labels_pc)
        offset_labels_pc = multi_bin_labels(offset_labels_pc, global_config['DATA']['labels']['offset_bins'])
    
    return dir_labels_pc_cam, offset_labels_pc, grasp_success_labels_pc, approach_labels_pc_cam
