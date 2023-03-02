import numpy as np
from PIL import Image
import glob
import os
import cv2
import torch
import torch.nn.functional as F

def load_scene_contacts(dataset_folder, test_split_only=False, num_test=None, scene_contacts_path='scene_contacts'):
    """
    Load contact grasp annotations from acronym scenes 

    Arguments:
        dataset_folder {str} -- folder with acronym data and scene contacts

    Keyword Arguments:
        test_split_only {bool} -- whether to only return test split scenes (default: {False})
        num_test {int} -- how many test scenes to use (default: {None})
        scene_contacts_path {str} -- name of folder with scene contact grasp annotations (default: {'scene_contacts_new'})

    Returns:
        list(dicts) -- list of scene annotations dicts with object paths and transforms and grasp contacts and transforms.
    """
    
    scene_contact_paths = sorted(glob.glob(os.path.join(dataset_folder, scene_contacts_path, "*")))

    if test_split_only:
        scene_contact_paths = scene_contact_paths[-num_test:]
    contact_infos = []
    for contact_path in scene_contact_paths:
        try:
            npz = np.load(contact_path, allow_pickle=False)
            contact_info = {'scene_contact_points':npz['scene_contact_points'],
                            'obj_paths':npz['obj_paths'],
                            'obj_transforms':npz['obj_transforms'],
                            'obj_scales':npz['obj_scales'],
                            'grasp_transforms':npz['grasp_transforms']}
            contact_infos.append(contact_info)
            
        except:
            print('corrupt, ignoring..')
    return contact_infos


def load_graspnet_data(rgb_image_path):
    
    """
    Loads data from the GraspNet-1Billion dataset
    # https://graspnet.net/

    :param rgb_image_path: .png file path to depth image in graspnet dataset
    :returns: (depth, rgb, segmap, K)
    """
    
    depth = np.array(Image.open(rgb_image_path))/1000. # m to mm
    segmap = np.array(Image.open(rgb_image_path.replace('depth', 'label')))
    rgb = np.array(Image.open(rgb_image_path.replace('depth', 'rgb')))

    # graspnet images are upside down, rotate for inference
    # careful: rotate grasp poses back for evaluation
    depth = np.rot90(depth,2)
    segmap = np.rot90(segmap,2)
    rgb = np.rot90(rgb,2)
    
    if 'kinect' in rgb_image_path:
        # Kinect azure:
        K=np.array([[631.54864502 ,  0.    ,     638.43517329],
                    [  0.    ,     631.20751953, 366.49904066],
                    [  0.    ,       0.    ,       1.        ]])
    else:
        # Realsense:
        K=np.array([[616.36529541 ,  0.    ,     310.25881958],
                    [  0.    ,     616.20294189, 236.59980774],
                    [  0.    ,       0.    ,       1.        ]])

    return depth, rgb, segmap, K


def load_available_input_data(p, K=None):
    """
    Load available data from input file path. 
    
    Numpy files .npz/.npy should have keys
    'depth' + 'K' + (optionally) 'segmap' + (optionally) 'rgb'
    or for point clouds:
    'xyz' + (optionally) 'xyz_color'
    
    png files with only depth data (in mm) can be also loaded.
    If the image path is from the GraspNet dataset, corresponding rgb, segmap and intrinic are also loaded.
      
    :param p: .png/.npz/.npy file path that contain depth/pointcloud and optionally intrinsics/segmentation/rgb
    :param K: 3x3 Camera Matrix with intrinsics
    :returns: All available data among segmap, rgb, depth, cam_K, pc_full, pc_colors
    """
    
    segmap, rgb, depth, pc_full, pc_colors, cam_K = None, None, None, None, None, None

    if K is not None:
        if isinstance(K,str):
            cam_K = eval(K)
        #cam_K = np.array(K).reshape(3,3)

    if '.np' in p:
        data = np.load(p, allow_pickle=True)
        if '.npz' in p:
            keys = data.files
        else:
            keys = []
            if len(data.shape) == 0:
                data = data.item()
                keys = data.keys()
            elif data.shape[-1] == 3:
                pc_full = data
            else:
                depth = data

        if 'depth' in keys:
            depth = data['depth']
            if K is None and 'K' in keys:
                cam_K = data['K'].reshape(3,3)
            if 'segmap' in keys:    
                segmap = data['segmap']
            if 'seg' in keys:    
                segmap = data['seg']
            if 'rgb' in keys:    
                rgb = data['rgb']
                rgb = np.array(cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB))
        elif 'xyz' in keys:
            pc_full = np.array(data['xyz']).reshape(-1,3)
            if 'xyz_color' in keys:
                pc_colors = data['xyz_color']
    elif '.png' in p:
        if os.path.exists(p.replace('depth', 'label')):
            # graspnet data
            depth, rgb, segmap, K = load_graspnet_data(p)
        elif os.path.exists(p.replace('depths', 'images').replace('npy', 'png')):
            rgb = np.array(Image.open(p.replace('depths', 'images').replace('npy', 'png')))
        else:
            depth = np.array(Image.open(p))
    else:
        raise ValueError('{} is neither png nor npz/npy file'.format(p))
    
    return segmap, rgb, depth, cam_K, pc_full, pc_colors


def load_contact_grasps(contact_list, data_config):
    """
    Loads fixed amount of contact grasp data per scene into tf CPU/GPU memory

    Arguments:
        contact_infos {list(dicts)} -- Per scene mesh: grasp contact information  
        data_config {dict} -- data config

    Returns:
        [tf_pos_contact_points, tf_pos_contact_dirs, tf_pos_contact_offsets, 
        tf_pos_contact_approaches, tf_pos_finger_diffs, tf_scene_idcs, 
        all_obj_paths, all_obj_transforms] -- tf.constants with per scene grasp data, object paths/transforms in scene
    """

    num_pos_contacts = data_config['labels']['num_pos_contacts'] 

    pos_contact_points = []
    pos_contact_dirs = []
    pos_finger_diffs = []
    pos_approach_dirs = []

    for i,c in enumerate(contact_list):
        contact_directions_01 = c['scene_contact_points'][:,0,:] - c['scene_contact_points'][:,1,:]
        all_contact_points = c['scene_contact_points'].reshape(-1,3)
        all_finger_diffs = np.maximum(np.linalg.norm(contact_directions_01,axis=1), np.finfo(np.float32).eps)
        all_contact_directions = np.empty((contact_directions_01.shape[0]*2, contact_directions_01.shape[1],))
        all_contact_directions[0::2] = -contact_directions_01 / all_finger_diffs[:,np.newaxis]
        all_contact_directions[1::2] = contact_directions_01 / all_finger_diffs[:,np.newaxis]
        all_contact_suc = np.ones_like(all_contact_points[:,0])
        all_grasp_transform = c['grasp_transforms'].reshape(-1,4,4)
        all_approach_directions = all_grasp_transform[:,:3,2]

        pos_idcs = np.where(all_contact_suc>0)[0]
        if len(pos_idcs) == 0:
            continue
        # print('total positive contact points ', len(pos_idcs))
        
        all_pos_contact_points = all_contact_points[pos_idcs]
        all_pos_finger_diffs = all_finger_diffs[pos_idcs//2]
        all_pos_contact_dirs = all_contact_directions[pos_idcs]
        all_pos_approach_dirs = all_approach_directions[pos_idcs//2]
        
        # Use all positive contacts then mesh_utils with replacement
        if num_pos_contacts > len(all_pos_contact_points)/2:
            pos_sampled_contact_idcs = np.arange(len(all_pos_contact_points))
            pos_sampled_contact_idcs_replacement = np.random.choice(np.arange(len(all_pos_contact_points)), num_pos_contacts*2 - len(all_pos_contact_points) , replace=True) 
            pos_sampled_contact_idcs= np.hstack((pos_sampled_contact_idcs, pos_sampled_contact_idcs_replacement))
        else:
            pos_sampled_contact_idcs = np.random.choice(np.arange(len(all_pos_contact_points)), num_pos_contacts*2, replace=False)

        pos_contact_points.append(all_pos_contact_points[pos_sampled_contact_idcs,:])
        pos_contact_dirs.append(all_pos_contact_dirs[pos_sampled_contact_idcs,:])
        pos_finger_diffs.append(all_pos_finger_diffs[pos_sampled_contact_idcs])
        pos_approach_dirs.append(all_pos_approach_dirs[pos_sampled_contact_idcs])

    # with torch.device(device):
    tf_scene_idcs = torch.tensor(np.arange(0,len(pos_contact_points)), dtype=torch.int32)
    tf_pos_contact_points = torch.tensor(np.array(pos_contact_points), dtype=torch.float32)
    tf_pos_contact_dirs =  F.normalize(torch.tensor(np.array(pos_contact_dirs), dtype=torch.float32), dim=2)
    tf_pos_finger_diffs = torch.tensor(np.array(pos_finger_diffs), dtype=torch.float32)
    tf_pos_contact_approaches =  F.normalize(torch.tensor(np.array(pos_approach_dirs), dtype=torch.float32), dim=2)

    return tf_pos_contact_points, tf_pos_contact_dirs, tf_pos_contact_approaches, tf_pos_finger_diffs, tf_scene_idcs