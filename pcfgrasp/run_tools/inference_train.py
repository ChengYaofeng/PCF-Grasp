import numpy as np
from torch.utils.data import DataLoader
from utils.load_data import load_available_input_data
from utils.grasp_estimator import GraspEstimatior, extract_point_clouds
from grasp_pub.grasp_pub.utils.visual_grasp_old import visualize_grasps


import cv2
import glob
import pyrealsense2 as rs
import multiprocessing as mp
from mask_rcnn.demo.demo import get_parser, setup_cfg
from mask_rcnn.demo.predictor import VisualizationDemo


def save_npy(pc, pc_num=1, color=None, num=None):
    """
    input:
        pc: {ndarray} N x 3
        color: {ndarray} N x 3
    """
    dict = {}

    dict['xyz'] = pc
    if color is not None:
        dict['xyz_color'] = color

    np.save('/home/cyf/contact_graspnet/test_data/{}_{}.npy'.format(pc_num, num), dict)
    print('saved file in /home/cyf/contact_graspnet/test_data/{}_{}.npy'.format(pc_num, num))


def gen_region_pc_box(pc_pred, pc_full):

    """
    Input: 
        point_clouds:{ndarray}  500 3
        pc_full: {ndarray}  640*480 3
    
    """
    # center = torch.mean(pc_pred, 1)

    max = np.argmax(pc_pred, axis=0)
    min = np.argmin(pc_pred, axis=0)

    max_x = pc_pred[max[0], 0]
    max_y = pc_pred[max[1], 1]
    max_z = pc_pred[max[2], 2]

    min_x = pc_pred[min[0], 0]
    min_y = pc_pred[min[1], 1]
    min_z = pc_pred[min[2], 2]

    box_x_max = max_x + 0.05
    box_y_max = max_y + 0.05
    box_z_max = max_z + 0.05

    box_x_min = min_x - 0.05
    box_y_min = min_y - 0.05
    box_z_min = min_z - 0.05

    # print(pc_full[:,0]>box_x_max)
    fil_x = (pc_full[:,0]>box_x_min) & (pc_full[:,0]<box_x_max)
    # print(fil_x)
    # fil_x = fil_x.reshape(1, pc_full.shape[1])
    pc_full = (pc_full[fil_x])
    # print(pc_full.shape)

    fil_y = (pc_full[:,1]>box_y_min) & (pc_full[:,1]<box_y_max)
    pc_full = (pc_full[fil_y])

    fil_z = (pc_full[:,2]>box_z_min) & (pc_full[:,2]<box_z_max)
    pc_full = (pc_full[fil_z])

    return pc_full

    

def train_inference(args, global_config, data_set=None, K=None, z_range=[0.2, 1.8] ,forward_passes=1):
    """
    Predict 6-DoF grasp distribution for given model and input data
    
    :param global_config: config.yaml from checkpoint directory
    :param checkpoint_dir: checkpoint directory
    :param input_paths: .png/.npz/.npy file paths that contain depth/pointcloud and optionally intrinsics/segmentation/rgb
    :param K: Camera Matrix with intrinsics to convert depth to point cloud
    :param local_regions: Crop 3D local regions around given segments. 
    :param skip_border_objects: When extracting local_regions, ignore segments at depth map boundary.
    :param filter_grasps: Filter and assign grasp contacts according to segmap.
    :param segmap_id: only return grasps from specified segmap_id.
    :param z_range: crop point cloud at a minimum/maximum z distance from camera to filter out outlier points. Default: [0.2, 1.8] m
    :param forward_passes: Number of forward passes to run on each point cloud. Default: 1
    """
    grasp_estimatior = GraspEstimatior(global_config)
    # print(f'------{args.input_path}-----------------')

    if args.input_path is not None:
        for p in glob.glob(args.input_path):
            print(f'Loading data from {p}')

            segmap, rgb, depth, cam, pc_full, pc_colors = load_available_input_data(p, K=K)

            if pc_full is None:
                print('Converting depth to point cloud(s)...')
                pc_full, pc_segments, pc_colors = extract_point_clouds(depth, cam, segmap=segmap, rgb=rgb, skip_border_objects=False, z_range=z_range)

            pred_grasps_cam, scores, contact_pts, _, coarse = grasp_estimatior.predict_scene_grasps(pc_full, args, pc_segments={}, forward_passes=forward_passes)

            visualize_grasps(pc_full, coarse, pred_grasps_cam, scores, plot_opencv_cam=True, pc_colors=pc_colors)


    else:
        eval_dataloader = DataLoader(data_set, batch_size=1, shuffle=False, num_workers=0)

        for i, data in enumerate(eval_dataloader):
            
            pc = data[0].squeeze().numpy()
            obj_pc = data[1].squeeze().numpy()
            # print(pc.shape)
            # print(obj_pc.shape)

            # print(i)
            pred_grasps_cam, scores, contact_pts, _, coarse= grasp_estimatior.predict_scene_grasps(pc, args, obj_pc, forward_passes=forward_passes)

            # save_npy(pc, pc_num=0, num=i)
            # save_npy(obj_pc, pc_num=1, num=i)
            visualize_grasps(pc, coarse, pred_grasps_cam, scores, obj_pc=obj_pc, plot_opencv_cam=True)
            # print(pc_segments)


        


