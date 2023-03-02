import torch
import numpy as np
from model.pcfnet import PCFNet
from utils.data import preprocess_pc_for_inference, farthest_points, distance_by_translation_point, regularize_pc_point_count, reject_median_outliers, depth2pc
from utils.grasp_utils import build_6d_grasp, get_bin_vals
import open3d as o3d
import torch.nn.functional as F

def convert_tf_gather_nd(params, indices):
    # print(list(indices.T))
    out = params[list(indices.T)]

    return out.T

def select_grasps(contact_pts, contact_conf, max_farthest_points = 150, num_grasps = 50, first_thres = 0.23, second_thres = 0.19, with_replacement=False):
    """
    Select subset of num_grasps by contact confidence thresholds and farthest contact point sampling. 

    1.) Samples max_farthest_points among grasp contacts with conf > first_thres
    2.) Fills up remaining grasp contacts to a maximum of num_grasps with highest confidence contacts with conf > second_thres
    
    Arguments:
        contact_pts {np.ndarray} -- num_point x 3 subset of input point cloud for which we have predictions 
        contact_conf {[type]} -- num_point x 1 confidence of the points being a stable grasp contact

    Keyword Arguments:
        max_farthest_points {int} -- Maximum amount from num_grasps sampled with farthest point sampling (default: {150})
        num_grasps {int} -- Maximum number of grasp proposals to select (default: {200})
        first_thres {float} -- first confidence threshold for farthest point sampling (default: {0.6})
        second_thres {float} -- second confidence threshold for filling up grasp proposals (default: {0.6})
        with_replacement {bool} -- Return fixed number of num_grasps with conf > first_thres and repeat if there are not enough (default: {False})

    Returns:
        [np.ndarray] -- Indices of selected contact_pts 
    """

    grasp_conf = contact_conf.squeeze()

    contact_pts = contact_pts.squeeze()

    result = grasp_conf > first_thres

    conf_idcs_greater_than = np.nonzero(result)[0]

    _, center_indexes = farthest_points(contact_pts[conf_idcs_greater_than,:3], np.minimum(max_farthest_points, len(conf_idcs_greater_than)), distance_by_translation_point, return_center_indexes = True)

    remaining_confidences = np.setdiff1d(np.arange(len(grasp_conf)), conf_idcs_greater_than[center_indexes])
    sorted_confidences = np.argsort(grasp_conf)[::-1]
    mask = np.in1d(sorted_confidences, remaining_confidences)
    sorted_remaining_confidence_idcs = sorted_confidences[mask]
    
    if with_replacement:
        selection_idcs = list(conf_idcs_greater_than[center_indexes])
        j=len(selection_idcs)
        while j < num_grasps and conf_idcs_greater_than.shape[0] > 0:
            selection_idcs.append(conf_idcs_greater_than[j%len(conf_idcs_greater_than)])
            j+=1
        selection_idcs = np.array(selection_idcs)

    else:
        remaining_idcs = sorted_remaining_confidence_idcs[:num_grasps-len(conf_idcs_greater_than[center_indexes])]
        remaining_conf_idcs_greater_than = np.nonzero(grasp_conf[remaining_idcs] > second_thres)[0]
        selection_idcs = np.union1d(conf_idcs_greater_than[center_indexes], remaining_idcs[remaining_conf_idcs_greater_than])

    return selection_idcs

def filter_segment(contact_pts, segment_pc, thres=0.00001):
    """
    Filter grasps to obtain contacts on specified point cloud segment
    
    :param contact_pts: Nx3 contact points of all grasps in the scene
    :param segment_pc: Mx3 segmented point cloud of the object of interest
    :param thres: maximum distance in m of filtered contact points from segmented point cloud
    :returns: Contact/Grasp indices that lie in the point cloud segment
    """
    filtered_grasp_idcs = np.array([],dtype=np.int32)
    
    if contact_pts.shape[0] > 0 and segment_pc.shape[0] > 0:
        try:
            dists = contact_pts[:,:3].reshape(-1,1,3) - segment_pc.reshape(1,-1,3)           
            min_dists = np.min(np.linalg.norm(dists,axis=2),axis=1)
            filtered_grasp_idcs = np.where(min_dists<thres)
        except:
            pass
        
    return filtered_grasp_idcs


def extract_point_clouds(depth, K, segmap=None, rgb=None, z_range=[0.2,1.2], segmap_id=0, skip_border_objects=False, margin_px=5):
    """
    Converts depth map + intrinsics to point cloud. 
    If segmap is given, also returns segmented point clouds. If rgb is given, also returns pc_colors.

    Arguments:
        depth {np.ndarray} -- HxW depth map in m
        K {np.ndarray} -- 3x3 camera Matrix

    Keyword Arguments:
        segmap {np.ndarray} -- HxW integer array that describes segeents (default: {None})
        rgb {np.ndarray} -- HxW rgb image (default: {None})
        z_range {list} -- Clip point cloud at minimum/maximum z distance (default: {[0.2,1.8]})
        segmap_id {int} -- Only return point cloud segment for the defined id (default: {0})
        skip_border_objects {bool} -- Skip segments that are at the border of the depth map to avoid artificial edges (default: {False})
        margin_px {int} -- Pixel margin of skip_border_objects (default: {5})

    Returns:
        [np.ndarray, dict[int:np.ndarray], np.ndarray] -- Full point cloud, point cloud segments, point cloud colors
    """

    if K is None:
        raise ValueError('K is required either as argument --K or from the input numpy file')
        
    # Convert to pc 
    pc_full, pc_colors = depth2pc(depth, K, rgb)

    num_points = 180  # irrelevant point removal strategy
    radius = 0.01
    # sor_pcd, ind = pcd.remove_radius_outlier(num_points, radius)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc_full)
    sor_pcd, ind = pcd.remove_radius_outlier(num_points, radius)
    sor_pcd.paint_uniform_color([0, 0, 1])

    pc_full = np.array(sor_pcd.points)
    pc_colors = pc_colors[ind]

    # Threshold distance
    if pc_colors is not None:
        pc_colors = pc_colors[(pc_full[:,2] < z_range[1]) & (pc_full[:,2] > z_range[0])] 
    pc_full = pc_full[(pc_full[:,2] < z_range[1]) & (pc_full[:,2] > z_range[0])]
    
    # Extract instance point clouds from segmap and depth map
    pc_segments = {}
    if segmap is not None:
        pc_segments = {}
        obj_instances = [segmap_id] if segmap_id else np.unique(segmap[segmap>0])
        for i in obj_instances:
            if skip_border_objects and not i==segmap_id:
                obj_i_y, obj_i_x = np.where(segmap==i)
                if np.any(obj_i_x < margin_px) or np.any(obj_i_x > segmap.shape[1]-margin_px) or np.any(obj_i_y < margin_px) or np.any(obj_i_y > segmap.shape[0]-margin_px):
                    print('object {} not entirely in image bounds, skipping'.format(i))
                    continue
            inst_mask = segmap==i
            pc_segment,_ = depth2pc(depth*inst_mask, K)
            pc_segments[i] = pc_segment[(pc_segment[:,2] < z_range[1]) & (pc_segment[:,2] > z_range[0])] #regularize_pc_point_count(pc_segment, grasp_estimator._contact_grasp_cfg['DATA']['num_point'])

    return pc_full, pc_segments, pc_colors

class GraspEstimatior():
    def __init__(self, cfg):

        self.global_config = cfg
        self.num_input_points = cfg['DATA']['raw_num_points'] if 'raw_num_points' in cfg['DATA'] else cfg['DATA']['num_point']

    def extract_3d_cam_boxes(self, full_pc, pc_segments, min_size=0.3, max_size=0.6):
        """
        Extract 3D bounding boxes around the pc_segments for inference to create 
        dense and zoomed-in predictions but still take context into account.
        
        :param full_pc: Nx3 scene point cloud
        :param pc_segments: Mx3 segmented point cloud of the object of interest
        :param min_size: minimum side length of the 3D bounding box
        :param max_size: maximum side length of the 3D bounding box
        :returns: (pc_regions, obj_centers) Point cloud box regions and their centers        
        """
        
        pc_regions = {}
        obj_centers = {}
        
        for i in pc_segments:
            pc_segments[i] = reject_median_outliers(pc_segments[i], m=0.4, z_only=False)
            
            if np.any(pc_segments[i]):
                max_bounds = np.max(pc_segments[i][:,:3], axis=0)
                min_bounds = np.min(pc_segments[i][:,:3], axis=0)

                obj_extent = max_bounds - min_bounds
                obj_center = min_bounds + obj_extent/2
                
                # cube size is between 0.3 and 0.6 depending on object extents
                size = np.minimum(np.maximum(np.max(obj_extent)*2, min_size), max_size)
                print('Extracted Region Cube Size: ', size)
                partial_pc = full_pc[np.all(full_pc > (obj_center - size/2), axis=1) & np.all(full_pc < (obj_center + size/2),axis=1)]
                if np.any(partial_pc):
                    partial_pc = regularize_pc_point_count(partial_pc, self.global_config['DATA']['raw_num_points'], use_farthest_point=self.global_config['DATA']['use_farthest_point'])
                    pc_regions[i] = partial_pc
                    obj_centers[i] = obj_center

        return pc_regions, obj_centers


    def predict_grasps(self, args, pc_full, obj_pc, constant_offset=False, convert_cam_coords=True, forward_passes=1):
        """
        Predict raw grasps on point cloud

        :param pc: Nx3 point cloud in camera coordinates
        :param convert_cam_coords: Convert from OpenCV to internal training camera coordinates (x left, y up, z front) and converts grasps back to openCV coordinates
        :param constant_offset: do not predict offset and place gripper at constant `extra_opening` distance from contact point
        :param forward_passes: Number of forward passes to run on each point cloud. default: 1
        :returns: (pred_grasps_cam, pred_scores, pred_points, gripper_openings) Predicted grasps/scores/contact-points/gripper-openings

        10.31 global_config : config for model
        data{dict}
        """

        pc, pc_mean = preprocess_pc_for_inference(pc_full.squeeze(), 1024, return_mean=True, convert_to_internal_coords=False)
        if obj_pc is not None:
            obj_pc += np.expand_dims(pc_mean, 0)
            
        if len(pc.shape) == 2:
            pc_batch = pc[np.newaxis, :, :]

        if forward_passes > 1:
            pc_batch = np.tile(pc_batch, (forward_passes, 1, 1))

        pc_batch = pc_batch.astype(np.float32)
        model = PCFNet(args, self.global_config)

        if args.use_gpu:
            model.load_state_dict(torch.load(args.ckpt_dir, map_location='gpu'))
        else:
            model.load_state_dict(torch.load(args.ckpt_dir, map_location='cpu'))
            
        model.eval()
        with torch.no_grad():

            pc = torch.from_numpy(pc_batch)
            end_points = model(pc)

            grasp_dir_head = end_points['grasp_dir_head']
            grasp_offset_head = end_points['grasp_offset_head']
            approach_dir_head = end_points['approach_dir_head']
            pred_scores = end_points['binary_score_pred']
            pred_points = end_points['pred_points']

            grasp_dir_head = grasp_dir_head.detach()#.numpy()
            grasp_offset_head = grasp_offset_head.detach()#.numpy()
            approach_dir_head = approach_dir_head.detach()#.numpy()
            pred_scores = pred_scores.detach()#.numpy()
            pred_points = pred_points.detach()#.numpy()

            tf_bin_vals = get_bin_vals(self.global_config)

            offset_bin_pred_vals = convert_tf_gather_nd(tf_bin_vals, torch.unsqueeze(torch.argmax(end_points['grasp_offset_head'], dim=2), dim=2)) if self.global_config['MODEL']['bin_offsets'] else end_points['grasp_offset_pred'][:,:,0]


            if offset_bin_pred_vals is None:
                offset_pred = end_points['grasp_offset_head']
            else:
                offset_pred = offset_bin_pred_vals


            if self.global_config['MODEL']['bin_offsets']:
                offset_bin_pred_vals = convert_tf_gather_nd(get_bin_vals(self.global_config), torch.unsqueeze(torch.argmax(grasp_offset_head, dim=2),dim=2))

            else:
                offset_bin_pred_vals = end_points['grasp_offset_pred'][:, :, 0]

            pred_grasps_cam = build_6d_grasp(approach_dir_head, grasp_dir_head, pred_points, offset_bin_pred_vals, use_cpu=True)

            if args.filter is not True:

                pred_grasps_cam = pred_grasps_cam.reshape(-1, *pred_grasps_cam.shape[-2:])
                pred_grasps_cam = pred_grasps_cam.numpy()
                pred_points = pred_points.reshape(-1, pred_points.shape[-1]).numpy()
                pred_scores = pred_scores.reshape(-1).numpy()
                offset_pred = offset_pred.reshape(-1)
                
                coarse_points = None

                if args.ori_inference is not True:
                    coarse_points = end_points['coarse'].squeeze().numpy()
                    coarse_points += np.expand_dims(pc_mean, 0)

                pred_grasps_cam[:,:3, 3] += pc_mean.reshape(-1,3)
                pred_points[:,:3] += pc_mean.reshape(-1,3)

                if constant_offset:
                    offset_pred = np.array([[self.global_config['DATA']['gripper_width']-self.global_config['TEST']['extra_opening']]*self.global_config['DATA']['num_point']])
                
                gripper_openings = np.minimum(offset_pred + self.global_config['TEST']['extra_opening'], self.global_config['DATA']['gripper_width']).numpy()

                selection_idcs = select_grasps(pred_points[:,:3], pred_scores, 
                                                    self.global_config['TEST']['max_farthest_points'], 
                                                    self.global_config['TEST']['num_samples'], 
                                                    self.global_config['TEST']['first_thres'], 
                                                    self.global_config['TEST']['second_thres'] if 'second_thres' in self.global_config['TEST'] else self.global_config['TEST']['first_thres'], 
                                                    with_replacement=self.global_config['TEST']['with_replacement'])

                if not np.any(selection_idcs):
                    selection_idcs=np.array([], dtype=np.int32)

                if 'center_to_tip' in self.global_config['TEST'] and self.global_config['TEST']['center_to_tip']:
                    pred_grasps_cam[:,:3, 3] -= pred_grasps_cam[:,:3,2]*(self.global_config['TEST']['center_to_tip']/2)
                
                # convert back to opencv coordinates

                if convert_cam_coords:
                    pred_grasps_cam[:,:2, :] *= -1
                    pred_points[:,:2] *= -1

                return pred_grasps_cam[selection_idcs], pred_scores[selection_idcs], pred_points[selection_idcs].squeeze(), gripper_openings[selection_idcs].squeeze(), coarse_points#, contact_pc, es_pc
            
            else:
                print('------------------------Filter----------------------')
                #change the R_rc and T_rc when the Hand-Eye Calibration is finished
                R_rc = torch.tensor([[-9.46586592e-03,  4.42036885e-01, -8.96946927e-01],
                                    [ 9.99740143e-01,  2.27856463e-02,  6.78624280e-04],
                                    [ 2.07374924e-02, -8.96707425e-01, -4.42137705e-01]])
                T_rc = torch.tensor([ 0.98905186, -0.03816485,  0.56813563])

                A_gr = torch.matmul(pred_grasps_cam[:, :, :3, 3].unsqueeze(2), R_rc.repeat(pred_grasps_cam.shape[0], pred_grasps_cam.shape[1], 1, 1)) + T_rc.repeat(pred_grasps_cam.shape[0],pred_grasps_cam.shape[1], 1, 1)

                A_gr = A_gr.squeeze(2) #BN3
                A_gr[:, :, 2] = 0
                A_gr = F.normalize(A_gr, dim=2)

                #[b | axb | a]
                a_gr = torch.matmul(R_rc.repeat(pred_grasps_cam.shape[0], pred_grasps_cam.shape[1], 1, 1), pred_grasps_cam[:, :, :3, :3])[:, :, :3, 2] #B N 3
                a_z = torch.abs(a_gr[:, :, 2])

                a_gr[:, : ,2] = 0
                a_gr = F.normalize(a_gr, dim=2)

                grasp_score_weight = torch.sigmoid(torch.sum(torch.mul(A_gr, a_gr),dim=2)) #B N

                pred_scores = torch.mul(grasp_score_weight, pred_scores.squeeze(2))

                pred_grasps_cam = pred_grasps_cam.reshape(-1, *pred_grasps_cam.shape[-2:])
                pred_grasps_cam = pred_grasps_cam.numpy()
                pred_points = pred_points.reshape(-1, pred_points.shape[-1]).numpy()
                pred_scores = pred_scores.reshape(-1).numpy()
                offset_pred = offset_pred.reshape(-1)
                
                coarse_points = None

                if args.ori_inference is not True:
                    coarse_points = end_points['coarse'].squeeze().numpy()
                    # print(coarse_points.shape)
                    coarse_points += np.expand_dims(pc_mean, 0)

                pred_grasps_cam[:,:3, 3] += pc_mean.reshape(-1,3)
                pred_points[:,:3] += pc_mean.reshape(-1,3)

                if constant_offset:
                    offset_pred = np.array([[self.global_config['DATA']['gripper_width']-self.global_config['TEST']['extra_opening']]*self.global_config['DATA']['num_point']])
                
                gripper_openings = np.minimum(offset_pred + self.global_config['TEST']['extra_opening'], self.global_config['DATA']['gripper_width']).numpy()

                selection_idcs = select_grasps(pred_points[:,:3], pred_scores, 
                                                    self.global_config['TEST']['max_farthest_points'], 
                                                    self.global_config['TEST']['num_samples'], 
                                                    self.global_config['TEST']['first_thres'], 
                                                    self.global_config['TEST']['second_thres'] if 'second_thres' in self.global_config['TEST'] else self.global_config['TEST']['first_thres'], 
                                                    with_replacement=self.global_config['TEST']['with_replacement'])

                if not np.any(selection_idcs):
                    selection_idcs=np.array([], dtype=np.int32)

                if 'center_to_tip' in self.global_config['TEST'] and self.global_config['TEST']['center_to_tip']:
                    pred_grasps_cam[:,:3, 3] -= pred_grasps_cam[:,:3,2]*(self.global_config['TEST']['center_to_tip']/2)
                
                # convert back to opencv coordinates

                if convert_cam_coords:
                    pred_grasps_cam[:,:2, :] *= -1
                    pred_points[:,:2] *= -1

                return pred_grasps_cam[selection_idcs], pred_scores[selection_idcs], pred_points[selection_idcs].squeeze(), gripper_openings[selection_idcs].squeeze(), coarse_points

    def predict_scene_grasps(self, pc_full, args, obj_pc=None, pc_segments={}, local_regions=False, filter_grasps=True, forward_passes=1):
        """
        Predict num_point grasps on a full point cloud or in local box regions around point cloud segments.

        Arguments:
            sess {tf.Session} -- Tensorflow Session
            pc_full {np.ndarray} -- Nx3 full scene point cloud  

        Keyword Arguments:
            pc_segments {dict[int, np.ndarray]} -- Dict of Mx3 segmented point clouds of objects of interest (default: {{}})
            local_regions {bool} -- crop 3D local regions around object segments for prediction (default: {False})
            filter_grasps {bool} -- filter grasp contacts such that they only lie within object segments (default: {False})
            forward_passes {int} -- Number of forward passes to run on each point cloud. (default: {1})

        Returns:
            [np.ndarray, np.ndarray, np.ndarray, np.ndarray] -- pred_grasps_cam, scores, contact_pts, gripper_openings
        """

        pred_grasps_cam, scores, contact_pts, gripper_openings = {}, {}, {}, {}

        pred_grasps_cam[-1], scores[-1], contact_pts[-1], gripper_openings[-1], coarse= self.predict_grasps(args, pc_full, obj_pc, convert_cam_coords=False, forward_passes=forward_passes)
        print('Generated {} grasps'.format(len(pred_grasps_cam[-1])))

        if filter_grasps:
            segment_keys = contact_pts.keys() if local_regions else pc_segments.keys()
            for k in segment_keys:
                j = k if local_regions else -1
                if np.any(pc_segments[k]) and np.any(contact_pts[j]):
                    segment_idcs = filter_segment(contact_pts[j], pc_segments[k], thres=self.global_config['TEST']['filter_thres'])

                    pred_grasps_cam[k] = pred_grasps_cam[j][segment_idcs]
                    scores[k] = scores[j][segment_idcs]
                    contact_pts[k] = contact_pts[j][segment_idcs]
                    try:
                        gripper_openings[k] = gripper_openings[j][segment_idcs]
                    except:
                        print('skipped gripper openings {}'.format(gripper_openings[j]))

                    if local_regions and np.any(pred_grasps_cam[k]):
                        print('Generated {} grasps for object {}'.format(len(pred_grasps_cam[k]), k))
                else:
                    print('skipping obj {} since  np.any(pc_segments[k]) {} and np.any(contact_pts[j]) is {}'.format(k, np.any(pc_segments[k]), np.any(contact_pts[j])))


        return pred_grasps_cam, scores, contact_pts, gripper_openings, coarse

