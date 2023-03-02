import os
import numpy as np
import copy
import cv2
import trimesh.transformations as tra
from scipy.spatial import cKDTree
import open3d as o3d
from pointnet2_ops import pointnet2_utils
from run_utils.logger import print_log
from utils import provider
from utils.scene_renderer import SceneRenderer
from copy import deepcopy

####from contact grasp net  https://github.com/NVlabs/contact_graspnet  ########
# with irrelevant points removal strategy

def fps(pc, num):
    fps_idx = pointnet2_utils.furthest_point_sample(pc, num)
    sub_pc = pointnet2_utils.gather_operation(pc.transpose(1,2).contiguous(), fps_idx).transpose(1,2).contiguous()
    return sub_pc

def vis_camera(depth):
    geometrie_added = False
    vis = o3d.visualization.VisualizerWithKeyCallback()

    vis.create_window("Pointcloud",960,720)
    pointcloud = o3d.geometry.PointCloud()
    pinhole_camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(640, 480, 616.36529541, 616.20294189, 310.25881958, 236.59980774)

    while True:
        pointcloud.clear()
        depth_3d = o3d.geometry.Image(depth)

        pcd = o3d.geometry.PointCloud.create_from_depth_image(depth_3d, pinhole_camera_intrinsic)
        pcd.transform([[1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,1]])
        pointcloud += pcd

        if not geometrie_added:
            vis.add_geometry(pointcloud)
            geometrie_added = True
        
        vis.update_geometry(pointcloud)
        vis.poll_events()
        # vis.update_renderer()
        vis.run()
        vis.destroy_window()

        break


def vis_pc(pc):
    """
        pc: {numpy} N x 3
    """
    geometrie_added = False
    vis = o3d.visualization.VisualizerWithKeyCallback()

    vis.create_window("Pointcloud",960,720)
    pointcloud = o3d.geometry.PointCloud()

    while True:
        pointcloud.clear()
        pointcloud.points = o3d.utility.Vector3dVector(pc)

        if not geometrie_added:
            vis.add_geometry(pointcloud)
            geometrie_added = True
        
        vis.update_geometry(pointcloud)
        vis.poll_events()
        # vis.update_renderer()
        vis.run()
        vis.destroy_window()

        break
    
def farthest_point_sample(point, npoint):
    """
    Input:
        xyz: pointcloud data, [N, D]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint, D]
    """
    N, D = point.shape
    xyz = point[:,:3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    point = point[centroids.astype(np.int32)]
    return point


def preprocess_pc_for_inference(input_pc, num_point, pc_mean=None, return_mean=False, use_farthest_point=False, convert_to_internal_coords=False):
    """
    Various preprocessing of the point cloud (downsampling, centering, coordinate transforms)  

    Arguments:
        input_pc {np.ndarray} -- Nx3 input point cloud
        num_point {int} -- downsample to this amount of points

    Keyword Arguments:
        pc_mean {np.ndarray} -- use 3x1 pre-computed mean of point cloud  (default: {None})
        return_mean {bool} -- whether to return the point cloud mean (default: {False})
        use_farthest_point {bool} -- use farthest point for downsampling (slow and suspectible to outliers) (default: {False})
        convert_to_internal_coords {bool} -- Convert from opencv to internal coordinates (x left, y up, z front) (default: {False})

    Returns:
        [np.ndarray] -- num_pointx3 preprocessed point cloud
    """
    normalize_pc_count = input_pc.shape[0] != num_point
    if normalize_pc_count:
        pc = regularize_pc_point_count(input_pc, num_point, use_farthest_point=use_farthest_point).copy()
    else:
        pc = input_pc.copy()
    
    if convert_to_internal_coords:
        pc[:,:2] *= -1

    if pc_mean is None:
        pc_mean = np.mean(pc, 0)

    pc -= np.expand_dims(pc_mean, 0)
    if return_mean:
        return pc, pc_mean
    else:
        return pc


def inverse_transform(trans):
    """
    Computes the inverse of 4x4 transform.

    Arguments:
        trans {np.ndarray} -- 4x4 transform.

    Returns:
        [np.ndarray] -- inverse 4x4 transform
    """
    rot = trans[:3, :3]
    t = trans[:3, 3]
    rot = np.transpose(rot)
    t = -np.matmul(rot, t)
    output = np.zeros((4, 4), dtype=np.float32)
    output[3][3] = 1
    output[:3, :3] = rot
    output[:3, 3] = t

    return output

def distance_by_translation_point(p1, p2):
    """
      Gets two nx3 points and computes the distance between point p1 and p2.
    """
    return np.sqrt(np.sum(np.square(p1 - p2), axis=-1))


def farthest_points(data, nclusters, dist_func, return_center_indexes=False, return_distances=False, verbose=False):
    """
      Performs farthest point sampling on data points.
      Args:
        data: numpy array of the data points.
        nclusters: int, number of clusters.
        dist_dunc: distance function that is used to compare two data points.
        return_center_indexes: bool, If True, returns the indexes of the center of 
          clusters.
        return_distances: bool, If True, return distances of each point from centers.
      
      Returns clusters, [centers, distances]:
        clusters: numpy array containing the cluster index for each element in 
          data.
        centers: numpy array containing the integer index of each center.
        distances: numpy array of [npoints] that contains the closest distance of 
          each point to any of the cluster centers.
    """
    if nclusters >= data.shape[0]:
        if return_center_indexes:
            return np.arange(data.shape[0], dtype=np.int32), np.arange(data.shape[0], dtype=np.int32)

        return np.arange(data.shape[0], dtype=np.int32)

    clusters = np.ones((data.shape[0],), dtype=np.int32) * -1
    distances = np.ones((data.shape[0],), dtype=np.float32) * 1e7
    centers = []
    for iter in range(nclusters):
        index = np.argmax(distances)
        centers.append(index)
        shape = list(data.shape)
        for i in range(1, len(shape)):
            shape[i] = 1

        broadcasted_data = np.tile(np.expand_dims(data[index], 0), shape)
        new_distances = dist_func(broadcasted_data, data)
        distances = np.minimum(distances, new_distances)
        clusters[distances == new_distances] = iter
        if verbose:
            print('farthest points max distance : {}'.format(np.max(distances)))

    if return_center_indexes:
        if return_distances:
            return clusters, np.asarray(centers, dtype=np.int32), distances
        return clusters, np.asarray(centers, dtype=np.int32)

    return clusters

def reject_median_outliers(data, m=0.4, z_only=False):
    """
    Reject outliers with median absolute distance m

    Arguments:
        data {[np.ndarray]} -- Numpy array such as point cloud

    Keyword Arguments:
        m {[float]} -- Maximum absolute distance from median in m (default: {0.4})
        z_only {[bool]} -- filter only via z_component (default: {False})

    Returns:
        [np.ndarray] -- Filtered data without outliers
    """
    if z_only:
        d = np.abs(data[:,2:3] - np.median(data[:,2:3]))
    else:
        d = np.abs(data - np.median(data, axis=0, keepdims=True))

    return data[np.sum(d, axis=1) < m]

def regularize_pc_point_count(pc, npoints, use_farthest_point=False):
    """
      If point cloud pc has less points than npoints, it oversamples.
      Otherwise, it downsample the input pc to have npoint points.
      use_farthest_point: indicates 
      
      :param pc: Nx3 point cloud
      :param npoints: number of points the regularized point cloud should have
      :param use_farthest_point: use farthest point sampling to downsample the points, runs slower.
      :returns: npointsx3 regularized point cloud
    """
    
    if pc.shape[0] > npoints:
        if use_farthest_point:
            _, center_indexes = farthest_points(pc, npoints, distance_by_translation_point, return_center_indexes=True)
        else:
            center_indexes = np.random.choice(range(pc.shape[0]), size=npoints, replace=False)
        pc = pc[center_indexes, :]
    else:
        required = npoints - pc.shape[0]
        if required > 0:
            index = np.random.choice(range(pc.shape[0]), size=required)
            pc = np.concatenate((pc, pc[index, :]), axis=0)
    return pc

def depth2pc(depth, K, rgb=None):
    """
    Convert depth and intrinsics to point cloud and optionally point cloud color
    :param depth: hxw depth map in m
    :param K: 3x3 Camera Matrix with intrinsics
    :returns: (Nx3 point cloud, point cloud color)
    """

    mask = np.where(depth > 0)
    x,y = mask[1], mask[0]
    
    normalized_x = (x.astype(np.float32) - K[0,2])
    normalized_y = (y.astype(np.float32) - K[1,2])

    world_x = normalized_x * depth[y, x] / K[0,0]
    world_y = normalized_y * depth[y, x] / K[1,1]
    world_z = depth[y, x]

    if rgb is not None:
        rgb = rgb[y,x,:]
        
    pc = np.vstack((world_x, world_y, world_z)).T
    return (pc, rgb)


def estimate_normals_cam_from_pc(self, pc_cam, max_radius=0.05, k=12):
    """
    Estimates normals in camera coords from given point cloud.

    Arguments:
        pc_cam {np.ndarray} -- Nx3 point cloud in camera coordinates

    Keyword Arguments:
        max_radius {float} -- maximum radius for normal computation (default: {0.05})
        k {int} -- Number of neighbors for normal computation (default: {12})

    Returns:
        [np.ndarray] -- Nx3 point cloud normals
    """
    tree = cKDTree(pc_cam, leafsize=pc_cam.shape[0]+1)
    _, ndx = tree.query(pc_cam, k=k, distance_upper_bound=max_radius, n_jobs=-1) # num_points x k
    
    for c,idcs in enumerate(ndx):
        idcs[idcs==pc_cam.shape[0]] = c
        ndx[c,:] = idcs
    neighbors = np.array([pc_cam[ndx[:,n],:] for n in range(k)]).transpose((1,0,2))
    pc_normals = vectorized_normal_computation(pc_cam, neighbors)
    return pc_normals

def vectorized_normal_computation(pc, neighbors):
    """
    Vectorized normal computation with numpy

    Arguments:
        pc {np.ndarray} -- Nx3 point cloud
        neighbors {np.ndarray} -- Nxkx3 neigbours

    Returns:
        [np.ndarray] -- Nx3 normal directions
    """
    diffs = neighbors - np.expand_dims(pc, 1) # num_point x k x 3
    covs = np.matmul(np.transpose(diffs, (0, 2, 1)), diffs) # num_point x 3 x 3
    covs /= diffs.shape[1]**2
    # takes most time: 6-7ms
    eigen_values, eigen_vectors = np.linalg.eig(covs) # num_point x 3, num_point x 3 x 3
    orders = np.argsort(-eigen_values, axis=1) # num_point x 3
    orders_third = orders[:,2] # num_point
    directions = eigen_vectors[np.arange(pc.shape[0]),:,orders_third]  # num_point x 3
    dots = np.sum(directions * pc, axis=1) # num_point
    directions[dots >= 0] = -directions[dots >= 0]
    return directions


def center_pc_convert_cam(cam_poses, batch_data):
    """
    Converts from OpenGL to OpenCV coordinates, computes inverse of camera pose and centers point cloud
    
    :param cam_poses: (bx4x4) Camera poses in OpenGL format
    :param batch_data: (bxNx3) point clouds 
    :returns: (cam_poses, batch_data) converted
    """
    # OpenCV OpenGL conversion
    cam_poses[:3,1] = -cam_poses[:3,1]
    cam_poses[:3,2] = -cam_poses[:3,2]
    cam_poses = inverse_transform(cam_poses)
    cam_poses_new = copy.deepcopy(cam_poses)

    pc_mean = np.mean(batch_data, axis=0, keepdims=True)
    batch_data_new = batch_data[:,:3] - pc_mean[:,:3]
    cam_poses_new[:3, 3] -= pc_mean[0,:3]

    
    return batch_data_new, cam_poses_new

def center_pc_gt_convert_cam(cam_poses, batch_data, gt_data):
    """
    Converts from OpenGL to OpenCV coordinates, computes inverse of camera pose and centers point cloud
    
    :param cam_poses: (bx4x4) Camera poses in OpenGL format
    :param batch_data: (bxNx3) point clouds 
    :returns: (cam_poses, batch_data) converted
    """
    # OpenCV OpenGL conversion
    cam_poses[:3,1] = -cam_poses[:3,1]
    cam_poses[:3,2] = -cam_poses[:3,2]
    cam_poses = inverse_transform(cam_poses)
    pc_mean = np.mean(batch_data, axis=0, keepdims=True)
    gt_data[:,:3] -= pc_mean[:,:3]
    cam_poses[:3,3] -= pc_mean[0,:3]
    
    return gt_data, cam_poses

class PointCloudReader:
    """
    Class to load scenes, render point clouds and augment them during training

    Arguments:
        root_folder {str} -- acronym root folder
        batch_size {int} -- number of rendered point clouds per-batch

    Keyword Arguments:
        raw_num_points {int} -- Number of random/farthest point samples per scene (default: {20000})
        estimate_normals {bool} -- compute normals from rendered point cloud (default: {False})
        caching {bool} -- cache scenes in memory (default: {True})
        use_uniform_quaternions {bool} -- use uniform quaternions for camera sampling (default: {False})
        scene_obj_scales {list} -- object scales in scene (default: {None})
        scene_obj_paths {list} -- object paths in scene (default: {None})
        scene_obj_transforms {np.ndarray} -- object transforms in scene (default: {None})
        num_train_samples {int} -- training scenes (default: {None})
        num_test_samples {int} -- test scenes (default: {None})
        use_farthest_point {bool} -- use farthest point sampling to reduce point cloud dimension (default: {False})
        intrinsics {str} -- intrinsics to for rendering depth maps (default: {None})
        distance_range {tuple} -- distance range from camera to center of table (default: {(0.9,1.3)})
        elevation {tuple} -- elevation range (90 deg is top-down) (default: {(30,150)})
        pc_augm_config {dict} -- point cloud augmentation config (default: {None})
        depth_augm_config {dict} -- depth map augmentation config (default: {None})
    """
    def __init__(
        self,
        root_folder,
        batch_size=1,
        raw_num_points = 20000,
        estimate_normals = False,
        caching=True,
        use_uniform_quaternions=False,
        scene_obj_scales=None,
        scene_obj_paths=None,
        scene_obj_transforms=None,
        num_train_samples=None,
        num_test_samples=None,
        use_farthest_point = False,
        intrinsics = None,
        distance_range = (0.9,1.3),
        elevation = (30,150),
        pc_augm_config = None,
        depth_augm_config = None
    ):

        self._root_folder = root_folder
        self._batch_size = batch_size
        self._raw_num_points = raw_num_points
        self._caching = caching
        self._num_train_samples = num_train_samples
        self._num_test_samples = num_test_samples
        self._estimate_normals = estimate_normals
        self._use_farthest_point = use_farthest_point
        self._scene_obj_scales = scene_obj_scales
        self._scene_obj_paths = scene_obj_paths
        self._scene_obj_transforms = scene_obj_transforms
        self._distance_range = distance_range
        self._pc_augm_config = pc_augm_config
        self._depth_augm_config = depth_augm_config

        # print(pc_augm_config, depth_augm_config)
        self._current_pc = None
        self._cache = {}

        self._renderer = SceneRenderer(caching=True, intrinsics=intrinsics)

        if use_uniform_quaternions:
            quat_path = os.path.join(self._root_folder, 'uniform_quaternions/data2_4608.qua')
            quaternions = [l[:-1].split('\t') for l in open(quat_path, 'r').readlines()]

            quaternions = [[float(t[0]),
                            float(t[1]),
                            float(t[2]),
                            float(t[3])] for t in quaternions]
            quaternions = np.asarray(quaternions)
            quaternions = np.roll(quaternions, 1, axis=1)
            self._all_poses = [tra.quaternion_matrix(q) for q in quaternions]
        else:
            self._cam_orientations = []
            self._elevation = np.array(elevation)/180. 
            for az in np.linspace(0, np.pi * 2, 30):
                # print(az)
                for el in np.linspace(self._elevation[0], self._elevation[1], 30):
                    # print(el)
                    self._cam_orientations.append(tra.euler_matrix(0, -el, az))
            # print(len(self._cam_orientations)) 900
            self._coordinate_transform = tra.euler_matrix(np.pi/2, 0, 0).dot(tra.euler_matrix(0, np.pi/2, 0))

    def get_cam_pose(self, cam_orientation):
        """
        Samples camera pose on shell around table center 

        Arguments:
            cam_orientation {np.ndarray} -- 3x3 camera orientation matrix

        Returns:
            [np.ndarray] -- 4x4 homogeneous camera pose
        """
        distance = self._distance_range[0] + np.random.rand()*(self._distance_range[1]-self._distance_range[0])

        extrinsics = np.eye(4)
        extrinsics[0, 3] += distance

        extrinsics = cam_orientation.dot(extrinsics)

        cam_pose = extrinsics.dot(self._coordinate_transform)
        # table height
        # print(self._renderer._table_dims[2])
        cam_pose[2,3] += self._renderer._table_dims[2]
        # cam_pose[:3,:2]= -cam_pose[:3,:2]
        return cam_pose

    def _augment_pc(self, pc):
        """
        Augments point cloud with jitter and dropout according to config

        Arguments:
            pc {np.ndarray} -- Nx3 point cloud

        Returns:
            np.ndarray -- augmented point cloud
        """
        
        # not used because no artificial occlusion
        if 'occlusion_nclusters' in self._pc_augm_config and self._pc_augm_config['occlusion_nclusters'] > 0:
            pc = self.apply_dropout(pc,
                                    self._pc_augm_config['occlusion_nclusters'], 
                                    self._pc_augm_config['occlusion_dropout_rate'])

        if 'sigma' in self._pc_augm_config and self._pc_augm_config['sigma'] > 0:
            pc = provider.jitter_point_cloud(pc[np.newaxis, :, :], 
                                            sigma=self._pc_augm_config['sigma'], 
                                            clip=self._pc_augm_config['clip'])[0]
        
        
        return pc[:,:3]

    def _augment_depth(self, depth):
        """
        Augments depth map with z-noise and smoothing according to config

        Arguments:
            depth {np.ndarray} -- depth map

        Returns:
            np.ndarray -- augmented depth map
        """

        if 'sigma' in self._depth_augm_config and self._depth_augm_config['sigma'] > 0:
            clip = self._depth_augm_config['clip']
            sigma = self._depth_augm_config['sigma']
            noise = np.clip(sigma*np.random.randn(*depth.shape), -clip, clip)
            depth += noise
        if 'gaussian_kernel' in self._depth_augm_config and self._depth_augm_config['gaussian_kernel'] > 0:
            kernel = self._depth_augm_config['gaussian_kernel']
            depth_copy = depth.copy()
            depth = cv2.GaussianBlur(depth,(kernel,kernel),0)
            depth[depth_copy==0] = depth_copy[depth_copy==0]
                
        return depth

    def apply_dropout(self, pc, occlusion_nclusters, occlusion_dropout_rate):
        """
        Remove occlusion_nclusters farthest points from point cloud with occlusion_dropout_rate probability

        Arguments:
            pc {np.ndarray} -- Nx3 point cloud
            occlusion_nclusters {int} -- noof cluster to remove
            occlusion_dropout_rate {float} -- prob of removal

        Returns:
            [np.ndarray] -- N > Mx3 point cloud
        """
        if occlusion_nclusters == 0 or occlusion_dropout_rate == 0.:
            return pc

        labels = farthest_points(pc, occlusion_nclusters, distance_by_translation_point)

        removed_labels = np.unique(labels)
        removed_labels = removed_labels[np.random.rand(removed_labels.shape[0]) < occlusion_dropout_rate]
        if removed_labels.shape[0] == 0:
            return pc
        mask = np.ones(labels.shape, labels.dtype)
        for l in removed_labels:
            mask = np.logical_and(mask, labels != l)
        return pc[mask]
    
    def get_scene_batch(self, scene_idx=None, return_segmap=False, save=False):
        """
        Render a batch of scene point clouds

        Keyword Arguments:
            scene_idx {int} -- index of the scene (default: {None})
            return_segmap {bool} -- whether to render a segmap of objects (default: {False})
            save {bool} -- Save training/validation data to npz file for later inference (default: {False})

        Returns:
            [batch_data, cam_poses, scene_idx] -- batch of rendered point clouds, camera poses and the scene_idx
        """
        dims = 6 if self._estimate_normals else 3
        batch_data = np.empty((self._raw_num_points, dims), dtype=np.float32)
        cam_poses = np.empty((4, 4), dtype=np.float32)

        obj_data = np.empty((2048, dims), dtype=np.float32)
        # obj_data = np.empty((16384, dims), dtype=np.float32)



        if scene_idx is None:
            scene_idx = np.random.randint(0,self._num_train_samples)

        obj_paths = [os.path.join(self._root_folder, p) for p in self._scene_obj_paths[scene_idx]]
        mesh_scales = self._scene_obj_scales[scene_idx]
        obj_trafos = self._scene_obj_transforms[scene_idx]
        
        obj_pc = self.change_scene(obj_paths, mesh_scales, obj_trafos, visualize=False)


        batch_segmap, batch_obj_pcs = [], []

        pc_cam, pc_normals, camera_pose, depth, obj_pc_n = self.render_random_scene(obj_pc = obj_pc, estimate_normals = self._estimate_normals)

        idx = np.random.permutation(obj_pc_n.shape[0])
        if idx.shape[0] < 2048:
            idx = np.concatenate([idx, np.random.randint(obj_pc_n.shape[0], size=2048-obj_pc_n.shape[0])])
        obj_pc_se = obj_pc_n[idx[:2048]]
        # obj_pc_idx = farthest_point_sample(obj_pc_n, 2048)
        # obj_pc_se = index_points(obj_pc_n, obj_pc_idx)

        if return_segmap:
            segmap, _, obj_pcs = self._renderer.render_labels(depth, obj_paths, mesh_scales, render_pc=True)


            batch_obj_pcs.append(obj_pcs)
            batch_segmap.append(segmap)

        batch_data[:,0:3] = pc_cam[:,:3]
        obj_data[:,0:3] = obj_pc_se[:, :3]

        if self._estimate_normals:
            batch_data[:,3:6] = pc_normals[:,:3]
        cam_poses[:,:] = camera_pose
            
        if save:
            K = np.array([[616.36529541,0,310.25881958 ],[0,616.20294189,236.59980774],[0,0,1]])
            data = {'depth':depth, 'K':K, 'camera_pose':camera_pose, 'scene_idx':scene_idx}
            if return_segmap:
                data.update(segmap=segmap)
            np.savez('results/{}_acronym.npz'.format(scene_idx), data)

        if return_segmap:
            return batch_data, cam_poses, scene_idx, batch_segmap, batch_obj_pcs
        else:
            return batch_data, cam_poses, scene_idx, obj_data

    def render_random_scene(self, obj_pc, estimate_normals=False, camera_pose=None):
        """
        Renders scene depth map, transforms to regularized pointcloud and applies augmentations

        Par: 
            obj_pc: N 3+1 {np.ndarray} points in world

        Keyword Arguments:
            estimate_normals {bool} -- calculate and return normals (default: {False})
            camera_pose {[type]} -- camera pose to render the scene from. (default: {None})

        Returns:
            [pc, pc_normals, camera_pose, depth] -- [point cloud, point cloud normals, camera pose, depth]
        """
        if camera_pose is None:
            viewing_index = np.random.randint(0, high=len(self._cam_orientations))
            camera_orientation = self._cam_orientations[viewing_index]
            
            camera_pose = self.get_cam_pose(camera_orientation)

        in_camera_pose = copy.deepcopy(camera_pose)

        # 0.005 s
        _, depth, _, camera_pose = self._renderer.render(in_camera_pose, render_pc=False)

        depth = self._augment_depth(depth)
        obj = deepcopy(obj_pc)
        try:
            pc = self._renderer._to_pointcloud(depth)  #N 3+1

            obj = np.dot(np.linalg.inv(camera_pose), obj.T).T
            # print(obj_pc.shape)
            obj = -obj
            obj[:, 0] = -obj[:, 0]

            pc = np.where(np.abs(pc)>0.005, pc, 0)

            pc_n = pc[:, :3]
            pc = pc[np.all(pc_n !=0, 1), :]

            if pc.shape == (0,4):
                print_log('Pc_None & refine')
                pc = obj

        finally:
            

            pc = regularize_pc_point_count(pc, self._raw_num_points, use_farthest_point=self._use_farthest_point)
            
            pc_normals = estimate_normals_cam_from_pc(pc[:,:3], raw_num_points=self._raw_num_points) if estimate_normals else []

        return pc, pc_normals, camera_pose, depth, obj

    def change_object(self, cad_path, cad_scale):
        """
        Change object in pyrender scene

        Arguments:
            cad_path {str} -- path to CAD model
            cad_scale {float} -- scale of CAD model
        """

        self._renderer.change_object(cad_path, cad_scale)

    def change_scene(self, obj_paths, obj_scales, obj_transforms, visualize=False):
        """
        Change pyrender scene

        Arguments:
            obj_paths {list[str]} -- path to CAD models in scene
            obj_scales {list[float]} -- scales of CAD models
            obj_transforms {list[np.ndarray]} -- poses of CAD models

        Keyword Arguments:
            visualize {bool} -- whether to update the visualizer as well (default: {False})
        """
        pc = self._renderer.change_scene(obj_paths, obj_scales, obj_transforms)
        if visualize:
            self._visualizer.change_scene(obj_paths, obj_scales, obj_transforms)
        return pc
 

    def __del__(self):
        print('********** terminating renderer **************')
    

