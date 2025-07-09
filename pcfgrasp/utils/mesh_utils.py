import os
import numpy as np
import pickle
from tqdm import tqdm
import trimesh
import trimesh.transformations as tra
import torch

class Object(object):
    """
    抓取目标定义
    """
    def __init__(self, filename):
        """
        filename: mesh to load
        """
        self.mesh = trimesh.load(filename)

        self.pc = None

        self.scale = 1.0
        self.filename = filename

        if isinstance(self.mesh, list):
            print("Warinig: Will do a concatenation")
            self.mesh = trimesh.util.concatenate(self.mesh)
        
        self.collision_manager = trimesh.collision.CollisionManager()
        #碰撞检测
        self.collision_manager.add_object('object', self.mesh)
    
    def rescale(self, scale=1.0):
        """
        模型的mesh比例
        """
        self.scale = scale
        self.mesh.apply_scale(self.scale)

    def resize(self, size=1.0):
        """
        模型的尺寸
        """
        self.scale = size / np.max(self.mesh.extents)
        self.mesh.apply_scale(self.scale)
    
    def in_collision_with(self, mesh, transform):
        """
        碰撞检测
        """
        return self.collision_manager.in_collision_single(mesh, transform = transform)
    
    def to_pointcloud(self):
        v_mesh = np.array(self.mesh.vertices)
        self.pc = trimesh.points.PointCloud(v_mesh)

        return self.pc


class PandaGripper(object):
    """
    franka 夹爪
    """

    def __init__(self, q=None, num_contact_points_per_finger=10, root_folder=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))):
        """
        franka夹爪模型
        参数：
            q {list of int} --夹爪打开状态初始默认
            num_contact_points_per_finger {int} --每个夹爪的接触点
            root_folder {str} --franka夹爪模型位置
        """
        self.joint_limits = [0.0, 0.04]
        #夹爪开合大小，0-4cm

        self.root_folder = root_folder
        self.default_pregrasp_configuration = 0.04

        if q is None:
            q = self.default_pregrasp_configuration
        
        self.q = q

        fn_base = os.path.join(root_folder, 'gripper_models/panda_gripper/hand.stl')
        fn_finger = os.path.join(root_folder, 'gripper_models/panda_gripper/finger.stl')
        #夹爪模型的身体和夹爪

        self.base = trimesh.load(fn_base)
        self.finger_l = trimesh.load(fn_finger)
        self.finger_r = self.finger_l.copy()
        #模型加载

        self.finger_l.apply_transform(tra.euler_matrix(0, 0, np.pi))
        self.finger_l.apply_translation([+q, 0, 0.0584])
        self.finger_r.apply_translation([-q, 0, 0.0584])
        #定义模型的位置

        self.fingers = trimesh.util.concatenate([self.finger_l, self.finger_r])
        self.hand = trimesh.util.concatenate([self.fingers, self.base])
        #夹爪模型组合

        self.contact_ray_origins = []
        self.contact_ray_directions = []
        
        #coords_path = os.path.join(root_folder, 'gripper_control_points/panda_gripper_coords.npy')
        with open(os.path.join(root_folder, 'gripper_control_points/panda_gripper_coords.pickle'), 'rb') as f:
            self.finger_coords = pickle.load(f, encoding='latin1')

        finger_direction = self.finger_coords['gripper_right_center_flat'] - self.finger_coords['gripper_left_center_flat']
        #夹爪的朝向，文中对应b

        self.contact_ray_origins.append(np.r_[self.finger_coords['gripper_left_center_flat'], 1])
        self.contact_ray_origins.append(np.r_[self.finger_coords['gripper_right_center_flat'], 1])

        self.contact_ray_directions.append(finger_direction / np.linalg.norm(finger_direction))
        self.contact_ray_directions.append(-finger_direction / np.linalg.norm(finger_direction))
        #夹爪指向

        self.contact_ray_origins = np.array(self.contact_ray_origins)
        self.contact_ray_directions = np.array(self.contact_ray_directions)


    def get_meshes(self):
        """
        获得夹爪mesh模型

        返回：
            夹爪外轮廓
        """
        return [self.finger_l, self.finger_r, self.base]
    
    def get_closing_rays_contacts(self, transform):
        """
        获得接触点接触点的坐标矩阵

        参数：
            transform {[numpy.array]} --4x4 分层矩阵
            contact_ray_origin {[numpy.array]} --4x1 分层向量
            contact_ray_direction {[numpy.array]} -- 4x1 分层向量
        
        返回：
            numpy.array 转换矩阵origin and direction
        """
        return transform[:3, :].dot(self.contact_ray_origins.T).T, transform[:3, :3].dot(self.contact_ray_directions.T).T
    
    def get_control_point_tensor(self, batch_size, use_tc=True, symmetric=False, convex_hull=True):
        """
        输出一个5点确定的夹爪位置  batch_size x 5 x 3

        参数：
            batch_size {int}

            use_tf {bool} 
        """
        control_points = np.load(os.path.join(self.root_folder, 'gripper_control_points/panda.npy'))[:, :3]
        #npy文件中是一个[20,4]的矩阵，第4列全1，后面的操作是把其变成3列
        # print(control_points)

        if symmetric:
            control_points = [[0, 0, 0], control_points[1, :], control_points[0, :], control_points[-1, :], control_points[-2, :]]
        else:
            control_points = [[0, 0, 0], control_points[0, :], control_points[1, :], control_points[-2, :], control_points[-1, :]]
        
        # print(control_points)
        # print(type(control_points))


        control_points = np.asarray(control_points, dtype=np.float32)

        if not convex_hull:
            # control_points[1:3, 2] = 0.0584
            control_points[1:3, 2] = 0.0584


        control_points = np.tile(np.expand_dims(control_points, 0), [batch_size, 1, 1])

        if use_tc:
            return torch.from_numpy(control_points)
        
        return control_points
    

def create_gripper(name, configuration=None, root_folder=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))):
    """
    创建抓取模型

    参数：
        configuration {list of float}  --状态
        root_folder {str} --模型文件的路径
    返回：
        [type] --抓取目标
    """
    if name.lower() == 'panda':
        return PandaGripper(q=configuration, root_folder=root_folder)
    else:
        raise Exception("Unknown gripper: {}".format(name))


def in_collision_with_gripper(object_mesh, gripper_transforms, gripper_name, silent=False):
    """
    判断是否接触

    参数：
        obeject_mesh {trimesh} --物体的模型
        gripper_transforms {list of numpy.array} --夹爪的分层矩阵
        gripper_name {str} --夹爪的名字

        silent {bool} --verbosity
    
    返回：
        [list of bool] -哪个夹爪姿态和目标模型接触了
     """
    manager = trimesh.collision.CollisionManger()
    #这是一个类，所以可以用后面的min_distance_single
    manager.add_object('object', object_mesh)
    gripper_meshes = [create_gripper(gripper_name).hand]
    #夹爪和目标物体的模型

    min_distance = []
    for tf in tqdm(gripper_transforms, disable=silent):
        """
        min_distance_single是由于manager是一个trimesh的类

        min_distance_single(mesh, transform=None, return_name=False, return_data=False)
        Get the minimum distance between a single object and any object in the manager.

        Parameters
        mesh (Trimesh object) – The geometry of the collision object

        transform ((4,4) float) – Homogeneous transform matrix for the object

        return_names (bool) – If true, return name of the closest object

        return_data (bool) – If true, a DistanceData object is returned as well

        Returns
        distance (float) – Min distance between mesh and any object in the manager

        name (str) – The name of the object in the manager that was closest

        data (DistanceData) – Extra data about the distance query
        """
        min_distance.append(np.min([manager.min_distance_single](gripper_mesh, transform=tf) for gripper_mesh in gripper_meshes))
    
    return [d == 0 for d in min_distance], min_distance


def grasp_contact_location(transforms, successfuls, collisions, object_mesh, gripper_name='panda', silent=False):
    """
    计算抓取接触点，补偿和方向

    参数：
        transforms {[type]} --抓取姿态
        collisions {[type]} --接触信息
        object_mesh {trimesh} --目标mesh
    
    返回：
        抓取的一个列表信息
    """
    res = []

    gripper = create_gripper(gripper_name)
    if trimesh.ray.has_embree:
        intersector = trimesh.ray.ray_pyembree.RayMeshIntersector(object_mesh, scale_to_box=True)
        #classtrimesh.ray.ray_pyembree.RayMeshIntersector(geometry, scale_to_box=True)
    else:
        intersector = trimesh.ray.ray_triangle.RayMeshIntersector(object_mesh)
    
    for p, colliding, outcome in tqdm(zip(transforms, collisions, successfuls), total=len(transforms), disable=silent):
        contact_dict = {}
        contact_dict['collisions'] = 0
        #这里表示未接触
        contact_dict['valid_locations'] = 0
        contact_dict['successful'] = outcome
        contact_dict['gradp_transform'] = p

        contact_dict['contact_points'] = []
        contact_dict['contact_directions'] = []
        contact_dict['contact_face_normals'] = []
        contact_dict['contact_offsets'] = []

        if colliding:
            contact_dict['collisions'] = 1
        else:
            ray_origins, ray_directions = gripper.get_closing_rays_contacts(p)

            locations, index_ray, index_tri = intersector.intersects_location(ray_origins, ray_directions, multiple_hits=False)
            """
            Parameters
                ray_origins ((m, 3) float) – Ray origin points

                ray_directions ((m, 3) float) – Ray direction vectors

            Returns
                locations ((n) sequence of (m,3) float) – Intersection points
                index_ray ((n,) int) – Array of ray indexes
                index_tri ((n,) int) – Array of triangle (face) indexes
            """
            #intersects_location(ray_origins, ray_directions, **kwargs)
            if len(locations) > 0:
                #由夹爪宽度决定
                valid_locations = np.linalg.norm(ray_origins[index_ray] - locations, axis=1) <= 2.0*gripper.q
                #np.linalg.norm(x, ord=None, axis=None, keepdims=False)

                if sum(valid_locations) > 1:
                    contact_dict['valid_locations'] = 1
                    contact_dict['contact_points'] = locations[valid_locations]
                    contact_dict['contact_face_normals'] = object_mesh.face_normals[index_tri[valid_locations]]
                    contact_dict['contact_directions'] = ray_directions[index_ray[valid_locations]]
                    contact_dict['contact_offsets'] = np.linalg.norm(ray_origins[index_ray[valid_locations]] - locations[valid_locations], axis=1)
 
                    res.append(contact_dict)
    return res
