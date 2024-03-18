import os
import numpy as np
import pickle
from tqdm import tqdm
import trimesh
import trimesh.transformations as tra
import torch

class Object(object):
    """
    grasp objects
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
        self.collision_manager.add_object('object', self.mesh)
    
    def rescale(self, scale=1.0):
        """
        mesh scale
        """
        self.scale = scale
        self.mesh.apply_scale(self.scale)

    def resize(self, size=1.0):
        """
        mesh size
        """
        self.scale = size / np.max(self.mesh.extents)
        self.mesh.apply_scale(self.scale)
    
    def in_collision_with(self, mesh, transform):
        """
        collision detection
        """
        return self.collision_manager.in_collision_single(mesh, transform = transform)
    
    def to_pointcloud(self):
        v_mesh = np.array(self.mesh.vertices)
        self.pc = trimesh.points.PointCloud(v_mesh)

        return self.pc


class PandaGripper(object):
    """
    franka gripper
    """

    def __init__(self, q=None, num_contact_points_per_finger=10, root_folder=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))):
        """
        franka model
        parameter:
            q {list of int}
            num_contact_points_per_finger {int}
            root_folder {str}
        """
        self.joint_limits = [0.0, 0.04]

        self.root_folder = root_folder
        self.default_pregrasp_configuration = 0.04

        if q is None:
            q = self.default_pregrasp_configuration
        
        self.q = q

        fn_base = os.path.join(root_folder, 'gripper_models/panda_gripper/hand.stl')
        fn_finger = os.path.join(root_folder, 'gripper_models/panda_gripper/finger.stl')

        self.base = trimesh.load(fn_base)
        self.finger_l = trimesh.load(fn_finger)
        self.finger_r = self.finger_l.copy()

        self.finger_l.apply_transform(tra.euler_matrix(0, 0, np.pi))
        self.finger_l.apply_translation([+q, 0, 0.0584])
        self.finger_r.apply_translation([-q, 0, 0.0584])

        self.fingers = trimesh.util.concatenate([self.finger_l, self.finger_r])
        self.hand = trimesh.util.concatenate([self.fingers, self.base])

        self.contact_ray_origins = []
        self.contact_ray_directions = []
        
        #coords_path = os.path.join(root_folder, 'gripper_control_points/panda_gripper_coords.npy')
        with open(os.path.join(root_folder, 'gripper_control_points/panda_gripper_coords.pickle'), 'rb') as f:
            self.finger_coords = pickle.load(f, encoding='latin1')

        finger_direction = self.finger_coords['gripper_right_center_flat'] - self.finger_coords['gripper_left_center_flat']

        self.contact_ray_origins.append(np.r_[self.finger_coords['gripper_left_center_flat'], 1])
        self.contact_ray_origins.append(np.r_[self.finger_coords['gripper_right_center_flat'], 1])

        self.contact_ray_directions.append(finger_direction / np.linalg.norm(finger_direction))
        self.contact_ray_directions.append(-finger_direction / np.linalg.norm(finger_direction))

        self.contact_ray_origins = np.array(self.contact_ray_origins)
        self.contact_ray_directions = np.array(self.contact_ray_directions)


    def get_meshes(self):
        """
        get gripper mesh

        """
        return [self.finger_l, self.finger_r, self.base]
    
    def get_closing_rays_contacts(self, transform):
        """
        get contact points

        parameter:
            transform {[numpy.array]} --4x4
            contact_ray_origin {[numpy.array]} --4x1
            contact_ray_direction {[numpy.array]} -- 4x1
        
        return:
            numpy.array
        """
        return transform[:3, :].dot(self.contact_ray_origins.T).T, transform[:3, :3].dot(self.contact_ray_directions.T).T
    
    def get_control_point_tensor(self, batch_size, use_tc=True, symmetric=False, convex_hull=True):
        """
        gripper position  batch_size x 5 x 3

        parameter:
            batch_size {int}

            use_tf {bool} 
        """
        control_points = np.load(os.path.join(self.root_folder, 'gripper_control_points/panda.npy'))[:, :3]

        if symmetric:
            control_points = [[0, 0, 0], control_points[1, :], control_points[0, :], control_points[-1, :], control_points[-2, :]]
        else:
            control_points = [[0, 0, 0], control_points[0, :], control_points[1, :], control_points[-2, :], control_points[-1, :]]
        

        control_points = np.asarray(control_points, dtype=np.float32)

        if not convex_hull:
            control_points[1:3, 2] = 0.0584


        control_points = np.tile(np.expand_dims(control_points, 0), [batch_size, 1, 1])

        if use_tc:
            return torch.from_numpy(control_points)
        
        return control_points
    

def create_gripper(name, configuration=None, root_folder=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))):
    """
    parameter:
        configuration {list of float}
        root_folder {str} 
    """
    if name.lower() == 'panda':
        return PandaGripper(q=configuration, root_folder=root_folder)
    else:
        raise Exception("Unknown gripper: {}".format(name))


def in_collision_with_gripper(object_mesh, gripper_transforms, gripper_name, silent=False):
    """
    parameter:
        obeject_mesh {trimesh}
        gripper_transforms {list of numpy.array}
        gripper_name {str}

        silent {bool} --verbosity
    
    return:
        [list of bool]
     """
    manager = trimesh.collision.CollisionManger()
    manager.add_object('object', object_mesh)
    gripper_meshes = [create_gripper(gripper_name).hand]

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
    parameter:
        transforms {[type]}
        collisions {[type]}
        object_mesh {trimesh}
    
    return:
        grasp message {list}
    """
    res = []

    gripper = create_gripper(gripper_name)
    if trimesh.ray.has_embree:
        intersector = trimesh.ray.ray_pyembree.RayMeshIntersector(object_mesh, scale_to_box=True)
    else:
        intersector = trimesh.ray.ray_triangle.RayMeshIntersector(object_mesh)
    
    for p, colliding, outcome in tqdm(zip(transforms, collisions, successfuls), total=len(transforms), disable=silent):
        contact_dict = {}
        contact_dict['collisions'] = 0
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
