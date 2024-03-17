import numpy as np
import copy
import os
from copy import deepcopy
# os.environ["PYOPENGL_PLATFORM"] = "egl"
import pyrender
from torch import float32
import trimesh
import trimesh.transformations as tra
import open3d as o3d

from utils.mesh_utils import Object

class SceneRenderer:
    def __init__(self, intrinsics=None, fov=np.pi / 6, caching=True, viewing_mode=False):
        """
        parameter:
            intrinsics {str} -- camera name from 'kinect_azure', 'realsense' (default: {None})
            fov {float} -- field of view, ignored if inrinsics is not None (default: {np.pi/6})
            caching {bool} -- whether to cache object meshes (default: {True})
            viewing_mode {bool} -- visualize scene (default: {False})
        
        """
        self._fov = fov
        self._scene = pyrender.Scene()
        self._table_dims = [1.0, 1.2, 0.6]
        self._table_pose = np.eye(4)
        self._viewer = viewing_mode

        # self._viewer_test = pyrender.viewer.Viewer()

        if viewing_mode:
            self._viewer = pyrender.Viewer(
                self._scene,
                viewport_size=(640,480),
                use_raymond_lighting=True,
                run_in_thread=True
            )
        
        self._intrinsics = intrinsics
        if self._intrinsics == 'realsense':
            self._fx = 616.36529541
            self._fy = 616.20294189
            self._cx = 310.25881958
            self._cy = 236.59980774
            self._znear = 0.04
            self._zfar = 20
            self._height = 480
            self._width = 640
        
        elif self._intrinsics == 'kinect_azure':
            self._fx = 631.54864502
            self._fy = 631.20751953
            self._cx = 638.43517329
            self._cy = 366.49904066
            self._znear = 0.04
            self._zfar = 20
            self._height = 720
            self._width = 1280
        
        else:
            print("Unknow camera. Exitint...")
            exit()
    
        # self._add_table_node()
        self._init_camera_renderer()

        self._current_context = None
        self._cache = {} if caching else None
        self._caching = caching

    def _init_camera_renderer(self):
        """
        camera init
        """
        if self._viewer:
            return
    
        if self._intrinsics in ['kinect_azure', 'realsense']:
            camera = pyrender.IntrinsicsCamera(self._fx, self._fy, self._cx, self._cy, self._znear, self._zfar)
            self._camera_node = self._scene.add(camera, pose=np.eye(4), name='camera')
            self.renderer = pyrender.OffscreenRenderer(
                                                        viewport_width=self._width,
                                                        viewport_height=self._height,
                                                        point_size=1.0
                                                      ) 
        else:
            camera = pyrender.PerspectiveCamera(yfov=self._fov, aspectRatio=1.0, znear=0.001) # do not change aspect ratio
            self._camera_node = self._scene.add(camera, pose=tra.euler_matrix(np.pi, 0, 0), name='camera')
            self.renderer = pyrender.OffscreenRenderer(400, 400)
    
    def _add_table_node(self):
        """
        table parameter
        """
        if self._viewer:
            return
        
        table_mesh = trimesh.creation.box(self._table_dims)
        #trimesh.creation.box(extents=None, transform=None, **kwargs)
        mesh = pyrender.Mesh.from_trimesh(table_mesh)

        table_node = pyrender.Node(mesh=mesh, name='table')
        # self._scene.add_node(table_node)
        # self._scene.set_pose(table_node, self._table_pose)
    
    def _load_object(self, path, scale):
        """
        parameter:
            path {str} -- path to mesh
            scale {float} -- scale of the mesh 
        return:
            dict -- contex with loaded mesh info
        """
        if (path, scale) in self._cache:
            return self._cache[(path, scale)]
        
        obj = Object(path)
        obj.rescale(scale)

        tmesh = obj.mesh
        pc = obj.to_pointcloud()

        tmesh_mean = np.mean(tmesh.vertices, 0)
        tmesh.vertices -= np.expand_dims(tmesh_mean, 0)

        lbs = np.min(tmesh.vertices, 0)
        ubs = np.max(tmesh.vertices, 0)
        object_distance = np.max(ubs - lbs) * 5

        mesh = pyrender.Mesh.from_trimesh(tmesh)

        context = {
            'name' : path + '_' + str(scale),
            'tmesh': copy.deepcopy(tmesh),
            'distance': object_distance,
            'node': pyrender.Node(mesh=mesh, name=path + '_' + str(scale)),
            'mesh_mean': np.expand_dims(tmesh_mean, 0),
            'obj_pc': pc,
        }

        self._cache[(path, scale)] = context

        return self._cache[(path, scale)]

    @staticmethod
    def mesh_to_points(obj_path):

        mesh_obj = o3d.geometry.TriangleMesh()
        mesh_obj = o3d.io.read_triangle_mesh(obj_path)
        mesh_obj.compute_vertex_normals()

        V_mesh = np.array(mesh_obj.vertices)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(V_mesh)

        return pcd

    def change_scene(self, obj_paths, obj_scales, obj_transforms):
        """
        parameter:
            obj_paths {list} -- list of object mesh paths
            obj_scales {list} -- list of object scales
            obj_transforms {list} -- list of object transforms
        """

        if self._viewer:
            self._viewer.render_lock.acquire()
        
        for n in self._scene.get_nodes():
            if n.name not in ['table', 'camera', 'parent']:
                self._scene.remove_node(n)
        
        if not self._caching:
            self._cache = {}
        
        scene_obj_pc = []

        for p,t,s in zip(obj_paths, obj_transforms, obj_scales):
            
            object_context = self._load_object(p, s)
            object_context = deepcopy(object_context)

            self._scene.add_node(object_context['node'])
            self._scene.set_pose(object_context['node'], t)

            obj = np.array(object_context['obj_pc'].vertices)
            
            obj_mean = np.mean(obj, 0, keepdims=True)
            obj -= obj_mean

            n, _ = obj.shape
            ones = np.ones((n, 1),dtype=np.float)
            pc_cat = np.concatenate([obj, ones], axis=1)
            obj_pc =np.dot(t, pc_cat.T).T
            scene_obj_pc.append(obj_pc)

        pc_all = np.concatenate(scene_obj_pc, axis=0)
        
        if self._viewer:
            self._viewer.render_lock.release()

        return pc_all

    
    def _to_pointcloud(self, depth):
        """
        将深度图像转换成点云

        参数：
            depth {np.ndarray} -- HxW depth map
        返回：
            np.ndarray -- Nx4 homog. point cloud
        """
        if self._intrinsics in ['kinect_azure', 'realsense']:
            fx = self._fx
            fy = self._fy

            height = self._height
            width = self._width

            cx = self._cx
            cy = self._cy

            mask = np.where(depth > 0)

            x = mask[1]
            y = mask[0]

            normalized_x = (x.astype(np.float32) - cx)
            normalized_y = (y.astype(np.float32) - cy)

        else:
            fy = fx = 0.5 / np.tan(self._fov * 0.5)
            height = depth.shape[0]
            width = depth.shape[1]

            mask = np.where(depth > 0)

            x = mask[1]
            y = mask[0]

            normalized_x = (x.astype(np.float32) - width * 0.5) / width
            normalized_y = (y.astype(np.float32) - height * 0.5) / height
        
        world_x = normalized_x * depth[y, x] / fx
        world_y = normalized_y * depth[y, x] / fy
        world_z = depth[y, x]

        ones = np.ones(world_z.shape[0], dtype=np.float32)

        return np.vstack((world_x, world_y, world_z, ones)).T
    

    def render(self, pose, render_pc=True):
        """
        parameter:
            pose {np.ndarray} -- 4x4 camera pose
        return:
            [np.ndarray, np.ndarray, np.ndarray, np.ndarray] -- HxWx3 color, HxW depth, Nx4 point cloud, 4x4 camera pose
        """
        transferred_pose = pose.copy()

        self._scene.set_pose(self._camera_node, transferred_pose)

        color, depth = self.renderer.render(self._scene)

        if render_pc:
            pc = self._to_pointcloud(depth)
        else:
            pc = None
        
        return color, depth, pc, transferred_pose


    def render_labels(self, full_depth, obj_paths, obj_scales, render_pc=False):
        """
        parameter:
            full_depth {np.ndarray} -- HxW depth map
            obj_paths {list} -- list of object paths in scene
            obj_scales {list} -- list of object scales in scene
        
        return:
            [np.ndarray, list, dict] -- integer segmap with 0=background, list of 
                                        corresponding object names, dict of corresponding point clouds
        """ 
        scene_object_nodes = []
        for n in self._scene.get_nodes():
            if n.name not in ['camera', 'parent']:
                n.mesh.is_visible = False
                if n.name != 'table':
                    scene_object_nodes.append(n)
        
        obj_names = [path + '_' + str(scale) for path, scale in zip(obj_paths, obj_scales)]

        pcs = {}
        output = np.zeros(full_depth.shape, np.uint8)
        for n in scene_object_nodes:
            n.mesh.is_visible = True

            depth = self.renderer.render(self._scene)[1]
            mask = np.logical_and((np.abs(depth - full_depth) < 1e-6), np.abs(full_depth) > 0)

            if not np.any(mask):
                continue
            if np.any(output[mask] != 0):
                raise ValueError('wrong label')
            
            indices = [i+1 for i, x in enumerate(obj_names) if x == n.name]
            for i in indices:
                if not np.any(output == i):
                    print('')
                    output[mask] = i
                    break
            
            n.mesh.is_visible = False

            if render_pc:
                pcs[i] = self._to_pointcloud(depth * mask)
        
        for n in self._scene.get_nodes():
            if n.name not in ['camera', 'parent']:
                n.mesh.is_visible = True
        
        return output, ['BACKGROUND'] + obj_names, pcs
    
