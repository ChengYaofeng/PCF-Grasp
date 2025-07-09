import numpy as np
import matplotlib.pyplot as plt
from utils import mesh_utils
import open3d as o3d
import trimesh
from scipy.spatial.transform import Rotation as R

def plot_mesh_o3d(mesh: trimesh.Trimesh, cam_trafo=np.eye(4), mesh_pose=np.eye(4)) -> o3d.geometry.TriangleMesh:
    """
    将 Trimesh 网格转为 Open3D 网格，并应用 cam_trafo 和 mesh_pose 变换

    Args:
        mesh (trimesh.Trimesh): 输入三角网格
        cam_trafo (np.ndarray): 相机外参（世界到相机）4x4
        mesh_pose (np.ndarray): 网格姿态（mesh 到世界）4x4

    Returns:
        o3d.geometry.TriangleMesh: 已变换后的 Open3D 网格对象
    """

    # 顶点变换：mesh local -> world -> camera
    vertices = mesh.vertices  # (N, 3)
    faces = mesh.faces        # (M, 3)

    homog_vert = np.concatenate([vertices, np.ones((vertices.shape[0], 1))], axis=1)  # (N, 4)
    transformed_vert = (homog_vert @ mesh_pose.T @ cam_trafo.T)[:, :3]

    # 构造 Open3D 网格
    o3d_mesh = o3d.geometry.TriangleMesh()
    o3d_mesh.vertices = o3d.utility.Vector3dVector(transformed_vert)
    o3d_mesh.triangles = o3d.utility.Vector3iVector(faces)
    o3d_mesh.paint_uniform_color([0.3, 0.6, 1.0])  # 淡蓝色
    o3d_mesh.compute_vertex_normals()

    return o3d_mesh

def plot_coordinates_o3d(t: np.ndarray, r: np.ndarray, size=0.2) -> o3d.geometry.TriangleMesh:
    """
    使用 Open3D 绘制一个坐标轴

    Args:
        t (np.ndarray): 位置向量 (3,)
        r (np.ndarray): 旋转矩阵 (3, 3)
        size (float): 坐标轴长度（等效于 0.2）

    Returns:
        o3d.geometry.TriangleMesh: 坐标轴网格，可添加到可视化场景中
    """
    coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size)
    T = np.eye(4)
    T[:3, :3] = r
    T[:3, 3] = t
    coord.transform(T)
    return coord


def show_image(rgb, segmap):
    """
    Overlay rgb image with segmentation and imshow segment

    Arguments:
        rgb {np.ndarray} -- color image
        segmap {np.ndarray} -- integer segmap of same size as rgb
    """
    plt.figure()
    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()
    
    plt.ion()
    plt.show()
    
    if rgb is not None:
        plt.imshow(rgb)
    if segmap is not None:
        cmap = plt.get_cmap('rainbow')
        cmap.set_under(alpha=0.0)   
        plt.imshow(segmap, cmap=cmap, alpha=0.5, vmin=0.0001)
    plt.draw()
    plt.pause(0.001)


def visualize_grasps_o3d(full_pc, coarse, pred_grasps_cam, scores,
                         obj_pc=None,
                         plot_opencv_cam=True,
                         pc_colors=None,
                         gripper_openings=None,
                         gripper_width=0.08):
    """
    Open3D 抓取姿态可视化函数，支持点云着色、姿态线框、最佳抓取、相机坐标系。
    """
    print('Visualizing (Open3D)...')

    geometries = []

    # 主点云
    full_pcd = draw_pc_with_colors_o3d(full_pc, pc_colors, single_color=(1, 0, 0))
    geometries.append(full_pcd)

    # # coarse 点云
    # if coarse is not None:
    #     coarse_pcd = draw_pc_with_colors_o3d(coarse, single_color=(0.5, 0.1, 0.1))
    #     geometries.append(coarse_pcd)

    # 相机坐标系
    if plot_opencv_cam:
        cam_coord = plot_coordinates_o3d(np.zeros(3,), np.eye(3), size=0.1)
        geometries.append(cam_coord)

    # colormap
    cm = plt.get_cmap('rainbow')
    cm2 = plt.get_cmap('gist_rainbow')

    best_grasp = []

    for i, k in enumerate(pred_grasps_cam):
        if np.any(pred_grasps_cam[k]):
            grasps = pred_grasps_cam[k]
            scores_k = scores[k]
            openings_k = gripper_openings.get(k, np.ones(len(grasps)) * gripper_width) if gripper_openings else np.ones(len(grasps)) * gripper_width

            # 所有抓取（细线）
            grasp_color = cm(i / len(pred_grasps_cam))[:3]
            geometries += draw_grasps_o3d(grasps, np.eye(4), openings_k, color=grasp_color, tube_radius=0.001)

            # 最佳抓取（粗线）
            best_idx = np.argmax(scores_k)
            # sorted_indices = np.argsort(scores_k)
            # best_idx = sorted_indices[-10]
            best_pose = grasps[best_idx]
            best_opening = openings_k[best_idx]
            highlight_color = cm2(0.5)[:3]

            geometries += draw_grasps_o3d([best_pose], np.eye(4), [best_opening], color=highlight_color, tube_radius=0.003)
            print(f'---- best score for segment {k}: {scores_k[best_idx]:.4f}')
            best_grasp.append(best_pose)

    o3d.visualization.draw_geometries(geometries)
    return best_grasp

def draw_pc_with_colors_o3d(pc, pc_colors=None, single_color=(0, 1, 0)):
    """
    使用 Open3D 绘制带颜色点云

    Args:
        pc (np.ndarray): Nx3 点云坐标
        pc_colors (np.ndarray, optional): Nx3 RGB 颜色 (0-255 or 0-1). Defaults to None.
        single_color (tuple, optional): 如果不提供 pc_colors，统一使用该颜色. Defaults to (0,1,0).

    Returns:
        o3d.geometry.PointCloud: 可添加到场景中的点云对象
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc)

    if pc_colors is not None:
        # 转为 [0,1] 范围
        pc_colors = np.asarray(pc_colors)
        if pc_colors.max() > 1.0:
            pc_colors = pc_colors / 255.0
        pcd.colors = o3d.utility.Vector3dVector(pc_colors)
    else:
        pcd.paint_uniform_color(single_color)

    return pcd


def create_thick_line(p1, p2, radius=0.002, resolution=20):
    direction = p2 - p1
    height = np.linalg.norm(direction)
    if height < 1e-6:
        return None
    direction_unit = direction / height

    cylinder = o3d.geometry.TriangleMesh.create_cylinder(radius=radius, height=height, resolution=resolution)
    cylinder.compute_vertex_normals()

    # Align with z-axis
    z_axis = np.array([0, 0, 1])
    rot = R.align_vectors([direction_unit], [z_axis])[0].as_matrix()
    transform = np.eye(4)
    transform[:3, :3] = rot
    transform[:3, 3] = (p1 + p2) / 2
    cylinder.transform(transform)
    return cylinder

# def draw_grasps_o3d(grasps, cam_pose, gripper_openings, color=(1, 0, 0), colors=None,
#                     show_gripper_mesh=False, tube_radius=0.003):
#     """
#     使用 Open3D 的粗 cylinder 渲染抓取姿态
#     """
#     geometries = []
#     gripper = mesh_utils.create_gripper('panda')
#     control_pts = gripper.get_control_point_tensor(1, False, convex_hull=False).squeeze()

#     mid_point = 0.5 * (control_pts[1] + control_pts[2])
#     grasp_line = np.array([
#         [0, 0, 0],         # base
#         mid_point,         # center
#         control_pts[1],    # finger left root
#         control_pts[3],    # finger left tip
#         control_pts[2],    # finger right root
#         control_pts[4],    # finger right tip
#         mid_point          # tip center
#     ])

#     connections = [
#         [0, 1], [1, 2], [2, 3], [1, 4], [4, 5], [1, 6]
#     ]

#     for i, (g_pose, opening) in enumerate(zip(grasps, gripper_openings)):
#         grasp_pts = grasp_line.copy()
#         grasp_pts[2:, 0] = np.sign(grasp_pts[2:, 0]) * opening / 2.0

#         pts_world = (g_pose[:3, :3] @ grasp_pts.T).T + g_pose[:3, 3]
#         pts_homog = np.concatenate([pts_world, np.ones((pts_world.shape[0], 1))], axis=1)
#         pts_cam = (cam_pose @ pts_homog.T).T[:, :3]

#         grasp_color = color if colors is None else colors[i]
#         for c0, c1 in connections:
#             cyl = create_thick_line(pts_cam[c0], pts_cam[c1], radius=tube_radius)
#             if cyl:
#                 cyl.paint_uniform_color(grasp_color)
#                 geometries.append(cyl)

#         if show_gripper_mesh and i == 0:
#             mesh = gripper.hand.copy()
#             mesh.transform(cam_pose @ g_pose)
#             mesh.paint_uniform_color(grasp_color)
#             geometries.append(mesh)

#     return geometries

def draw_grasps_o3d(grasps, cam_pose, gripper_openings, color=(1, 0, 0), colors=None,
                    show_gripper_mesh=False, tube_radius=0.003):
    """
    使用 Open3D 渲染抓取姿态（粗 cylinder 模型）
    """
    geometries = []
    gripper = mesh_utils.create_gripper('panda')  # 你已有的接口
    control_pts = gripper.get_control_point_tensor(1, False, convex_hull=False).squeeze()

    mid_point = 0.5 * (control_pts[1] + control_pts[2])
    grasp_line = np.array([
        [0, 0, 0],         # base
        mid_point,         # center
        control_pts[1],    # finger left root
        control_pts[3],    # finger left tip
        control_pts[2],    # finger right root
        control_pts[4],    # finger right tip
        mid_point          # tip center
    ])
    connections = [
        [0, 1], [1, 2], [2, 3], [1, 4], [4, 5], [1, 6]
    ]

    for i, (g_pose, opening) in enumerate(zip(grasps, gripper_openings)):
        grasp_pts = grasp_line.copy()
        grasp_pts[2:, 0] = np.sign(grasp_pts[2:, 0]) * opening / 2.0

        # 应用抓取姿态和相机外参变换
        pts_world = (g_pose[:3, :3] @ grasp_pts.T).T + g_pose[:3, 3]
        pts_homog = np.concatenate([pts_world, np.ones((pts_world.shape[0], 1))], axis=1)
        pts_cam = (cam_pose @ pts_homog.T).T[:, :3]

        grasp_color = color if colors is None else colors[i]

        for c0, c1 in connections:
            cyl = create_thick_line(pts_cam[c0], pts_cam[c1], radius=tube_radius)
            if cyl:
                cyl.paint_uniform_color(grasp_color)
                geometries.append(cyl)

        if show_gripper_mesh and i == 0:
            mesh = gripper.hand.copy()
            mesh.transform(cam_pose @ g_pose)
            mesh.paint_uniform_color(grasp_color)
            geometries.append(mesh)

    return geometries

    
def plot_grasp_o3d(pred_grasps_cam, gripper_openings=None, gripper_width=0.08):
    """
    用 Open3D 显示抓取姿态集合
    """
    assert isinstance(pred_grasps_cam, dict), "Expected dict of {int: List[np.ndarray]}"
    geometries = []
    cm = plt.get_cmap('rainbow')
    cm2 = plt.get_cmap('gist_rainbow')

    scores = {k: np.ones(len(pred_grasps_cam[k])) for k in pred_grasps_cam}
    colors2 = {k: cm2(0.5)[:3] for k in pred_grasps_cam}

    for k in pred_grasps_cam:
        if np.any(pred_grasps_cam[k]):
            grasp_list = pred_grasps_cam[k]
            openings = np.ones(len(grasp_list)) * gripper_width if gripper_openings is None else gripper_openings[k]

            # 所有抓取（细线）
            grasp_geoms = draw_grasps_o3d(grasp_list, cam_pose=np.eye(4),
                                          gripper_openings=openings,
                                          color=(0.2, 0.9, 0.5),
                                          tube_radius=0.01)
            geometries += grasp_geoms

            # 最佳抓取（粗线）
            best_idx = np.argmax(scores[k])
            best_grasp = [grasp_list[best_idx]]
            best_geom = draw_grasps_o3d(best_grasp, cam_pose=np.eye(4),
                                        gripper_openings=[openings[best_idx]],
                                        color=colors2[k],
                                        tube_radius=0.03)
            geometries += best_geom

    o3d.visualization.draw_geometries(geometries)
