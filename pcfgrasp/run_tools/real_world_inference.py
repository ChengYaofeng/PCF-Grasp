#! /home/cyf/anaconda3/envs/tcg/bin/python3.7
import os
# os.environ['PYOPENGL_PLATFORM']='osmesa'
# os.environ['CUDA_VISIBLE_DEVICES']='0'

import sys
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) #/home/cyf/6d_grasp/completion_method
# print(BASE_DIR)
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'utils'))
sys.path.append(os.path.join(BASE_DIR, 'run_tools'))
sys.path.append(os.path.join(BASE_DIR, 'model'))
sys.path.append(os.path.join(BASE_DIR, 'run_utils'))

# from autolab_core import RigidTransform

from utils.grasp_estimator import GraspEstimatior, extract_point_clouds, extract_point_clouds_mask
from utils.visual_grasp import visualize_grasps_o3d
from run_utils.config import load_config
import numpy as np
import argparse
import torch
import torch.functional as F
import cv2
import open3d as o3d
import pyrealsense2 as rs
import multiprocessing as mp

# from contact_graspnet_ros.msg import objects_grasp_pose
# from geometry_msgs.msg import Pose
# import rospy
from ultralytics import YOLO




def save_npy(pc, pc_num=1, color=None):
    """
    input:
        pc: {ndarray} N x 3
        color: {ndarray} N x 3
    """
    dict = {}

    dict['xyz'] = pc
    if color is not None:
        dict['xyz_color'] = color

    np.save('/home/franka/contact_graspnet/test_data/614_{}.npy'.format(pc_num), dict)
    print('saved file in /home/franka/contact_graspnet/test_data/614_{}.npy'.format(pc_num))


def main(args, K=None, z_range=[0.2,1.2] ,forward_passes=1):

    global_config = load_config(args.config_dir, batch_size=1, max_epoch=1, 
                                          data_path= args.data_path, arg_configs=args.arg_configs, save=True)
    torch.backends.cudnn.benchmark = True

    grasp_estimatior = GraspEstimatior(global_config)

    align = rs.align(rs.stream.color) #深度图和rgb图对齐

    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)
    pipeline = rs.pipeline()
    profile = pipeline.start(config)

    # get camera intrinsics
    intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()

    file_num = 0

    try:
        while True:
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)

            color_frame = aligned_frames.get_color_frame()
            color_image = np.asanyarray(color_frame.get_data())
            depth_frame = aligned_frames.get_depth_frame()

            depth_frame = rs.decimation_filter(1).process(depth_frame)
            depth_frame = rs.disparity_transform(True).process(depth_frame)
            depth_frame = rs.spatial_filter().process(depth_frame)
            depth_frame = rs.temporal_filter().process(depth_frame)
            depth_frame = rs.disparity_transform(False).process(depth_frame)
            # depth_frame = rs.hole_filling_filter().process(depth_frame)

            depth_image = np.asanyarray(depth_frame.get_data())
            color_image1 = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)

            cv2.namedWindow('color image', cv2.WINDOW_AUTOSIZE)
            cv2.imshow('color image', cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR))

            ####### vis depth
            # colorizer = rs.colorizer()
            # colorizer_depth = np.asanyarray(colorizer.colorize(depth_frame).get_data())
            # cv2.imshow('colorizer depth', colorizer_depth)

            key = cv2.waitKey(1)
            
            ### yolo
            yolo_model = YOLO("ultralytics/yolov8s-seg.pt")


            # press ' ' to save current RGBD images and pointcloud.
            segmap = None
            pc_full = None
            pc_colors = None
            obj_pc = None

            if key & 0xFF == ord(' '):
                rgb = color_image1
                depth = depth_image / 1000.0
                cam_K = np.array([[intr.fx, 0, intr.ppx],
                                [0, intr.fy, intr.ppy], 
                                [0,    0,       1]])
                

                mp.set_start_method("spawn", force=True)
                # argparse = get_parser().parse_args()
                results = yolo_model(rgb)
                masks = results[0].masks.data.cpu().numpy()  # 获取掩码
                boxes = results[0].boxes.xyxy.cpu().numpy()  # 获取边界框

                overlay = rgb.copy() #
                #---------- 过去的可视化
                for i, mask in enumerate(masks):
                    # 处理掩码为二值化（0 或 255）
                    binary_mask = (mask > 0.5).astype(np.uint8) * 255

                    # 为掩码生成随机颜色
                    color = np.random.randint(0, 255, (3,), dtype=np.uint8)

                    # 使用掩码为图像添加颜色
                    overlay[binary_mask > 0] = overlay[binary_mask > 0] * 0.5 + color * 0.5

                    # 绘制边界框
                    x1, y1, x2, y2 = boxes[i]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    cv2.rectangle(overlay, (x1, y1), (x2, y2), color.tolist(), 2)

                    # 绘制类别标签
                    label = f"{int(i)}"
                    cv2.putText(overlay, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color.tolist(), 2)
                
                # mask 可视化
                WINDOW_NAME = "Grasp detections"
                cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
                cv2.imshow(WINDOW_NAME, overlay)
                
                if pc_full is None:
                    print('Converting depth to point cloud(s)...')
                    pc_full, pc_segments, pc_colors = extract_point_clouds(depth, cam_K, segmap=segmap, rgb=color_image, skip_border_objects=False, z_range=z_range)
                #------------------------------------------

                masks_d = masks * depth

                # select object to generate grasp
                key_select = cv2.waitKey(0)
                if key_select & 0xFF == ord('0'):
                    idc = 0
                elif key_select & 0xFF == ord('1'):
                    idc = 1
                elif key_select & 0xFF == ord('2'):
                    idc = 2
                elif key_select & 0xFF == ord('3'):
                    idc = 3
                elif key_select & 0xFF == ord('4'):
                    idc = 4
                elif key_select & 0xFF == ord('5'):
                    idc = 5
                elif key_select & 0xFF == ord('6'):
                    idc = 6
                elif key_select & 0xFF == ord('7'):
                    idc = 7
                elif key_select & 0xFF == ord('8'):
                    idc = 8
                elif key_select & 0xFF == ord('9'):
                    idc = 9
                else:
                    print('plece input number 0-9 object to select')
                    raise Exception('Wrong Input', key_select-48)
                
                obj_num = len(masks)
                if idc > obj_num:
                    raise Exception('Input out of bounds', key_select-48)


                if obj_pc is None:
                    print('Converting depth to point cloud(s)...')
                    obj_pc, _, obj_colors = extract_point_clouds(masks_d[idc], cam_K, segmap=segmap, rgb=color_image, skip_border_objects=False, z_range=z_range)
                # save_npy(obj_pc, pc_num=file_num, color=None)
                file_num += 1
                # #----------------------------------------------------------------
                #-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
                if pc_full is None:
                    print('Converting depth to point cloud(s)......')
                    pc_full, pc_segments, pc_colors = extract_point_clouds(depth, cam_K, segmap=segmap, rgb=color_image, skip_border_objects=False, z_range=z_range)

            #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

                print('Generating Grasps...')
                pred_grasps_cam, scores, contact_pts, _, coarse = grasp_estimatior.predict_scene_grasps(obj_pc, args, pc_segments= pc_segments, forward_passes=forward_passes)

                best_grasp = visualize_grasps_o3d(obj_pc, coarse, pred_grasps_cam, scores, plot_opencv_cam=True, pc_colors=obj_colors)
                best_grasp = []
                best_grasp.append([pred_grasps_cam[-1][np.argmax(scores[-1])]][0])
                
            #     try:
            #         pose_publisher(idc, best_grasp[0])
            #     except rospy.ROSInterruptException:
            #         pass

            # elif key & 0xFF == ord('q') or key == 27:
            #     cv2.destroyAllWindows()
            #     break   

    finally:
        pipeline.stop()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--pretrain_vis', type=bool, default=False)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--pretrain_ckpt', type=str, default='/home/cyf/04_grasp/contact_graspnet_ros/src/pcf_grasp/checkpoints/best_l1_cd_320.pth')

    parser.add_argument('--ori_inference', type=bool, default=False)
    parser.add_argument('--ckpt_dir', type=str, default='/home/cyf/04_grasp/contact_graspnet_ros/src/pcf_grasp/checkpoints/specific_model/plan_h_large_AdamW/06-24-10-40_best_train_289.pth')

    # parser.add_argument('--ckpt_dir', type=str, default='/home/cyf/04_grasp/contact_graspnet_ros/src/pcf_grasp/checkpoints/specific_model/best_ori/06-18-10-20_best_ori_277.pth')
    # parser.add_argument('--ori_inference', type=bool, default=True)

    parser.add_argument('--filter', type=bool, default=False)
    parser.add_argument('--rot_matrix', type=list, default=[-0.611174926447, -0.629137760148, 0.363306169293, 0.314101122875]) #xyzw
    parser.add_argument('--tra_matrix', type=list, default=[0.953067844932, 0.0356128837323, 0.616674880853])

    parser.add_argument('--input_path', type=str, default=None, help='train_inference picture pcd scene or object waiting for grasp generation')
    parser.add_argument('--arg_configs', nargs="*", type=str, default=[], help='overwrite config parameters')

    parser.add_argument('--log_name', type=str, default='inference_log', help='logger name')
    parser.add_argument('--exp_name', type=str, default='vis', help='expariment name')
    parser.add_argument('--data_path', type=str, default=None)
    parser.add_argument('--config_dir', type=str, default='/home/cyf/04_grasp/contact_graspnet_ros/src/pcf_grasp/')

    args = parser.parse_args()

    main(args)