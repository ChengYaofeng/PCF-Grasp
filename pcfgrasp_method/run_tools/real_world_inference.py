#! /home/cyf/anaconda3/envs/tcg/bin/python3.7
import os
os.environ['PYOPENGL_PLATFORM']='osmesa'
os.environ['CUDA_VISIBLE_DEVICES']='0'

import sys
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) #/home/cyf/6d_grasp/completion_method
# print(BASE_DIR)
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'utils'))
sys.path.append(os.path.join(BASE_DIR, 'run_tools'))
sys.path.append(os.path.join(BASE_DIR, 'model'))
sys.path.append(os.path.join(BASE_DIR, 'run_utils'))


from utils.grasp_estimator import GraspEstimatior, extract_point_clouds
from utils.visual_grasp import visualize_grasps
from run_utils.config import load_config
import numpy as np
import argparse
import torch
import torch.functional as F
import cv2
import open3d as o3d
import pyrealsense2 as rs
import multiprocessing as mp
from mask_rcnn.demo.demo import get_parser, setup_cfg
from mask_rcnn.demo.predictor import VisualizationDemo

###############if ros##################
from contact_graspnet_ros.msg import objects_grasp_pose
from geometry_msgs.msg import Pose
import rospy
from autolab_core import RigidTransform

def pose_publisher(obj_id, grasp_pose):
    """
    input:
        obj_id: list
        grasp_pose: list
    """ 

    # 创建publisher
    pub = rospy.Publisher('grasp_pose', data_class=objects_grasp_pose, queue_size=1)
    rospy.init_node('talker', anonymous = True)
    rate = rospy.Rate(10)

    pose_grasp_msg = objects_grasp_pose()
    
    #can put total in a cyclic if there are more than 1 object ot grasp
    total = 1

    grasp_pose_msg = Pose()
    grasp_pose_t = np.array([grasp_pose[0][3], grasp_pose[1][3], grasp_pose[2][3]])
    grasp_pose_r = grasp_pose[:3, :3]

    #rotation to quaternion
    grasp_pose = RigidTransform(rotation=grasp_pose_r, translation=grasp_pose_t)
    #position
    grasp_pose_msg.position.x = grasp_pose.position[0]
    grasp_pose_msg.position.y = grasp_pose.position[1]
    grasp_pose_msg.position.z = grasp_pose.position[2]
    #orientation
    grasp_pose_msg.orientation.w = grasp_pose.quaternion[0]
    grasp_pose_msg.orientation.x = grasp_pose.quaternion[1]
    grasp_pose_msg.orientation.y = grasp_pose.quaternion[2]
    grasp_pose_msg.orientation.z = grasp_pose.quaternion[3]

    #push index and grasp_pose
    pose_grasp_msg.obj_index.append(obj_id)
    pose_grasp_msg.grasp_pose.append(grasp_pose_msg)
        # print(id)
        # print(pose)

    while pub.get_num_connections() < 1:
    # while not rospy.is_shutdown():

        rospy.loginfo("Publish {} objects's best grasp pose".format(total))
        for i in range(total):
            rospy.loginfo("object {}'s Pose".format(pose_grasp_msg.obj_index[i]))
            rospy.loginfo("\t position: x:{:.8f} y:{:.8f} z:{:.8f}".format(
                pose_grasp_msg.grasp_pose[i].position.x,
                pose_grasp_msg.grasp_pose[i].position.y, 
                pose_grasp_msg.grasp_pose[i].position.z
            ))
            rospy.loginfo("\t orientation: x:{:.8f} y:{:.8f} z:{:.8f} w:{:.8f}".format(
                          pose_grasp_msg.grasp_pose[i].orientation.x,
                          pose_grasp_msg.grasp_pose[i].orientation.y,
                          pose_grasp_msg.grasp_pose[i].orientation.z,
                          pose_grasp_msg.grasp_pose[i].orientation.w
            ))
            print("\n")
            rate.sleep()
    pub.publish(pose_grasp_msg)


def main(args, K=None, z_range=[0.2,1.2] ,forward_passes=1):

    global_config = load_config(args.config_dir, batch_size=1, max_epoch=1, 
                                          data_path= args.data_path, arg_configs=args.arg_configs, save=True)
    torch.backends.cudnn.benchmark = True

    grasp_estimatior = GraspEstimatior(global_config)

    align = rs.align(rs.stream.color)

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

            key = cv2.waitKey(1)

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
                args.config_file = '~/pcf_grasp/pcfgrasp_method/mask_rcnn/configs/COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml'
                args.opts = ['MODEL.WEIGHTS', 'detectron2://COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x/139653917/model_final_2d9806.pkl']
                cfg = setup_cfg(args)

                demo = VisualizationDemo(cfg)
                predictions, visualized_output = demo.run_on_image(rgb)
                WINDOW_NAME = "Grasp detections"
                cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
                # print(visualized_output.get_image()[:, :, ::-1].shape)  #(480, 640, 3)

                cv2.imshow(WINDOW_NAME, visualized_output.get_image()[:, :, ::-1])
                if pc_full is None:
                    print('Converting depth to point cloud(s)...')
                    pc_full, pc_segments, pc_colors = extract_point_clouds(depth, cam_K, segmap=segmap, rgb=rgb, skip_border_objects=False, z_range=z_range)

                mask = predictions['instances'].pred_masks.cpu().numpy() * depth

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
                
                obj_num = predictions['instances'].pred_classes.tolist()
                if idc > len(obj_num):
                    raise Exception('Input out of bounds', key_select-48)
                
                if obj_pc is None:
                    print('Converting depth to point cloud(s)...')
                    obj_pc, _, obj_colors = extract_point_clouds(mask[idc], cam_K, segmap=segmap, rgb=rgb, skip_border_objects=False, z_range=z_range)
                file_num += 1
                
                print('Generating Grasps...')
                pred_grasps_cam, scores, contact_pts, _, coarse = grasp_estimatior.predict_scene_grasps(obj_pc, args, pc_segments= pc_segments, forward_passes=forward_passes)

                best_grasp = visualize_grasps(obj_pc, coarse, pred_grasps_cam, scores, plot_opencv_cam=True, pc_colors=obj_colors)

                ###############if ros 
                # try:
                #     pose_publisher(idc, best_grasp[0])
                # except rospy.ROSInterruptException:
                #     pass

            #press 'q' or 'esc' to stop
            elif key & 0xFF == ord('q') or key == 27:
                cv2.destroyAllWindows()
                break   

    finally:
        pipeline.stop()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--pretrain_vis', type=bool, default=False)
    parser.add_argument('--pretrain_ckpt', type=str, default='path to pcn pth file')

    parser.add_argument('--ckpt_dir', type=str, default='path to pcf grasp pth file')
    parser.add_argument('--ori_inference', type=bool, default=True)

    parser.add_argument('--filter', type=bool, default=False)

    parser.add_argument('--input_path', type=str, default=None, help='train_inference picture pcd scene or object waiting for grasp generation')
    parser.add_argument('--arg_configs', nargs="*", type=str, default=[], help='overwrite config parameters')

    parser.add_argument('--log_name', type=str, default='inference_log', help='logger name')
    parser.add_argument('--exp_name', type=str, default='vis', help='expariment name')
    parser.add_argument('--data_path', type=str, default=None)
    parser.add_argument('--config_dir', type=str, default='~/pcf_grasp/pcfgrasp_method/')

    #mask-rcnn
    parser.add_argument('--config_file', metavar="FILE", default='~/pcf_grasp/pcfgrasp_method/mask_rcnn/configs/COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml')
    parser.add_argument('--opts', nargs=argparse.REMAINDER, default=['MODEL.WEIGHTS', 'detectron2://COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x/139653917/model_final_2d9806.pkl'])
    parser.add_argument('--confidence-threshold', type=float, default=0.5, help="Minimum score for instance predictions to be shown",)

    args = parser.parse_args()

    main(args)