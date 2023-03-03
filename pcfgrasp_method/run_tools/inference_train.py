from torch.utils.data import DataLoader
from utils.load_data import load_available_input_data
from utils.grasp_estimator import GraspEstimatior, extract_point_clouds
from utils.visual_grasp import visualize_grasps
import glob

def train_inference(args, global_config, data_set=None, K=None, z_range=[0.2, 1.8] ,forward_passes=1):
    """
    Predict 6-DoF grasp distribution for given model and input data
    """
    grasp_estimatior = GraspEstimatior(global_config)

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

            pred_grasps_cam, scores, contact_pts, _, coarse= grasp_estimatior.predict_scene_grasps(pc, args, obj_pc, forward_passes=forward_passes)

            visualize_grasps(pc, coarse, pred_grasps_cam, scores, obj_pc=obj_pc, plot_opencv_cam=True)


        


