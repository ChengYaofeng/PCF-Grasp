import imp
import numpy as np
from torch.utils.data import Dataset
from utils.load_data import load_contact_grasps
from utils.data import center_pc_convert_cam, center_pc_gt_convert_cam
from run_utils.logger import print_log
from utils.data import vis_pc
class ContactDataset(Dataset):
    def __init__(self, 
                global_config, 
                pcreader, 
                contact_infos,
                split='train', 
                logger = None,
                ):
        """
        split: 'train' or 'test' or 'eval'
        """
        super().__init__()

        self.pcreader = pcreader
        self.split = split
        
        tf_pos_contact_points, tf_pos_contact_dirs, tf_pos_contact_approaches, \
            tf_pos_finger_diffs, tf_scene_idcs = load_contact_grasps(contact_infos, global_config['DATA'])

        self.tf_pos_contact_points = tf_pos_contact_points
        self.tf_pos_contact_dirs = tf_pos_contact_dirs
        self.tf_pos_contact_approaches = tf_pos_contact_approaches
        self.tf_pos_finger_diffs = tf_pos_finger_diffs
        self.tf_scene_idcs = tf_scene_idcs

        num_test_samples = global_config['DATA']['num_test_scenes']
        num_train_samples = len(contact_infos)-num_test_samples

        scene_idxs = []

        if split == 'train':
            for scene_idx in range(num_train_samples):
                scene_idxs.append(scene_idx)
        elif split == 'test':
            for scene_idx in range(num_test_samples):
                scene_idxs.append(scene_idx + num_train_samples)
        else:
            num_all_samples = num_test_samples + num_train_samples
            for i in range(num_all_samples):
                scene_idxs.append(i)

        self.scene_idxs = scene_idxs

        print_log("Totally {} samples in {} set.".format(len(self.scene_idxs), split), logger=logger)


    def __getitem__(self, idx):
        """
        parameter:
            idx is the batch_idx of training or testing
        """
        tf_pos_contact_points_idx, tf_pos_contact_dirs_idx, tf_pos_contact_approaches_idx, tf_pos_finger_diffs_idx, tf_scene_idcs_idx = \
            self.tf_pos_contact_points[idx], self.tf_pos_contact_dirs[idx], self.tf_pos_contact_approaches[idx], self.tf_pos_finger_diffs[idx], self.tf_scene_idcs[idx]
        

        batch_points_raw, cam_poses, sce_idx, obj_pc = self.pcreader.get_scene_batch(scene_idx=self.scene_idxs[idx]) #BNC

        batch_points, cam_poses_1 = center_pc_convert_cam(cam_poses, batch_points_raw) #1024 3; 4 x 4
        obj_pc, _ = center_pc_gt_convert_cam(cam_poses, batch_points_raw, obj_pc) #2048 3

        labels_dict = {'tf_pos_contact_points_idx': tf_pos_contact_points_idx,
                        'tf_pos_contact_dirs_idx': tf_pos_contact_dirs_idx,
                        'tf_pos_contact_approaches_idx': tf_pos_contact_approaches_idx,
                        'tf_pos_finger_diffs_idx': tf_pos_finger_diffs_idx,
                        'tf_scene_idcs_idx': tf_scene_idcs_idx,
                        'target': obj_pc}

        if self.split == 'eval':
            return batch_points_raw, obj_pc
        else:
            return batch_points, cam_poses_1, labels_dict


    def __len__(self):
        return len(self.scene_idxs)
