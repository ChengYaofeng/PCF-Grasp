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
        mode: train or test
        num: num of points
        add_noise: {bool}
        root: path of dataset root
        noise_trans: 
                        parser.add_argument('--noise_trans', default=0.03, help='range of the random noise of translation added to the training data')

        refine: 
                        parser.add_argument('--resume_refinenet', type=str, default = '',  help='resume PoseRefineNet model')
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
        参数：
            pcreader is a class
            idx is the batch_idx of training or testing 这里存在一个疑问, 场景的idx和obj的idx是一样的不
            scene_idx 可以通过rgb的idx在达到一定的数量后进行迭代更新
        """
        tf_pos_contact_points_idx, tf_pos_contact_dirs_idx, tf_pos_contact_approaches_idx, tf_pos_finger_diffs_idx, tf_scene_idcs_idx = \
            self.tf_pos_contact_points[idx], self.tf_pos_contact_dirs[idx], self.tf_pos_contact_approaches[idx], self.tf_pos_finger_diffs[idx], self.tf_scene_idcs[idx]
        

        batch_points_raw, cam_poses, sce_idx, obj_pc = self.pcreader.get_scene_batch(scene_idx=self.scene_idxs[idx]) #BNC


        # print(batch_points.shape)
        # vis_pc(np.concatenate([batch_points_raw.squeeze()[:, :3],obj_pc[:, :3]],axis=0))
        # print(obj_pc.shape)
        # print('------------------{}'.format(cam_poses.shape))
        # print(batch_points_raw.shape)
        # print(np.expand_dims(cam_poses,axis=0).shape)
        batch_points, cam_poses_1 = center_pc_convert_cam(cam_poses, batch_points_raw)
        obj_pc, _ = center_pc_gt_convert_cam(cam_poses, batch_points_raw, obj_pc)
        # print(batch_points.shape)
        # print(obj_pc.shape)

        # vis_pc(batch_points[:,:3].squeeze())
        # vis_pc(np.concatenate([batch_points[:, :3].squeeze(),obj_pc.squeeze()[ :, :3].squeeze()],axis=0))

        # vis_pc(np.concatenate([batch_points[:, :3].squeeze(),obj_pc.squeeze()[ :, :3].squeeze()],axis=0))



        # print(obj_pc.shape)
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
