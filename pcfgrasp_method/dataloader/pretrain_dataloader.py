from torch.utils.data import Dataset
from utils.data import center_pc_convert_cam, center_pc_gt_convert_cam
from run_utils.logger import print_log
from utils.data import vis_pc

class CompleteDataset(Dataset):
    def __init__(self, global_config, pcreader, contact_infos, split='train', logger = None):

        super().__init__()

        self.pt = {}
        self.pcreader = pcreader

        num_test_samples = global_config['DATA']['num_test_scenes']
        num_train_samples = len(contact_infos)-num_test_samples

        scene_idxs = []

        if split == 'train':
            for scene_idx in range(num_train_samples):
                scene_idxs.append(scene_idx)
        else:
            for scene_idx in range(num_test_samples):
                scene_idxs.append(scene_idx + num_train_samples)

        self.scene_idxs = scene_idxs

        print_log("Totally {} samples in {} set.".format(len(self.scene_idxs), split), logger=logger)


    def __getitem__(self, idx):
        """
        parameters:
            idx is the batch_idx of training or testing
        """

        batch_points_raw, cam_poses, _, obj_pc = self.pcreader.get_scene_batch(scene_idx=self.scene_idxs[idx]) #BNC

        batch_points, _ = center_pc_convert_cam(cam_poses, batch_points_raw) #1024 x 3
        obj_pc, _ = center_pc_gt_convert_cam(cam_poses, batch_points_raw, obj_pc) #2048 x 3
        
        return batch_points, obj_pc[..., :3]
    

    def __len__(self):
        return len(self.scene_idxs)
