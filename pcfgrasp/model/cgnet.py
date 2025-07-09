import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import mesh_utils
from utils.pointnet2_utils import PointNetSetAbstractionMsg, PointNetSetAbstraction, PointNetFeaturePropagation
from utils.grasp_utils import build_6d_grasp, get_bin_vals

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np

def plot(data, label, title):
    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)

    fig = plt.figure()
    ax = plt.subplot(111)
    for i in range(data.shape[0]):
        plt.text(data[i, 0], data[i, 1], str(label[i]),
                 color=plt.cm.Set1(label[i] / 10.),
                 fontdict={'weight': 'bold', 'size': 9})
    plt.xticks([])
    plt.yticks([])
    plt.title(title)
    return fig



class CGNet(nn.Module):
    def __init__(self, global_config):
        super(CGNet,self).__init__()

        self.global_config = global_config
        self.model_config = global_config['MODEL']

        self.sa1 = PointNetSetAbstractionMsg(npoint=1024, radius_list=[0.02, 0.04, 0.08], nsample_list=[32, 64, 128], in_channel=0, mlp_list=[[32, 32, 64], [64, 64, 128], [64, 96, 128]])
        self.sa2 = PointNetSetAbstractionMsg(npoint=512, radius_list=[0.04, 0.08, 0.16], nsample_list=[64, 64, 128], in_channel=64+128+128, mlp_list=[[64, 64, 128], [128, 128, 256], [128, 128, 256]])
        self.sa3 = PointNetSetAbstractionMsg(npoint=128, radius_list=[0.08, 0.16, 0.32], nsample_list=[64, 64, 128], in_channel=128+256+256, mlp_list=[[64, 64, 128], [128, 128, 256], [128, 128, 256]])

        self.sa4 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=128+256+256+3, mlp=self.model_config['pointnet_sa_module']['mlp'], group_all=self.model_config['pointnet_sa_module']['group_all'])
        self.fp3 = PointNetFeaturePropagation(1664, [256, 256]) #1024+640 对应的是l2points和l3points的通道数目
        self.fp2 = PointNetFeaturePropagation(896, [256, 128]) #512+64=576
        self.fp1 = PointNetFeaturePropagation(448, [128, 128, 128]) #64+64=128

        self.layer1 = nn.Sequential(
            nn.Conv1d(128,128,1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.7),
            nn.Conv1d(128,3,1)
        )

        self.layer2 = nn.Sequential(
            nn.Conv1d(128,128,1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.7),
            nn.Conv1d(128,3,1)
        )

        self.layer3 = nn.Sequential(
            nn.Conv1d(128,128,1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Conv1d(128,1,1)
        )

        self.layer4 = nn.Sequential(
            nn.Conv1d(128,128,1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128,10,1)
        )

        # self.layer5 = nn.Sequential(
        #     nn.Conv1d(128,128,1),
        #     nn.BatchNorm1d(128),
        #     nn.ReLU(),
        #     nn.Conv1d(128,1,1)
        # )
        # self.layer6 = nn.Sequential(
        #     nn.Conv1d(128,128,1),
        #     nn.BatchNorm1d(128),
        #     nn.ReLU(),
        #     nn.Conv1d(128,1,1)
        # )

    def forward(self, points):
        """
        Input:
            points: [b,n,c]姿态估计产生的500个点
        
        Output:
            end_points: {dict}
        """
        # print(points.shape)
        data_config = self.global_config['DATA']

        input_normals = data_config['input_normals']
        
        l0_points = points[:, :, 3:] if input_normals else None
        l0_xyz = points[:, :, :3]

        l0_xyz = l0_xyz.permute(0,2,1)
        # print(l0_xyz.shape)

        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        # print(l1_xyz.shape, l1_points.shape)  torch.Size([1, 3, 2048]) torch.Size([1, 320, 2048])
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        # print(l2_xyz.shape, l2_points.shape) torch.Size([1, 3, 512]) torch.Size([1, 640, 512])
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        # print(l3_xyz.shape, l3_points.shape) torch.Size([1, 3, 128]) torch.Size([1, 640, 128])


        l4_xyz, l4_points = self.sa4(l3_xyz, l3_points)
        # print(l4_xyz.shape, l4_points.shape)  torch.Size([1, 3, 1]) torch.Size([1, 1024, 1])
        l3_points = self.fp3(l3_xyz, l4_xyz, l3_points, l4_points)
        # print(l3_points.shape)  torch.Size([1, 256, 128])
        l2_points = self.fp2(l2_xyz, l3_xyz, l2_points, l3_points)
        # print(l2_points.shape)  torch.Size([1, 128, 512])
        l1_points = self.fp1(l1_xyz, l2_xyz, l1_points, l2_points)
        # print(l1_points.shape)  torch.Size([1, 128, 2048])

        l0_points = l1_points
        #####################################
        # a = np.array(np.arange(1024.))
        # label = a.tolist()
        
        # tsne = TSNE(n_components=2, init='pca')
        # result = tsne.fit_transform(l0_points.squeeze())
        # fig = plot(result, label, 'tsne')
        # # print(fig)

        # plt.show()
        #######################################
        # print('--------{}'.format(l0_points.shape))
        pred_points = l1_xyz.permute(0,2,1)
        grasp_dir_head = self.layer1(l0_points)
        grasp_dir_head_normed = F.normalize(grasp_dir_head, dim=1)

        approach_dir_head = self.layer2(l0_points)
        approach_dir_orthog = approach_dir_head - torch.sum(torch.mul(grasp_dir_head_normed, approach_dir_head), dim=1, keepdim=True) * grasp_dir_head_normed
        approach_dir_head_orthog = F.normalize(approach_dir_orthog, dim=1)

        binary_score_head = self.layer3(l0_points)

        grasp_offset_head = self.layer4(l0_points)

        # a_score = self.layer5(l0_points)
        # a_score = torch.sigmoid(a_score)
        # # print(c_score.shape)  torch.Size([1, 1, 2048]) 取其中最高的作为自监督的参数来提取

        # a_score = a_score.permute(0,2,1)

        # b_score = self.layer6(l0_points)
        # b_score = torch.sigmoid(b_score)
        # b_score = b_score.permute(0,2,1)

        # c_score = self.layer6(l0_points)
        # c_score = torch.sigmoid(c_score)
        # c_score = c_score.permute(0,2,1)

        grasp_dir_head_normed = grasp_dir_head_normed.permute(0,2,1)
        approach_dir_head_orthog = approach_dir_head_orthog.permute(0,2,1)
        binary_score_head = binary_score_head.permute(0,2,1)
        grasp_offset_head = grasp_offset_head.permute(0,2,1)

        m = nn.Sigmoid()

        #转换
        grasp_dir_head = grasp_dir_head_normed
        approach_dir_head = approach_dir_head_orthog

        end_points = {}

        end_points['grasp_dir_head'] = grasp_dir_head_normed
        end_points['approach_dir_head'] = approach_dir_head_orthog
        end_points['binary_score_head'] = binary_score_head
        end_points['binary_score_pred'] = m(binary_score_head)
        end_points['grasp_offset_head'] = grasp_offset_head
        end_points['grasp_offset_pred'] = m(grasp_offset_head) if self.model_config['bin_offsets'] else grasp_offset_head
        # end_points['a_score'] = a_score
        # end_points['b_score'] = b_score
        # end_points['c_score'] = c_score
        # print(grasp_dir_head_normed.shape)
        # print(approach_dir_head_orthog.shape)
        # print(binary_score_head.shape)
        # print(grasp_offset_head.shape)
        # torch.Size([1, 2048, 3])
        # torch.Size([1, 2048, 3])
        # torch.Size([1, 2048, 1])
        # torch.Size([1, 2048, 10])
        # print(torch.sum(binary_score_head.detach(), dim=1))
        end_points['pred_points'] = pred_points

        return end_points


class CGLoss(nn.Module):
    def __init__(self, global_config):
        super(CGLoss, self).__init__()
        self.global_config = global_config

    def forward(self, end_points, dir_labels_pc_cam, offset_labels_pc, grasp_success_labels_pc, approach_labels_pc_cam):
        """
        Input:
            end_points: {dict} predict parameters
            end_points['points'] [B,N,C] {default: N=1024}
            end_points['grasp_offset_head'] [B, N, C] {default: 1, 1024, 10}

        Output:
            total_loss
        """
        # print(torch.sum(grasp_success_labels_pc))

        grasp_dir_head = end_points['grasp_dir_head']
        approach_dir_head = end_points['approach_dir_head']
        grasp_offset_head = end_points['grasp_offset_head']
        # c_score = end_points['c_score']
        bin_weights = self.global_config['DATA']['labels']['bin_weights']
        torch_bin_weights = torch.tensor(bin_weights)

        global_config = self.global_config
        min_geom_loss_divisor = torch.tensor(
            float(global_config['LOSS']['min_geom_loss_divisor'])) if 'min_geom_loss_divisor' in global_config[
            'LOSS'] else torch.tensor(1.)
        pos_grasps_in_view = torch.maximum(torch.sum(grasp_success_labels_pc, dim=1), min_geom_loss_divisor)

        pointclouds_pl = end_points['pred_points']
        ### ADS Gripper PC Loss
        if global_config['MODEL']['bin_offsets']:
            thickness_pred = get_bin_vals(global_config)[torch.argmax(grasp_offset_head, dim=2)]
            thickness_gt = get_bin_vals(global_config)[torch.argmax(offset_labels_pc, dim=2)]
        else:
            thickness_pred = grasp_offset_head[:, :, 0]
            thickness_gt = offset_labels_pc[:, :, 0]
        
        # print(thickness_pred)
        pred_grasps = build_6d_grasp(approach_dir_head, grasp_dir_head, pointclouds_pl, thickness_pred)  # b x num_point x 4 x 4
        gt_grasps_proj = build_6d_grasp(approach_labels_pc_cam, dir_labels_pc_cam, pointclouds_pl, thickness_gt)  # b x num_point x 4 x 4

        pos_gt_grasps_proj = torch.where(
            torch.broadcast_to(torch.unsqueeze(torch.unsqueeze(grasp_success_labels_pc.type(torch.bool), dim=2), dim=3),
                            gt_grasps_proj.shape), gt_grasps_proj, torch.ones_like(gt_grasps_proj) * 100000)
        # pos_gt_grasps_proj = tf.reshape(pos_gt_grasps_proj, (global_config['OPTIMIZER']['batch_size'], -1, 4, 4))

        gripper = mesh_utils.create_gripper('panda')
        gripper_control_points = gripper.get_control_point_tensor(global_config['OPTIMIZER']['batch_size'])  # b x 5 x 3
        sym_gripper_control_points = gripper.get_control_point_tensor(global_config['OPTIMIZER']['batch_size'], symmetric=True)

        gripper_control_points_homog = torch.cat([gripper_control_points, torch.ones((global_config['OPTIMIZER']['batch_size'], gripper_control_points.shape[1], 1))], dim=2)  # b x 5 x 4
        sym_gripper_control_points_homog = torch.cat([sym_gripper_control_points, torch.ones((global_config['OPTIMIZER']['batch_size'], gripper_control_points.shape[1], 1))], dim=2)  # b x 5 x 4

        # only use per point pred grasps but not per point gt grasps
        control_points = torch.unsqueeze(gripper_control_points_homog, dim=1).repeat(1, gt_grasps_proj.shape[1], 1, 1)  # b x num_point x 5 x 4
        # print(control_points)
        sym_control_points = torch.unsqueeze(sym_gripper_control_points_homog, dim=1).repeat(1, gt_grasps_proj.shape[1], 1, 1)  # b x num_point x 5 x 4
        # print(pred_grasps)
        pred_control_points = torch.matmul(control_points, torch.transpose(pred_grasps, dim0=2, dim1=3))[:, :, :, :3]  # b x num_point x 5 x 3
        pred_points = torch.matmul(control_points, torch.transpose(pred_grasps, dim0=2, dim1=3))
        # print(pred_points)
        ### Pred Grasp to GT Grasp ADD-S Loss
        gt_control_points = torch.matmul(control_points, torch.transpose(pos_gt_grasps_proj, dim0=2, dim1=3))[:, :, :, :3]  # b x num_pos_grasp_point x 5 x 3
        sym_gt_control_points = torch.matmul(sym_control_points, torch.transpose(pos_gt_grasps_proj, dim0=2, dim1=3))[:, :, :, :3] # b x num_pos_grasp_point x 5 x 3

        squared_add = torch.sum((torch.unsqueeze(pred_control_points, dim=2) - torch.unsqueeze(gt_control_points, dim=1)) ** 2, dim=(3, 4))  # b x num_point x num_pos_grasp_point x ( 5 x 3)
        sym_squared_add = torch.sum((torch.unsqueeze(pred_control_points, dim=2) - torch.unsqueeze(sym_gt_control_points, dim=1)) ** 2, dim=(3, 4))  # b x num_point x num_pos_grasp_point x ( 5 x 3)

        # symmetric ADD-S
        neg_squared_adds = -torch.cat([squared_add, sym_squared_add], dim=2)  # b x num_point x 2num_pos_grasp_point
        neg_squared_adds_k = torch.topk(neg_squared_adds, k=1, sorted=False)[0]  # b x num_point
        # If any pos grasp exists
        min_adds = torch.minimum(torch.sum(grasp_success_labels_pc, dim=1, keepdims=True),torch.ones_like(neg_squared_adds_k[:, :, 0])) * torch.sqrt(-neg_squared_adds_k[:, :, 0])  # tf.minimum(tf.sqrt(-neg_squared_adds_k), tf.ones_like(neg_squared_adds_k)) # b x num_point
        # print(min_adds)
        adds_loss = torch.mean(end_points['binary_score_pred'][:, :, 0] * min_adds)
        # print(adds_loss)
        # print(end_points['binary_score_head'].detach())

        ### GT Grasp to pred Grasp ADD-S Loss
        gt_control_points = torch.matmul(control_points, torch.transpose(gt_grasps_proj, dim0=2, dim1=3))[:, :, :, :3]  # b x num_pos_grasp_point x 5 x 3
        sym_gt_control_points = torch.matmul(sym_control_points, torch.transpose(gt_grasps_proj, dim0=2, dim1=3))[:, :, :, :3]  # b x num_pos_grasp_point x 5 x 3

        neg_squared_adds = -torch.sum((torch.unsqueeze(pred_control_points, dim=1) - torch.unsqueeze(gt_control_points, dim=2)) ** 2, dim=(3, 4))  # b x num_point x num_pos_grasp_point x ( 5 x 3)
        neg_squared_adds_sym = -torch.sum((torch.unsqueeze(pred_control_points, dim=1) - torch.unsqueeze(sym_gt_control_points, dim=2)) ** 2,dim=(3, 4))  # b x num_point x num_pos_grasp_point x ( 5 x 3)

        neg_squared_adds_k_gt2pred, pred_grasp_idcs = torch.topk(neg_squared_adds, k=1, sorted=False)  # b x num_pos_grasp_point
        neg_squared_adds_k_sym_gt2pred, pred_grasp_sym_idcs = torch.topk(neg_squared_adds_sym, k=1, sorted=False)  # b x num_pos_grasp_point
        pred_grasp_idcs_joined = torch.where(neg_squared_adds_k_gt2pred < neg_squared_adds_k_sym_gt2pred, pred_grasp_sym_idcs, pred_grasp_idcs)
        min_adds_gt2pred = torch.minimum(-neg_squared_adds_k_gt2pred, -neg_squared_adds_k_sym_gt2pred)  # b x num_pos_grasp_point x 1
        # min_adds_gt2pred = tf.math.exp(-min_adds_gt2pred)
        masked_min_adds_gt2pred = torch.multiply(min_adds_gt2pred[:, :, 0], grasp_success_labels_pc)
        # print(pred_grasp_idcs_joined.shape)
        batch_idcs = torch.meshgrid(torch.arange(pred_grasp_idcs_joined.shape[0]), torch.arange(pred_grasp_idcs_joined.shape[1])
                                    # torch.arange(pred_grasp_idcs_joined.shape[0]), indexing="xy")
                                    )
        # print(batch_idcs[0],batch_idcs[1])
        gather_idcs = torch.stack((batch_idcs[0], pred_grasp_idcs_joined[:, :, 0]), dim=2)
        # change to here

        nearest_pred_grasp_confidence = end_points['binary_score_pred'][:, :, 0][gather_idcs[:, :, 0], gather_idcs[:, :, 1]]
        adds_loss_gt2pred = torch.mean(torch.sum(nearest_pred_grasp_confidence * masked_min_adds_gt2pred, dim=1) / pos_grasps_in_view)

        ### Grasp baseline Loss
        # cosine_distance = torch.tensor(1.) - torch.sum(torch.multiply(end_points['b_score'],torch.multiply(dir_labels_pc_cam, grasp_dir_head)), dim=2)
        cosine_distance = torch.tensor(1.) - torch.sum(torch.multiply(dir_labels_pc_cam, grasp_dir_head), dim=2)

        # only pass loss where we have labeled contacts near pc points
        masked_cosine_loss = torch.multiply(cosine_distance, grasp_success_labels_pc)
        dir_cosine_loss = torch.mean(torch.sum(masked_cosine_loss, dim=1) / pos_grasps_in_view)

        ### Grasp Approach Loss
        approach_labels_orthog = F.normalize(approach_labels_pc_cam - torch.sum(torch.multiply(grasp_dir_head, approach_labels_pc_cam), dim=2,keepdims=True) * grasp_dir_head, dim=2)
        cosine_distance_approach = torch.tensor(1.) - torch.sum(torch.multiply(approach_labels_orthog, approach_dir_head),dim=2)
        # cosine_distance_approach = torch.tensor(1.) - torch.sum(torch.multiply(end_points['a_score'], torch.multiply(approach_labels_orthog, approach_dir_head)),dim=2)

        masked_approach_loss = torch.multiply(cosine_distance_approach, grasp_success_labels_pc)
        approach_cosine_loss = torch.mean(torch.sum(masked_approach_loss, dim=1) / pos_grasps_in_view)

        ### Grasp Offset/Thickness Loss
        # print(f'---------------{torch.sum(offset_labels_pc, dim=1)}-------------------') tensor([[1015.,    9.,    0.,    0.,    0.,    0.,    0.,    0.,    0.,    0.]])
        if global_config['MODEL']['bin_offsets']:
            if global_config['LOSS']['offset_loss_type'] == 'softmax_cross_entropy':
                # offset_loss_old = tf.losses.softmax_cross_entropy(tf.constant(offset_labels_pc.detach().numpy()), tf.constant(grasp_offset_head.detach().numpy()))
                offset_loss = torch.zeros(grasp_offset_head.shape[0], grasp_offset_head.shape[1])
                for batch in range(offset_loss.shape[0]):
                    offset_loss[batch] = F.cross_entropy(grasp_offset_head[batch],
                                                        torch.argmax(offset_labels_pc[batch], dim=1), reduction='none')
                offset_loss = torch.mean(offset_loss)
            else:
                offset_loss = offset_labels_pc * -torch.log(torch.sigmoid(grasp_offset_head)) + (
                            1 - offset_labels_pc) * -torch.log(1 - torch.sigmoid(grasp_offset_head))
                # offset_loss_old = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.constant(offset_labels_pc.detach().numpy()), logits=tf.constant(grasp_offset_head.detach().numpy()))

                if 'too_small_offset_pred_bin_factor' in global_config['LOSS'] and global_config['LOSS']['too_small_offset_pred_bin_factor']:
                    too_small_offset_pred_bin_factor = torch.tensor(global_config['LOSS']['too_small_offset_pred_bin_factor'], torch.float32)
                    # collision_weight = tf.math.cumsum(offset_labels_pc, axis=2,
                    #                                   reverse=True) * too_small_offset_pred_bin_factor + torch.constant(1.)
                    collision_weight = (offset_labels_pc + torch.sum(offset_labels_pc, dim=2, keepdims=True) - torch.cumsum(offset_labels_pc, dim=2)) * too_small_offset_pred_bin_factor + torch.constant(1.)
                    offset_loss = torch.multiply(collision_weight, offset_loss)

                offset_loss = torch.mean(torch.multiply(torch.reshape(torch_bin_weights, (1, 1, -1)), offset_loss), axis=2)
        else:
            offset_loss = (grasp_offset_head[:, :, 0] - offset_labels_pc[:, :, 0]) ** 2
        # masked_offset_loss = torch.multiply(end_points['c_score'],torch.multiply(offset_loss, grasp_success_labels_pc))
        masked_offset_loss = torch.multiply(offset_loss, grasp_success_labels_pc)
        # print(masked_offset_loss)
        offset_loss = torch.mean(torch.sum(masked_offset_loss, dim=1) / pos_grasps_in_view)

        ### Grasp Confidence Loss
        # bin_ce_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.expand_dims(grasp_success_labels_pc, axis=2),
        #                                                       logits=end_points['binary_seg_head'])

        bin_ce_loss = torch.unsqueeze(grasp_success_labels_pc, dim=2) * -torch.log(
            torch.sigmoid(end_points['binary_score_head'])) + (
                                1 - torch.unsqueeze(grasp_success_labels_pc, dim=2)) * -torch.log(
            1 - torch.sigmoid(end_points['binary_score_head']))
        if 'topk_confidence' in global_config['LOSS'] and global_config['LOSS']['topk_confidence']:
            bin_ce_loss, _ = torch.topk(torch.squeeze(bin_ce_loss), k=global_config['LOSS']['topk_confidence'])
        bin_ce_loss = torch.mean(bin_ce_loss)

        loss_dict = {}
        loss_dict['dir_cosine_loss'] = dir_cosine_loss
        loss_dict['app_cosine_loss'] = approach_cosine_loss
        loss_dict['offset_loss'] = offset_loss
        loss_dict['score_loss'] = bin_ce_loss
        loss_dict['adds_loss'] = adds_loss
        loss_dict['adds_loss_gt2pred'] = adds_loss_gt2pred

        total_loss = 0
        if self.global_config['MODEL']['pred_contact_base']:  #false
            total_loss += self.global_config['OPTIMIZER']['dir_cosine_loss_weight'] * dir_cosine_loss
        if self.global_config['MODEL']['pred_contact_success']:
            total_loss += self.global_config['OPTIMIZER']['score_ce_loss_weight'] * bin_ce_loss
        if self.global_config['MODEL']['pred_contact_offset']:
            total_loss += self.global_config['OPTIMIZER']['offset_loss_weight'] * offset_loss
        if self.global_config['MODEL']['pred_contact_approach']: #false
            total_loss += self.global_config['OPTIMIZER']['approach_cosine_loss_weight'] * approach_cosine_loss
        if self.global_config['MODEL']['pred_grasps_adds']:
            total_loss += self.global_config['OPTIMIZER']['adds_loss_weight'] * adds_loss
        if self.global_config['MODEL']['pred_grasps_adds_gt2pred']: #false
            total_loss += self.global_config['OPTIMIZER']['adds_gt2pred_loss_weight'] * adds_loss_gt2pred
        # print(loss_dict)
        return total_loss, loss_dict

