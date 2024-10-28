'''
gaussian_loss.py is used to compute the loss of gs
'''

import torch
import torch.nn as nn
import cv2
import numpy as np
import os
import imageio
import lpips
loss_fn_vgg = lpips.LPIPS(net='vgg').to(torch.device('cuda', torch.cuda.current_device()))
from random import randint
from core.gaussian.utils.loss_utils import l1_loss, l2_loss, ssim
from core.gaussian.gaussian_renderer import render
from core.gaussian.utils.graphics_utils import focal2fov, fov2focal
from typing import NamedTuple
from PIL import Image
from smpl.smpl_numpy import SMPL
from smpl.smpl_torch_batch import SMPLModel

class GS3DLoss(nn.Module):
    def __init__(self,
                 models=None,
                 pose_embeddings=None,
                 gaussians=None,
                 pipe=None,
                 background=None,
                 **kwargs
                 ):

        super(GS3DLoss, self).__init__()

        self.models = models
        self.pose_embeddings = pose_embeddings
        self.gaussians = gaussians
        self.pipe = pipe
        self.background = background
        self.scene = self.get_scene(self)
        self.viewpoints_stack = self.scene.getTrainCameras().copy()

    def forward(self, iteration):
        # 随机取一帧训练
        if not self.viewpoints_stack:
            self.viewpoints_stack = self.scene.getTrainCameras().copy()
        viewpoints_cam = self.viewpoint_stack.pop(randint(0, len(self.viewpoint_stack)-1))

        # 渲染
        render_pkg = render(self.viewpoint_cam, self.gaussians, self.pipe, self.background)
        image, alpha, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["render_alpha"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        # 求loss
        gt_image = self.viewpoint_cam.original_image.cuda()
        bkgd_mask = self.viewpoint_cam.bkgd_mask.cuda()
        bound_mask = self.viewpoint_cam.bound_mask.cuda()
        # l1 loss
        Ll1 = l1_loss(image.permute(1,2,0)[bound_mask[0]==1], gt_image.permute(1,2,0)[bound_mask[0]==1])
        # mask loss
        mask_loss = l2_loss(alpha[bound_mask==1], bkgd_mask[bound_mask==1])
        x, y, w, h = cv2.boundingRect(bound_mask[0].cpu().numpy().astype(np.uint8))
        img_pred = image[:, y:y + h, x:x + w].unsqueeze(0)
        img_gt = gt_image[:, y:y + h, x:x + w].unsqueeze(0)
        # ssim loss
        ssim_loss = ssim(img_pred, img_gt)
        # lipis loss
        lpips_loss = loss_fn_vgg(img_pred, img_gt).reshape(-1)

        loss = Ll1 + 0.1 * mask_loss + 0.01 * (1.0 - ssim_loss) + 0.01 * lpips_loss
        loss_dict = {}

        return loss, loss_dict

    def get_scene(self, output_view=[0]):
        # set间隔采样pose数量
        pose_start = 0
        pose_interval = 3
        pose_num = 100

        # 提取ims，cams信息
        path = 'data/3DOH'
        ann_file = os.path.join(path, 'annots.npy')
        annots = np.load(ann_file, allow_pickle=True).item()
        cams = annots['cams']
        ims = np.array([
            np.array(ims_data['ims'])[output_view]
            for ims_data in annots['ims'][pose_start:pose_start + pose_num * pose_interval][::pose_interval]
        ])
        cam_inds = np.array([
            np.arange(len(ims_data['ims']))[output_view]
            for ims_data in annots['ims'][pose_start:pose_start + pose_num * pose_interval][::pose_interval]
        ])

        # 标准空间smpl
        smpl_model = SMPL(sex='neutral', model_dir='assets/SMPL_NEUTRAL_renderpeople.pkl')
        big_pose_smpl_param = {}
        big_pose_smpl_param['R'] = np.eye(3).astype(np.float32)
        big_pose_smpl_param['Th'] = np.zeros((1, 3)).astype(np.float32)
        big_pose_smpl_param['shapes'] = np.zeros((1, 10)).astype(np.float32)
        big_pose_smpl_param['poses'] = np.zeros((1, 72)).astype(np.float32)
        big_pose_smpl_param['poses'][0, 5] = 45 / 180 * np.array(np.pi)
        big_pose_smpl_param['poses'][0, 8] = -45 / 180 * np.array(np.pi)
        big_pose_smpl_param['poses'][0, 23] = -30 / 180 * np.array(np.pi)
        big_pose_smpl_param['poses'][0, 26] = 30 / 180 * np.array(np.pi)
        big_pose_xyz, _ = smpl_model(big_pose_smpl_param['poses'], big_pose_smpl_param['shapes'].reshape(-1))
        big_pose_xyz = (np.matmul(big_pose_xyz, big_pose_smpl_param['R'].transpose()) + big_pose_smpl_param['Th']).astype(np.float32)
        # 获取bounds
        big_pose_min_xyz = np.min(big_pose_xyz, axis=0)
        big_pose_max_xyz = np.max(big_pose_xyz, axis=0)
        big_pose_min_xyz -= 0.05
        big_pose_max_xyz += 0.05
        big_pose_world_bound = np.stack([big_pose_min_xyz, big_pose_max_xyz], axis=0)

        # 获取每帧每视角的人物和相机参数
        cam_infos = []
        idx = 0
        for pose_index in range(pose_num):
            for view_index in range(len(output_view)):

                # 加载images
                image_path = os.path.join(path, ims[pose_index][view_index].replace('\\', '/'))
                image_name = ims[pose_index][view_index].split('.')[0]
                image = np.array(imageio.imread(image_path).astype(np.float32) / 255.)

                # 加载mask
                msk_path = image_path.replace('images', 'mask').replace('jpg', 'png')
                msk = imageio.imread(msk_path)
                msk = (msk != 0).astype(np.uint8)

                # 加载K, D, R, T
                cam_ind = cam_inds[pose_index][view_index]
                K = np.array(cams['K'][cam_ind])
                D = np.array(cams['D'][cam_ind])
                R = np.array(cams['R'][cam_ind])
                T = np.array(cams['T'][cam_ind])
                image = cv2.undistort(image, K, D)
                msk = cv2.undistort(msk, K, D)

                # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
                w2c = np.eye(4)
                w2c[:3, :3] = R
                w2c[:3, 3:4] = T

                # get the world-to-camera transform and set R, T
                R = np.transpose(w2c[:3, :3])  # R is stored transposed due to 'glm' in CUDA code
                T = w2c[:3, 3]

                # numpy -> Image
                image = Image.fromarray(np.array(image * 255.0, dtype=np.byte), "RGB")

                # 加载焦距
                focalX = K[0, 0]
                focalY = K[1, 1]
                FovX = focal2fov(focalX, image.size[0])
                FovY = focal2fov(focalY, image.size[1])

                # 获取smpl参数和顶点
                for idx, (model, pose_embedding) in enumerate(zip(self.models, self.pose_embeddings)):
                    body_pose = vposer.decode(pose_embedding, t=frames_seq).view(frames_seq, -1)
                    # 手和脚的点不准，不使用
                    body_pose[:, -6:] = 0.
                    body_pose = body_pose.detach().cpu().numpy()
                    orient = np.array(model.global_orient.detach().cpu().numpy())
                    pose = np.hstack((orient, body_pose)).astype(np.float32).reshape(-1, 72)
                    shape = model.betas.detach().cpu().numpy().astype(np.float32).reshape(-1, 10)
                    Rh = model.global_orient.detach().cpu().numpy().astype(np.float32).reshape(-1, 3)
                    Th = model.transl.detach().cpu().numpy().astype(np.float32).reshape(-1, 3)

                    # 获取每帧的smpl参数
                    smpl_param = {}
                    smpl_param['R'] = cv2.Rodrigues(Rh)[0].astype(np.float32)
                    smpl_param['Th'] = Th
                    smpl_param['shapes'] = shape
                    smpl_param['poses'] = pose

                    # 获取每帧顶点
                    s = torch.from_numpy(smpl_param['shapes'].astype(np.float32).reshape(-1, 10))
                    p = torch.from_numpy(smpl_param['poses'].astype(np.float32).reshape(-1, 72))
                    t = torch.from_numpy(smpl_param['Th'].astype(np.float32).reshape(-1, 3))
                    smpl = SMPLModel(device=torch.device('cpu'), model_path='assets/SMPL_NEUTRAL.pkl')
                    v, _ = smpl(s, p, t)
                    xyz = v.detach().cpu().numpy().reshape(-1, 3).astype(np.float32)

                    # 获取bounds, bound_mask, bkgs_mask
                    min_xyz = np.min(xyz, axis=0)
                    max_xyz = np.max(xyz, axis=0)
                    min_xyz -= 0.05
                    max_xyz += 0.05
                    world_bound = np.stack([min_xyz, max_xyz], axis=0)
                    bound_mask = get_bound_2d_mask(world_bound, K, w2c[:3], image.size[1], image.size[0])
                    bound_mask = Image.fromarray(np.array(bound_mask * 255.0, dtype=np.uint8))
                    bkgd_mask = Image.fromarray(np.array(msk * 255.0, dtype=np.uint8)).convert('L')

                    cam_infos.append(CameraInfo(uid=idx, pose_id=pose_index, R=R, T=T, K=K, FovY=FovY, FovX=FovX, image=image,
                                    image_path=image_path, image_name=image_name, bkgd_mask=bkgd_mask,
                                    bound_mask=bound_mask, width=image.size[0], height=image.size[1],
                                    smpl_param=smpl_param, world_vertex=xyz, world_bound=world_bound,
                                    big_pose_smpl_param=big_pose_smpl_param, big_pose_world_vertex=big_pose_xyz,
                                    big_pose_world_bound=big_pose_world_bound))

                    idx += 1
        return cam_infos

class CameraInfo(NamedTuple):
    uid: int
    pose_id: int
    R: np.array
    T: np.array
    K: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    image_path: str
    image_name: str
    bkgd_mask: np.array
    bound_mask: np.array
    width: int
    height: int
    smpl_param: dict
    world_vertex: np.array
    world_bound: np.array
    big_pose_smpl_param: dict
    big_pose_world_vertex: np.array
    big_pose_world_bound: np.array

def get_bound_corners(bounds):
    min_x, min_y, min_z = bounds[0]
    max_x, max_y, max_z = bounds[1]
    corners_3d = np.array([
        [min_x, min_y, min_z],
        [min_x, min_y, max_z],
        [min_x, max_y, min_z],
        [min_x, max_y, max_z],
        [max_x, min_y, min_z],
        [max_x, min_y, max_z],
        [max_x, max_y, min_z],
        [max_x, max_y, max_z],
    ])
    return corners_3d

def project(xyz, K, RT):
    """
    xyz: [N, 3]
    K: [3, 3]
    RT: [3, 4]
    """
    xyz = np.dot(xyz, RT[:, :3].T) + RT[:, 3]
    xyz = np.dot(xyz, K.T)
    xy = xyz[:, :2] / xyz[:, 2:]
    return xy

def get_bound_2d_mask(bounds, K, pose, H, W):
    corners_3d = get_bound_corners(bounds)
    corners_2d = project(corners_3d, K, pose)
    corners_2d = np.round(corners_2d).astype(int)
    mask = np.zeros((H, W), dtype=np.uint8)
    cv2.fillPoly(mask, [corners_2d[[0, 1, 3, 2, 0]]], 1)
    cv2.fillPoly(mask, [corners_2d[[4, 5, 7, 6, 4]]], 1) # 4,5,7,6,4
    cv2.fillPoly(mask, [corners_2d[[0, 1, 5, 4, 0]]], 1)
    cv2.fillPoly(mask, [corners_2d[[2, 3, 7, 6, 2]]], 1)
    cv2.fillPoly(mask, [corners_2d[[0, 2, 6, 4, 0]]], 1)
    cv2.fillPoly(mask, [corners_2d[[1, 3, 7, 5, 1]]], 1)
    return mask
