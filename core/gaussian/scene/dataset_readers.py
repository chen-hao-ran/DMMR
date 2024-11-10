import os
import sys
from PIL import Image
from typing import NamedTuple
from core.gaussian.utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
import torch
import json
import imageio
import cv2
import random
import time
from pathlib import Path
from plyfile import PlyData, PlyElement
from core.gaussian.utils.sh_utils import SH2RGB
from core.gaussian.scene.gaussian_model import BasicPointCloud
from core.gaussian.smpl.smpl_torch_batch import SMPLModel

from core.gaussian.smpl.smpl_numpy import SMPL

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

class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str

def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}

def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)

def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

def prepare_smpl_params(smpl_path, pose_index):
    params_ori = dict(np.load(smpl_path, allow_pickle=True))['smpl'].item()
    params = {}
    params['shapes'] = np.array(params_ori['betas']).astype(np.float32)
    params['poses'] = np.zeros((1,72)).astype(np.float32)
    params['poses'][:, :3] = np.array(params_ori['global_orient'][pose_index]).astype(np.float32)
    params['poses'][:, 3:] = np.array(params_ori['body_pose'][pose_index]).astype(np.float32)
    params['R'] = np.eye(3).astype(np.float32)
    params['Th'] = np.array(params_ori['transl'][pose_index:pose_index+1]).astype(np.float32)
    return params

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
    xyz = np.dot(xyz, RT[:, :3].T) + RT[:, 3:].T
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

def get_mask(path, index, view_index, ims):
    msk_path = os.path.join(path, 'mask_cihp',
                            ims[index][view_index])[:-4] + '.png'
    msk_cihp = imageio.imread(msk_path)
    msk_cihp = (msk_cihp != 0).astype(np.uint8)
    msk = msk_cihp.copy()

    return msk, msk_cihp

def readInfo(path, white_background, output_path, eval, setting, dataset_obj, mode):
    train_view = [0, 1, 2]
    test_view = [0]

    print("Reading Training Transforms")
    train_cam_infos = readCameras(path, train_view, white_background, setting, dataset_obj, mode, split='train')
    print("Reading Test Transforms")
    test_cam_infos = readCameras(path, test_view, white_background, setting, dataset_obj, mode, split='test')

    # if not eval:
    #     train_cam_infos.extend(test_cam_infos)
    #     test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)
    if len(train_view) == 1:
        nerf_normalization['radius'] = 1

    ply_path = os.path.join('output', output_path, "points3d.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 6890  # 100_000
        print(f"Generating random point cloud ({num_pts})...")

        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = train_cam_infos[0].big_pose_world_vertex

        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

def readCameras(path, output_view, white_background, setting, dataset_obj, mode, image_scaling=1., split='train', novel_view_vis=False):
    cam_infos = []

    pose_start = 0
    if split == 'train':
        pose_interval = 3
        pose_num = 100
    elif split == 'test':
        if mode == 1:
            pose_interval = 1
            pose_num = 1
        else:
            pose_interval = 1
            pose_num = 100

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

    if 'CoreView_313' in path or 'CoreView_315' in path:
        for i in range(ims.shape[0]):
            ims[i] = [x.split('/')[0] + '/' + x.split('/')[1].split('_')[4] + '.jpg' for x in ims[i]]

    if mode == 0:
        smpl_model = SMPL(sex='neutral', model_dir='')

        # SMPL in canonical space
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
        big_pose_xyz = (np.matmul(big_pose_xyz, big_pose_smpl_param['R'].transpose()) + big_pose_smpl_param['Th']).astype(
            np.float32)

        # obtain the original bounds for point sampling
        big_pose_min_xyz = np.min(big_pose_xyz, axis=0)
        big_pose_max_xyz = np.max(big_pose_xyz, axis=0)
        big_pose_min_xyz -= 0.05
        big_pose_max_xyz += 0.05
        big_pose_world_bound = np.stack([big_pose_min_xyz, big_pose_max_xyz], axis=0)

        # load smpl params
        model = setting['model'][0]
        pose_embedding = setting['pose_embedding'][0]
        vposer = setting['vposer']
        frames_seq = dataset_obj.frames
        body_pose = vposer.decode(pose_embedding, t=frames_seq).view(frames_seq, -1)
        body_pose[:, -6:] = 0.
        body_pose = body_pose.detach().cpu().numpy()
        orient = np.array(model.global_orient.detach().cpu().numpy())
        poses = np.hstack((orient, body_pose)).reshape(-1, 72)
        shapes = model.betas.detach().cpu().numpy().astype(np.float32).reshape(-1, 10)
        Ths = model.transl.detach().cpu().numpy().astype(np.float32).reshape(-1, 3)
        Rhs = model.global_orient.detach().cpu().numpy().astype(np.float32).reshape(-1, 3)
        smpl_chomp = SMPLModel(device=torch.device('cpu'), model_path='assets/SMPL_NEUTRAL.pkl')

        idx = 0
        for pose_index in range(pose_num):
            for view_index in range(len(output_view)):
                # Load image, mask, K, D, R, T
                image_path = os.path.join(path, ims[pose_index][view_index].replace('\\', '/'))
                image_name = ims[pose_index][view_index].split('.')[0]
                image = np.array(imageio.imread(image_path).astype(np.float32) / 255.)

                msk_path = image_path.replace('images', 'mask').replace('jpg', 'png')
                msk = imageio.imread(msk_path)
                msk = (msk != 0).astype(np.uint8)

                if not novel_view_vis:
                    cam_ind = cam_inds[pose_index][view_index]
                    K = np.array(cams['K'][cam_ind])
                    D = np.array(cams['D'][cam_ind])
                    R = np.array(cams['R'][cam_ind])
                    T = np.array(cams['T'][cam_ind])

                    image = cv2.undistort(image, K, D)
                    msk = cv2.undistort(msk, K, D)

                image[msk == 0] = 1 if white_background else 0

                # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
                w2c = np.eye(4)
                w2c[:3, :3] = R
                w2c[:3, 3:4] = T

                # get the world-to-camera transform and set R, T
                R = np.transpose(w2c[:3, :3])  # R is stored transposed due to 'glm' in CUDA code
                T = w2c[:3, 3]

                # Reduce the image resolution by ratio, then remove the back ground
                ratio = image_scaling
                if ratio != 1.:
                    H, W = int(image.shape[0] * ratio), int(image.shape[1] * ratio)
                    image = cv2.resize(image, (W, H), interpolation=cv2.INTER_AREA)
                    msk = cv2.resize(msk, (W, H), interpolation=cv2.INTER_NEAREST)
                    K[:2] = K[:2] * ratio

                image = Image.fromarray(np.array(image * 255.0, dtype=np.byte), "RGB")

                focalX = K[0, 0]
                focalY = K[1, 1]
                FovX = focal2fov(focalX, image.size[0])
                FovY = focal2fov(focalY, image.size[1])

                # # load smpl data
                # i = int(os.path.basename(image_path)[:-4])
                # vertices_path = os.path.join(path, 'smpl_vertices', '{}.npy'.format(i))
                # xyz = np.load(vertices_path).astype(np.float32)
                #
                # smpl_param_path = os.path.join(path, "smpl_params", '{}.npy'.format(i))
                # smpl_param = np.load(smpl_param_path, allow_pickle=True).item()
                # Rh = smpl_param['Rh']
                # smpl_param['R'] = cv2.Rodrigues(Rh)[0].astype(np.float32)
                # smpl_param['Th'] = smpl_param['Th'].astype(np.float32)
                # smpl_param['shapes'] = smpl_param['shapes'].astype(np.float32)
                # smpl_param['poses'] = smpl_param['poses'].astype(np.float32)

                # load smpl data
                i = int(os.path.basename(image_path)[:-4])
                pose = poses[i][None]
                Th = Ths[i][None]
                Rh = Rhs[i][None]
                smpl_param = {
                    'R': cv2.Rodrigues(Rh)[0],
                    'Th': Th,
                    'shapes': shapes,
                    'poses': pose
                }
                s = torch.from_numpy(shapes)
                p = torch.from_numpy(pose)
                t = torch.from_numpy(Th)
                xyz, _ = smpl_chomp(s, p, t)
                xyz = xyz.detach().cpu().numpy().reshape(-1, 3)

                # obtain the original bounds for point sampling
                min_xyz = np.min(xyz, axis=0)
                max_xyz = np.max(xyz, axis=0)
                min_xyz -= 0.05
                max_xyz += 0.05
                world_bound = np.stack([min_xyz, max_xyz], axis=0)

                # get bounding mask and bcakground mask
                bound_mask = get_bound_2d_mask(world_bound, K, w2c[:3], image.size[1], image.size[0])
                bound_mask = Image.fromarray(np.array(bound_mask * 255.0, dtype=np.byte))

                bkgd_mask = Image.fromarray(np.array(msk * 255.0, dtype=np.byte))

                cam_infos.append(CameraInfo(uid=idx, pose_id=pose_index, R=R, T=T, K=K, FovY=FovY, FovX=FovX, image=image,
                                            image_path=image_path, image_name=image_name, bkgd_mask=bkgd_mask,
                                            bound_mask=bound_mask, width=image.size[0], height=image.size[1],
                                            smpl_param=smpl_param, world_vertex=xyz, world_bound=world_bound,
                                            big_pose_smpl_param=big_pose_smpl_param, big_pose_world_vertex=big_pose_xyz,
                                            big_pose_world_bound=big_pose_world_bound))

                idx += 1
    else:
        # load smpl params
        model = setting['model'][0]
        pose_embedding = setting['pose_embedding'][0]
        vposer = setting['vposer']
        frames_seq = dataset_obj.frames
        body_pose = vposer.decode(pose_embedding, t=frames_seq).view(frames_seq, -1)
        body_pose[:, -6:] = 0.
        body_pose = body_pose.detach().cpu().numpy()
        orient = np.array(model.global_orient.detach().cpu().numpy())
        poses = np.hstack((orient, body_pose)).reshape(-1, 72)
        shapes = model.betas.detach().cpu().numpy().astype(np.float32).reshape(-1, 10)
        Ths = model.transl.detach().cpu().numpy().astype(np.float32).reshape(-1, 3)
        Rhs = model.global_orient.detach().cpu().numpy().astype(np.float32).reshape(-1, 3)
        smpl_chomp = SMPLModel(device=torch.device('cpu'), model_path='assets/SMPL_NEUTRAL.pkl')

        idx = 0
        for pose_index in range(pose_num):
            for view_index in range(len(output_view)):
                image_path = os.path.join(path, ims[pose_index][view_index].replace('\\', '/'))
                if not novel_view_vis:
                    cam_ind = cam_inds[pose_index][view_index]
                    K = np.array(cams['K'][cam_ind])
                    D = np.array(cams['D'][cam_ind])
                    R = np.array(cams['R'][cam_ind])
                    T = np.array(cams['T'][cam_ind])

                    # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
                    w2c = np.eye(4)
                    w2c[:3, :3] = R
                    w2c[:3, 3:4] = T

                    R = np.transpose(w2c[:3, :3])  # R is stored transposed due to 'glm' in CUDA code
                    T = w2c[:3, 3]

                    # load smpl data
                    i = int(os.path.basename(image_path)[:-4])
                    pose = poses[i][None]
                    Th = Ths[i][None]
                    Rh = Rhs[i][None]
                    smpl_param = {
                        'R': cv2.Rodrigues(Rh)[0],
                        'Th': Th,
                        'shapes': shapes,
                        'poses': pose
                    }
                    s = torch.from_numpy(shapes)
                    p = torch.from_numpy(pose)
                    t = torch.from_numpy(Th)
                    xyz, _ = smpl_chomp(s, p, t)
                    xyz = xyz.detach().cpu().numpy().reshape(-1, 3)

                    # obtain the original bounds for point sampling
                    min_xyz = np.min(xyz, axis=0)
                    max_xyz = np.max(xyz, axis=0)
                    min_xyz -= 0.05
                    max_xyz += 0.05
                    world_bound = np.stack([min_xyz, max_xyz], axis=0)

                    # get bounding mask and bcakground mask
                    h = 1536
                    w = 2048
                    bound_mask = get_bound_2d_mask(world_bound, K, w2c[:3], h, w)
                    bound_mask = Image.fromarray(np.array(bound_mask * 255.0, dtype=np.byte))

                    cam_infos.append(
                        CameraInfo(uid=idx, pose_id=pose_index, R=R, T=T, K=K, FovY=np.array([0]), FovX=np.array([0]), image=np.array([0]),
                                   image_path=image_path, image_name='', bkgd_mask=np.array([0]),
                                   bound_mask=bound_mask, width=w, height=h,
                                   smpl_param=smpl_param, world_vertex=xyz, world_bound=world_bound,
                                   big_pose_smpl_param={}, big_pose_world_vertex=np.array([0]),
                                   big_pose_world_bound=np.array([0])))

    return cam_infos
