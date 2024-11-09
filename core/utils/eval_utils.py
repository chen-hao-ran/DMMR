'''
 @FileName    : eval_utils.py
 @EditTime    : 2021-07-19 22:59:41
 @Author      : Buzhen Huang
 @Email       : hbz@seu.edu.cn
 @Description : 
'''

from core.utils.init_guess import rec_3D_joints
from pickle import TRUE
import torch
from torch._C import device
import torch.nn as nn
import numpy as np
import cv2
import os
from core.utils.module_utils import joint_projection, load_pkl, load_camera_para, rot_mesh
# from utils.imutils import vis_img
from core.utils.visualization3d import Visualization
from collections import OrderedDict
from tqdm import tqdm
from prettytable import PrettyTable
import torch.optim as optim
from torchgeometry import rotation_matrix_to_angle_axis, angle_axis_to_rotation_matrix
import sys
from core.utils.data_parser import FittingData
from core.utils.recompute3D import recompute3D

class HumanEval(nn.Module):
    def __init__(self, name, generator=None, smpl=None, dtype='float32', **kwargs):
        super(HumanEval, self).__init__()
        self.generator = generator
        self.smpl = smpl
        if dtype == 'float32':
            self.dtype = torch.float32
        else:
            self.dtype = torch.float64
        self.name = name
        self.dataset_scale = self.dataset_mapping(self.name)
        # self.J_regressor_H36 = np.load('data/J_regressor_h36m.npy').astype(np.float32)
        self.J_regressor_H36 = None
        self.J_regressor_LSP = np.load('data/J_regressor_lsp.npy').astype(np.float32)
        self.J_regressor_SMPL = self.smpl.J_regressor.clone().cpu().detach().numpy()

        self.eval_handler_mapper = dict(
            OcMotion=self.LSPEvalHandler,
            VCL_3DOH50K=self.LSPEvalHandler,
            VCLMP=self.LSPEvalHandler,
            h36m_synthetic_protocol2=self.LSPEvalHandler,
            h36m_valid_protocol1=self.LSPEvalHandler,
            h36m_valid_protocol2=self.LSPEvalHandler,
            Human36M=self.H36MEvalHandler,
            MPI3DPW=self.SMPLEvalHandler,
            MPI3DPWOC=self.SMPLEvalHandler,
            Panoptic_haggling1=self.PanopticEvalHandler,
            Panoptic_mafia2=self.PanopticEvalHandler,
            Panoptic_pizza1=self.PanopticEvalHandler,
            Panoptic_ultimatum1=self.PanopticEvalHandler,
            Panoptic_Eval=self.PanopticEvalHandler,
            MuPoTS_origin=self.MuPoTSEvalHandler,
            MPI3DHP=self.MuPoTSEvalHandler,
        )

    def dataset_mapping(self, name):
        if name == 'VCLMP':
            return 105
        if name == 'VCL_3DOH50K':
            return 7
        else:
            return 1

    def estimate_translation_from_intri(self, S, joints_2d, joints_conf, fx=5000., fy=5000., cx=128., cy=128.):
        num_joints = S.shape[0]
        # focal length
        f = np.array([fx, fy])
        # optical center
    # center = np.array([img_size/2., img_size/2.])
        center = np.array([cx, cy])
        # transformations
        Z = np.reshape(np.tile(S[:,2],(2,1)).T,-1)
        XY = np.reshape(S[:,0:2],-1)
        O = np.tile(center,num_joints)
        F = np.tile(f,num_joints)
        weight2 = np.reshape(np.tile(np.sqrt(joints_conf),(2,1)).T,-1)

        # least squares
        Q = np.array([F*np.tile(np.array([1,0]),num_joints), F*np.tile(np.array([0,1]),num_joints), O-np.reshape(joints_2d,-1)]).T
        c = (np.reshape(joints_2d,-1)-O)*Z - F*XY

        # weighted least squares
        W = np.diagflat(weight2)
        Q = np.dot(W,Q)
        c = np.dot(W,c)

        # square matrix
        A = np.dot(Q.T,Q)
        b = np.dot(Q.T,c)

        # test
        A += np.eye(A.shape[0]) * 1e-6

        # solution
        trans = np.linalg.solve(A, b)
        return trans

    def cal_trans(self, J3ds, J2ds, intris):
        trans = np.zeros((J3ds.shape[0], 3))
        for i, (J3d, J2d, intri) in enumerate(zip(J3ds, J2ds, intris)):
            fx = intri[0][0]
            fy = intri[1][1]
            cx = intri[0][2]
            cy = intri[1][2]
            j_conf = J2d[:,2] 
            trans[i] = self.estimate_translation_from_intri(J3d, J2d[:,:2], j_conf, cx=cx, cy=cy, fx=fx, fy=fy)
        return trans

    def get_abs_meshes(self, pre_meshes, joints_2ds, intri):
        lsp14_to_lsp13 = [0,1,2,3,4,5,6,7,8,9,10,11,13]
        pre_meshes = ((pre_meshes + 0.5) * 2. * self.dataset_scale)
        # get predicted 3D joints and estimate translation
        joints = np.matmul(self.J_regressor_LSP, pre_meshes)
        # we use 12 joints to calculate translation
        transl = self.cal_trans(joints[:,lsp14_to_lsp13], joints_2ds, intri)

        abs_mesh = pre_meshes + transl[:,np.newaxis,:]
        return abs_mesh

    def compute_similarity_transform(self, S1, S2):
        '''
        Computes a similarity transform (sR, t) that takes
        a set of 3D points S1 (3 x N) closest to a set of 3D points S2,
        where R is an 3x3 rotation matrix, t 3x1 translation, s scale.
        i.e. solves the orthogonal Procrutes problem.
        '''
        transposed = False
        if S1.shape[0] != 3 and S1.shape[0] != 2:
            S1 = S1.T
            S2 = S2.T
            transposed = True
        assert(S2.shape[1] == S1.shape[1])

        # 1. Remove mean.
        mu1 = S1.mean(axis=1, keepdims=True)
        mu2 = S2.mean(axis=1, keepdims=True)
        X1 = S1 - mu1
        X2 = S2 - mu2

        # 2. Compute variance of X1 used for scale.
        var1 = np.sum(X1**2)

        # 3. The outer product of X1 and X2.
        K = X1.dot(X2.T)

        # 4. Solution that Maximizes trace(R'K) is R=U*V', where U, V are
        # singular vectors of K.
        U, s, Vh = np.linalg.svd(K)
        V = Vh.T
        # Construct Z that fixes the orientation of R to get det(R)=1.
        Z = np.eye(U.shape[0])
        Z[-1, -1] *= np.sign(np.linalg.det(U.dot(V.T)))
        # Construct R.
        R = V.dot(Z.dot(U.T))

        # 5. Recover scale.
        scale = np.trace(R.dot(K)) / var1

        # 6. Recover translation.
        t = mu2 - scale*(R.dot(mu1))

        # 7. Error:
        S1_hat = scale*R.dot(S1) + t

        if transposed:
            S1_hat = S1_hat.T

        return S1_hat


    def align_by_pelvis(self, joints, get_pelvis=False, format='lsp'):
        """
        Assumes joints is 14 x 3 in LSP order.
        Then hips are: [3, 2]
        Takes mid point of these points, then subtracts it.
        """
        if format == 'lsp':
            left_id = 3
            right_id = 2

            pelvis = (joints[left_id, :] + joints[right_id, :]) / 2.
        elif format in ['smpl', 'h36m']:
            pelvis_id = 0
            pelvis = joints[pelvis_id, :]
        elif format in ['mpi']:
            pelvis_id = 14
            pelvis = joints[pelvis_id, :]
        if get_pelvis:
            return joints - np.expand_dims(pelvis, axis=0), pelvis
        else:
            return joints - np.expand_dims(pelvis, axis=0)

    def align_mesh_by_pelvis(self, mesh, joints, get_pelvis=False, format='lsp'):
        """
        Assumes joints is 14 x 3 in LSP order.
        Then hips are: [3, 2]
        Takes mid point of these points, then subtracts it.
        """
        if format == 'lsp':
            left_id = 3
            right_id = 2
            pelvis = (joints[left_id, :] + joints[right_id, :]) / 2.
        elif format in ['smpl', 'h36m']:
            pelvis_id = 0
            pelvis = joints[pelvis_id, :]
        elif format in ['mpi']:
            pelvis_id = 14
            pelvis = joints[pelvis_id, :]
        if get_pelvis:
            return mesh - np.expand_dims(pelvis, axis=0), pelvis
        else:
            return mesh - np.expand_dims(pelvis, axis=0)

    def compute_error_accel(self, joints_gt, joints_pred, vis=None):
        """
        Computes acceleration error:
            1/(n-2) \sum_{i=1}^{n-1} X_{i-1} - 2X_i + X_{i+1}
        Note that for each frame that is not visible, three entries in the
        acceleration error should be zero'd out.
        Args:
            joints_gt (Nx14x3).
            joints_pred (Nx14x3).
            vis (N).
        Returns:
            error_accel (N-2).
        """
        # (N-2)x14x3
        accel_gt = joints_gt[:-2] - 2 * joints_gt[1:-1] + joints_gt[2:]
        accel_pred = joints_pred[:-2] - 2 * joints_pred[1:-1] + joints_pred[2:]

        normed = np.linalg.norm(accel_pred - accel_gt, axis=2)

        if vis is None:
            new_vis = np.ones(len(normed), dtype=bool)
        else:
            invis = np.logical_not(vis)
            invis1 = np.roll(invis, -1)
            invis2 = np.roll(invis, -2)
            new_invis = np.logical_or(invis, np.logical_or(invis1, invis2))[:-2]
            new_vis = np.logical_not(new_invis)

        return np.mean(normed[new_vis], axis=1)

    def compute_errors(self, gt3ds, preds, format='lsp', confs=None):
        """
        Gets MPJPE after pelvis alignment + MPJPE after Procrustes.
        Evaluates on the 14 common joints.
        Inputs:
        - gt3ds: N x 14 x 3
        - preds: N x 14 x 3
        """
        if confs is None:
            confs = np.ones((gt3ds.shape[:2]))
        abs_errors, errors, errors_pa, abs_pck, pck = [], [], [], [], []
        for i, (gt3d, pred, conf) in enumerate(zip(gt3ds, preds, confs)):
            gt3d = gt3d.reshape(-1, 3)

            # Get abs error.
            joint_error = np.sqrt(np.sum((gt3d - pred)**2, axis=1)) * conf
            abs_errors.append(np.mean(joint_error))

            # Get abs pck.
            abs_pck.append(np.mean(joint_error < 150) * 100)

            # Root align.
            gt3d = self.align_by_pelvis(gt3d, format=format)
            pred3d = self.align_by_pelvis(pred, format=format)

            joint_error = np.sqrt(np.sum((gt3d - pred3d)**2, axis=1)) * conf
            errors.append(np.mean(joint_error))

            # Get pck
            pck.append(np.mean(joint_error < 150) * 100)

            # Get PA error.
            pred3d_sym = self.compute_similarity_transform(pred3d, gt3d)
            pa_error = np.sqrt(np.sum((gt3d - pred3d_sym)**2, axis=1)) * conf
            errors_pa.append(np.mean(pa_error))

        return abs_errors, errors, errors_pa, abs_pck, pck

    def LSPEvalHandler(self, premesh, gt_joint):
        joints = np.matmul(self.J_regressor_LSP, premesh)

        gt_joint = gt_joint / self.dataset_scale * 1000
        joints = joints * 1000
        abs_error, error, error_pa, abs_pck, pck = self.compute_errors(gt_joint, joints, format='lsp')
        accel = self.compute_error_accel(gt_joint, joints).tolist()
        return abs_error, error, error_pa, abs_pck, pck, accel

    def H36MEvalHandler(self, premesh, gt_joint):
        joints = np.matmul(self.J_regressor_H36, premesh)
        joints = joints[:,[0,4,5,6,1,2,3,7,8,9,10,11,12,13,14,15,16]]
        gt_joint = gt_joint / self.dataset_scale * 1000
        joints = joints * 1000
        abs_error, error, error_pa, abs_pck, pck = self.compute_errors(gt_joint, joints, format='h36m')
        accel = self.compute_error_accel(gt_joint, joints).tolist()
        return abs_error, error, error_pa, abs_pck, pck, accel

    def PanopticEvalHandler(self, premesh, gt_joint):
        joints = np.matmul(self.J_regressor_H36, premesh)
        conf = gt_joint[:,:,-1].copy()
        gt_joint = gt_joint[:,:,:3]
        gt_joint = gt_joint / self.dataset_scale * 1000
        joints = joints / self.dataset_scale * 1000
        abs_error, error, error_pa, abs_pck, pck = self.compute_errors(gt_joint, joints, format='h36m', confs=conf)
        return abs_error, error, error_pa, abs_pck, pck

    def MuPoTSEvalHandler(self, premesh, gt_joint):
        h36m_to_MPI = [10, 8, 14, 15, 16, 11, 12, 13, 4, 5, 6, 1, 2, 3, 0, 7, 9]
        joints = np.matmul(self.J_regressor_H36, premesh)
        # gt_joint = gt_joint / self.dataset_scale * 1000
        joints = joints / self.dataset_scale * 1000
        joints = joints[:,h36m_to_MPI]

        abs_error, error, error_pa, abs_pck, pck = self.compute_errors(gt_joint, joints, format='mpi')

        accel = self.compute_error_accel(gt_joint, joints).tolist()

        if self.name == 'MPI3DHP':
            return abs_error, error, error_pa, abs_pck, pck, accel
        else:
            return abs_error, error, error_pa, abs_pck, pck, joints

    def SMPLEvalHandler(self, premesh, gt_joint):
        joints = np.matmul(self.J_regressor_SMPL, premesh)

        gt_joint = gt_joint / self.dataset_scale * 1000
        joints = joints / self.dataset_scale * 1000
        abs_error, error, error_pa, abs_pck, pck = self.compute_errors(gt_joint, joints, format='smpl')
        return abs_error, error, error_pa, abs_pck, pck

    def SMPLMeshEvalHandler(self, premeshes, gt_meshes):
        premeshes = premeshes * 1000
        gt_meshes = gt_meshes * 1000

        joints = np.matmul(self.J_regressor_LSP, premeshes)
        gt_joints = np.matmul(self.J_regressor_LSP, gt_meshes)

        vertex_errors = []

        for i, (premesh, gt_mesh, joint, gt_joint) in enumerate(zip(premeshes, gt_meshes, joints, gt_joints)):
            # Root align.
            premesh = self.align_mesh_by_pelvis(premesh, joint, format='lsp')
            gt_mesh = self.align_mesh_by_pelvis(gt_mesh, gt_joint, format='lsp')

            vertex_error = np.sqrt(np.sum((premesh - gt_mesh)**2, axis=1))
            vertex_errors.append(np.mean(vertex_error))

        return vertex_errors

    def forward(self, output, data):
        gt_joint = data['gt_3d'] #.to(self.dtype).detach().cpu().numpy()
        
        if self.name == 'MuPoTS_origin':
            abs_error, error, error_pa, abs_pck, pck, joints = self.eval_handler_mapper[self.name](meshes, gt_joint)
            imnames = data['raw_img']
            joints_2ds = np.matmul(intris, joints.transpose((0,2,1)))
            joints_2ds = (joints_2ds[:,:2,:] / joints_2ds[:,-1:,:]).transpose((0,2,1))
            joints = joints.tolist()
            joints_2ds = joints_2ds.tolist()
        else:
            abs_error, error, error_pa, abs_pck, pck, accel = self.eval_handler_mapper[self.name](output, gt_joint)
            imnames = [None] * len(abs_error)
            joints = [None] * len(abs_error)
            joints_2ds = [None] * len(abs_error)

        # calculate vertex error
        if data['gt_mesh'].shape[1] < 6890:
            vertex_error = [None] * len(abs_error)
        else:
            # mesh in mm
            meshes = output
            vertex_error = self.SMPLMeshEvalHandler(meshes, data['gt_mesh'])
        return vertex_error, error, error_pa, abs_pck, pck, imnames, joints, joints_2ds, accel


class Eval_Dataset():
    def __init__(self, dataset_dir='', smpl=None, evaltool=None, align_cam=False, dataset_name='', dtype='float32'):
        self.dataset_dir = os.path.join(dataset_dir, 'template')
        self.joint_mapper = [0, 1, 2, 3, 6, 7, 8, 12, 13, 14, 15, 17, 18, 19, 25, 26, 27]
        self.smpl = smpl
        self.evaltool = evaltool
        self.align_cam = align_cam

    def add_camera_mesh(self, extrinsic, camerascale=1):
        # 8 points camera
        r = np.zeros((3,4,3))

        r[0][0] = np.array([-0.5, 0.5, 0]) * camerascale
        r[0][1] = np.array([0.5, 0.5, 0]) * camerascale
        r[0][2] = np.array([0.5, -0.5, 0]) * camerascale
        r[0][3] = np.array([-0.5, -0.5, 0]) * camerascale

        r[1][0] = np.array([-1, 1, 1]) * camerascale
        r[1][1] = np.array([1, 1, 1]) * camerascale
        r[1][2] = np.array([1, -1, 1]) * camerascale
        r[1][3] = np.array([-1, -1, 1]) * camerascale

        r[2][0] = np.array([-0.5, 0.5, -2]) * camerascale
        r[2][1] = np.array([0.5, 0.5, -2]) * camerascale
        r[2][2] = np.array([0.5, -0.5, -2]) * camerascale
        r[2][3] = np.array([-0.5, -0.5, -2]) * camerascale

        P = np.zeros((3, 40))
        for i in range(3):
            P[:,i * 8 + 0] = r[i][0] 
            P[:,i * 8 + 1] = r[i][1]
            P[:,i * 8 + 2] = r[i][1] 
            P[:,i * 8 + 3] = r[i][2]
            P[:,i * 8 + 4] = r[i][2] 
            P[:,i * 8 + 5] = r[i][3]
            P[:,i * 8 + 6] = r[i][3] 
            P[:,i * 8 + 7] = r[i][0]

        for i in range(2):
            P[:,24 + i * 8 + 0] = r[0][0] 
            P[:,24 + i * 8 + 1] = r[i + 1][0]
            P[:,24 + i * 8 + 2] = r[0][1] 
            P[:,24 + i * 8 + 3] = r[i + 1][1]
            P[:,24 + i * 8 + 4] = r[0][2] 
            P[:,24 + i * 8 + 5] = r[i + 1][2]
            P[:,24 + i * 8 + 6] = r[0][3] 
            P[:,24 + i * 8 + 7] = r[i + 1][3]

        # // transform from camera space to object space
        # // this step is critical for visualizing the cameras since our viewpoint is in the object space
        M = np.linalg.inv(extrinsic)
        for i in range(P.shape[1]):
            t = np.ones((4,))
            t[:3] = P[:,i]
            p = np.dot(M, t)
            P[:,i] = p[:3] / p[3]

        return P

    def rigid_transform_3D(self, A, B):
        assert len(A) == len(B)

        N = A.shape[0]  # total points
        centroid_A = np.mean(A, axis=0)
        centroid_B = np.mean(B, axis=0)

        # centre the points
        AA = A - np.tile(centroid_A, (N, 1))
        BB = B - np.tile(centroid_B, (N, 1))

        H = np.matmul(np.transpose(AA),BB)
        U, S, Vt = np.linalg.svd(H)
        R = np.matmul(Vt.T, U.T)

        # special reflection case
        flag = 0
        if np.linalg.det(R) < 0:
            #print("Reflection detected")
            Vt[2, :] *= -1
            R = np.matmul(Vt.T,U.T)
            flag = 1

        t = -np.matmul(R, centroid_A) + centroid_B
        # err = B - np.matmul(A,R.T) - t.reshape([1, 3])
        return R, t, flag

    def align_camera(self, gt_cams, pred_cams):
        
        # visualize input cameras
        # viz = Visualization()
        # for cam in gt_cams:
        #     t = self.add_camera_mesh(cam)
        #     viz.visualize_cameras(t.T, [1, 0.5, 0.5])

        # for cam in pred_cams:
        #     t = self.add_camera_mesh(cam)
        #     viz.visualize_cameras(t.T, [0.5, 0.5, 1])

        gt_pos = []
        pred_pos = []
        for cam in gt_cams:
            pos = np.dot(np.linalg.inv(cam), np.array([0,0,0,1]))[:3]
            gt_pos.append(pos)
        for cam in pred_cams:
            pos = np.dot(np.linalg.inv(cam), np.array([0,0,0,1]))[:3]
            pred_pos.append(pos)
        gt_pos = np.array(gt_pos)
        pred_pos = np.array(pred_pos)

        # viz.visualize_points(gt_pos, [1,0,0])
        # viz.visualize_points(pred_pos, [0,0,1])

        R_, t_, flag = self.rigid_transform_3D(pred_pos, gt_pos)

        # viz.visualize_points(out_pos, [0,1,1])

        transform = np.eye(4)
        transform[:3,:3] = R_
        transform[:3,3] = t_

        out_pos = (np.matmul(transform, np.insert(pred_pos, 3, 1, axis=1).T).T)[:,:3]

        rots = []
        for cam in pred_cams:
            rot = np.dot(cam, np.linalg.inv(transform))
            rots.append(rot)

        if 0:
            viz = Visualization()
            for cam in gt_cams:
                t = self.add_camera_mesh(cam)
                viz.visualize_cameras(t.T, [1, 0.5, 0.5])

            for cam in rots:
                t = self.add_camera_mesh(cam)
                viz.visualize_cameras(t.T, [0, 0.5, 1])

        rots_errors = []
        for pred_c, gt_c in zip(pred_cams, gt_cams):
            p_rot = np.dot(pred_c, np.linalg.inv(transform))
            p_rot = cv2.Rodrigues(p_rot[:3,:3])[0].reshape(1, -1)
            g_rot = cv2.Rodrigues(gt_c[:3,:3])[0].reshape(1, -1)
            error = np.linalg.norm(p_rot - g_rot)
            rots_errors.append(error)

        rot_error = np.mean(np.array(rots_errors))
        pos_error = np.mean(np.linalg.norm(out_pos - gt_pos, axis=0))

        return transform, pos_error, rot_error

    def align_camera_nonlinear(self, pred_cams, gt_cams, est_scale=False, viz=False):

        # visualize input cameras
        # viz = Visualization()
        # for cam in gt_cams:
        #     t = self.add_camera_mesh(cam)
        #     viz.visualize_cameras(t.T, [1, 0.5, 0.5])

        # for cam in pred_cams:
        #     t = self.add_camera_mesh(cam)
        #     viz.visualize_cameras(t.T, [0.5, 0.5, 1])

        gt_pos, pred_pos, cam_id = [], [], []
        for i, (cam_gt, cam_pre) in enumerate(zip(gt_cams, pred_cams)):
            if (cam_pre - np.eye(4)).max() < 1e-3 and i != 0:
                continue
            # if i == 9:
            #     continue
            pos = np.dot(np.linalg.inv(cam_gt), np.array([0,0,0,1]))[:3]
            gt_pos.append(pos)
            pos = np.dot(np.linalg.inv(cam_pre), np.array([0,0,0,1]))[:3]
            pred_pos.append(pos)
            cam_id.append(i)
        gt_pos = np.array(gt_pos)
        pred_pos = np.array(pred_pos)

        R_, t_, flag = self.rigid_transform_3D(pred_pos, gt_pos)
        R_ = cv2.Rodrigues(R_)[0]

        refine_R = torch.tensor(R_.reshape(-1, 3), device=torch.device('cuda'), requires_grad=True)
        refine_t = torch.tensor(t_.reshape(-1, 3), device=torch.device('cuda'), requires_grad=True)
        refine_s = torch.tensor([1.0], device=torch.device('cuda'), dtype=torch.float64, requires_grad=True)
        pred_pos = torch.tensor(pred_pos, device=torch.device('cuda'))
        gt_pos = torch.tensor(gt_pos, device=torch.device('cuda'))

        if est_scale:
            final_param = [refine_R, refine_t, refine_s]
        else:
            final_param = [refine_R, refine_t]

        optimizer = optim.Adam(filter(lambda p:p.requires_grad, final_param), lr=0.0001)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=10, verbose=True)

        def loss_func(pred_R, pred_t, pred_s, init, gt_pos):
            rot_mat = angle_axis_to_rotation_matrix(pred_R)
            rot_mat = rot_mat[0]
            rot_mat[:3,3] = pred_t
            ones = torch.ones((init.shape[0], 1), dtype=init.dtype, device=init.device)
            init = init * pred_s
            init = torch.cat([init, ones], dim=1).permute(1,0)
            pos = torch.matmul(rot_mat, init).permute(1,0)
            pos = pos[:,:3]
            loss = torch.norm(pos - gt_pos, dim=1).sum()
            return loss

        loss_init = loss_func(refine_R, refine_t, refine_s, pred_pos, gt_pos) + 1
        while True:

            loss = loss_func(refine_R, refine_t, refine_s, pred_pos, gt_pos)

            if loss < 1e-6:
                break

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # scheduler.step(loss)
            if torch.abs(loss - loss_init) < 1e-5:
                break
            print(loss)
            loss_init = loss

        transform = angle_axis_to_rotation_matrix(refine_R)[0].detach().cpu().numpy()
        transform[:3,3] = refine_t.detach().cpu().numpy()
        s = refine_s.detach().cpu().numpy()
        pred_pos = pred_pos.detach().cpu().numpy() * s
        gt_pos = gt_pos.detach().cpu().numpy()
        out_pos = (np.matmul(transform, np.insert(pred_pos, 3, 1, axis=1).T).T)[:,:3]


        aligned_cam, aligned_gt = [], []
        for c_id in cam_id:
            cam = pred_cams[c_id]
            cam[:3,3] = cam[:3,3] * s
            transform1 = np.linalg.inv(transform)
            rot = np.dot(cam, transform1)
            aligned_cam.append(rot)
            aligned_gt.append(gt_cams[c_id])
        aligned_cam = np.array(aligned_cam)
        aligned_gt = np.array(aligned_gt)
        
        # for i, cam in enumerate(pred_cams):
        #     if (cam - np.eye(4)).max() < 1e-3:
        #         rots.append(gt_cams[i])
        #     # elif i == 29:
        #     #     rots.append(gt_cams[29])
        #     else:
        #         cam[:3,3] = cam[:3,3] * s
        #         transform1 = np.linalg.inv(transform)
        #         rot = np.dot(cam, transform1)
        #         rots.append(rot)

        if viz:
            visualize = Visualization()
            visualize.visualize_points(gt_pos, [1,0,0])
            visualize.visualize_points(out_pos, [0,0,1])
            for i, (gt, pre) in enumerate(zip(aligned_gt, aligned_cam)):
                gt = self.add_camera_mesh(gt, camerascale=0.1)
                visualize.visualize_cameras(gt.T, [1, 0.5, 0.5])
                pre = self.add_camera_mesh(pre, camerascale=0.1)
                visualize.visualize_cameras(pre.T, [0, 0.5, 1])
                # print(i)
            while True:
                visualize.show()

        rots_errors = []
        for pred_c, gt_c in zip(aligned_cam, aligned_gt):
            p_rot = cv2.Rodrigues(pred_c[:3,:3])[0].reshape(1, -1)
            g_rot = cv2.Rodrigues(gt_c[:3,:3])[0].reshape(1, -1)
            error = np.linalg.norm(abs(p_rot) - abs(g_rot))
            rots_errors.append(error)

        rot_error = np.mean(np.array(rots_errors))
        pos_error = np.mean(np.linalg.norm(out_pos - gt_pos, axis=0))

        return transform, s, rot_error, pos_error, aligned_cam, cam_id

    def transform_with_cam(self, verts, transform, s):
        verts = verts * s
        verts = (np.matmul(transform, np.insert(verts, 3, 1, axis=1).T).T)[:,:3]

        return verts

    def eval_cam(self, pred_cam, gt_cam):
        pred_extris, pred_intris = load_camera_para(pred_cam)
        gt_extris, gt_intris = load_camera_para(gt_cam)

        transform, s, rot_error, pos_error, aligned_cam = self.align_camera_nonlinear(pred_extris, gt_extris, est_scale=False, viz=False)

        print('Camera Pos Error: %f mm  Camera Rot Error: %f deg' %(pos_error * 1000.0, rot_error * (180/np.pi)))
        return rot_error, pos_error

    # def align_camera_nonlinear(self, gt_cams, pred_cams):
        
    #     # visualize input cameras
    #     # viz = Visualization()
    #     # for cam in gt_cams:
    #     #     t = self.add_camera_mesh(cam)
    #     #     viz.visualize_cameras(t.T, [1, 0.5, 0.5])

    #     # for cam in pred_cams:
    #     #     t = self.add_camera_mesh(cam)
    #     #     viz.visualize_cameras(t.T, [0.5, 0.5, 1])

    #     gt_pos = []
    #     pred_pos = []
    #     for cam in gt_cams:
    #         pos = np.dot(np.linalg.inv(cam), np.array([0,0,0,1]))[:3]
    #         gt_pos.append(pos)
    #     for cam in pred_cams:
    #         pos = np.dot(np.linalg.inv(cam), np.array([0,0,0,1]))[:3]
    #         pred_pos.append(pos)
    #     gt_pos = np.array(gt_pos)
    #     pred_pos = np.array(pred_pos)

    #     # viz.visualize_points(gt_pos, [1,0,0])
    #     # viz.visualize_points(pred_pos, [0,0,1])

    #     R_, t_, flag = self.rigid_transform_3D(pred_pos, gt_pos)
    #     R_ = cv2.Rodrigues(R_)[0]
    #     # viz.visualize_points(out_pos, [0,1,1])

    #     refine_R = torch.tensor(R_.reshape(-1, 3), device=torch.device('cuda'), requires_grad=True)
    #     refine_t = torch.tensor(t_.reshape(-1, 3), device=torch.device('cuda'), requires_grad=True)
    #     pred_pos = torch.tensor(pred_pos, device=torch.device('cuda'))
    #     gt_pos = torch.tensor(gt_pos, device=torch.device('cuda'))

    #     final_param = [refine_R, refine_t]
    #     optimizer = optim.Adam(filter(lambda p:p.requires_grad, final_param), lr=0.0001)
    #     scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=10, verbose=True)

    #     def loss_func(pred_R, pred_t, init, gt_pos):
    #         rot_mat = angle_axis_to_rotation_matrix(pred_R)
    #         rot_mat = rot_mat[0]
    #         rot_mat[:3,3] = pred_t
    #         ones = torch.ones((init.shape[0], 1), dtype=init.dtype, device=init.device)
    #         init = torch.cat([init, ones], dim=1).permute(1,0)
    #         pos = torch.matmul(rot_mat, init).permute(1,0)
    #         pos = pos[:,:3]
    #         loss = torch.norm(pos - gt_pos, dim=1).sum()
    #         return loss

    #     loss_init = loss_func(refine_R, refine_t, pred_pos, gt_pos) + 1
    #     while True:

    #         loss = loss_func(refine_R, refine_t, pred_pos, gt_pos)

    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()
    #         # scheduler.step(loss)
    #         if torch.abs(loss - loss_init) < 1e-5:
    #             break

    #         loss_init = loss

    #     transform = angle_axis_to_rotation_matrix(refine_R)[0].detach().cpu().numpy()
    #     transform[:3,3] = refine_t.detach().cpu().numpy()
    #     pred_pos = pred_pos.detach().cpu().numpy()
    #     gt_pos = gt_pos.detach().cpu().numpy()

    #     out_pos = (np.matmul(transform, np.insert(pred_pos, 3, 1, axis=1).T).T)[:,:3]

    #     rots = []
    #     for cam in pred_cams:
    #         rot = np.dot(cam, np.linalg.inv(transform))
    #         rots.append(rot)

    #     if 0:
    #         viz = Visualization()
    #         while True:
    #             for cam in gt_cams:
    #                 t = self.add_camera_mesh(cam)
    #                 viz.visualize_cameras(t.T, [1, 0.5, 0.5])

    #             for cam in rots:
    #                 t = self.add_camera_mesh(cam)
    #                 viz.visualize_cameras(t.T, [0, 0.5, 1])

    #     rots_errors = []
    #     for pred_c, gt_c in zip(pred_cams, gt_cams):
    #         p_rot = np.dot(pred_c, np.linalg.inv(transform))
    #         p_rot = cv2.Rodrigues(p_rot[:3,:3])[0].reshape(1, -1)
    #         g_rot = cv2.Rodrigues(gt_c[:3,:3])[0].reshape(1, -1)
    #         error = np.linalg.norm(p_rot - g_rot)
    #         rots_errors.append(error)

    #     rot_error = np.mean(np.array(rots_errors))
    #     pos_error = np.mean(np.linalg.norm(out_pos - gt_pos, axis=0))

    #     return transform, pos_error, rot_error

class Eval_Human36M(Eval_Dataset):
    def __init__(self, dataset_dir, smpl=None, evaltool=None, align_cam=False, dataset_name='', dtype='float32'):
        super(Eval_Dataset, self).__init__()
        # self.dataset_dir = os.path.join(dataset_dir, 'Eval_Human36M10FPS')
        # self.annots = load_pkl(os.path.join(self.dataset_dir, 'annot.pkl'))
        self.dataset_dir = os.path.join(dataset_dir, 'Eval_Human36M')
        self.annots = np.load(os.path.join(self.dataset_dir, 'data_3d_h36m.npz'), allow_pickle=True)['positions_3d'].item()
        self.annots = {ant:self.annots[ant] for ant in self.annots if ant in ['S9', 'S11']}
        self.joint_mapper = [0, 1, 2, 3, 6, 7, 8, 12, 13, 14, 15, 17, 18, 19, 25, 26, 27]
        self.smpl = smpl
        self.evaltool = evaltool
        self.align_cam = align_cam



    def evaluate(self, results_dir):
        vertex_errors, errors, error_pas, abs_pcks, pcks, imnames, joints, joints_2ds, error_cam_p, error_cam_r, error_accel = [], [], [], [], [], [], [], [], [], [], []
        
        for seq in self.annots:
            seq_dir = os.path.join(results_dir, seq)
            actions = self.annots[seq]
            for act in actions:
                # if act == 'Directions':
                #     continue
                meshes = []
                print('processing %s %s' %(seq, act))

                if act[-1] in ['1', '2', '3']:
                    act_name = act[:-2] + act[-1]
                else:
                    act_name = act
                # act_name = act
                act_dir = os.path.join(seq_dir, act_name)
                files = os.listdir(act_dir)
                data = self.annots[seq][act][:,self.joint_mapper,:]
                data = data[:]
                for i, f in enumerate(files):
                    f_dir = os.path.join(act_dir, f)
                    param = load_pkl(f_dir)#['person00']

                    pose = torch.from_numpy(param['pose']).reshape(-1, 72)
                    betas = torch.from_numpy(param['betas']).reshape(-1, 10)
                    transl = torch.from_numpy(param['transl']).reshape(-1, 3)
                    scale = torch.from_numpy(param['scale']).reshape(-1, 1)

                    model_output = self.smpl(betas=betas, body_pose=pose[:,3:], global_orient=pose[:,:3], transl=transl, scale=scale, return_verts=True, return_full_pose=False)

                    meshes.append(model_output.vertices.numpy())
                    
                output = np.concatenate(meshes)
                assert output.shape[0] == data.shape[0]
                vertex_error, error, error_pa, abs_pck, pck, imname, joint, joints_2d, accel = self.evaltool(output, {'gt_3d':data})

                # print(error, error_pa)
                vertex_errors += vertex_error
                errors += error
                error_pas += error_pa
                abs_pcks += abs_pck
                pcks += pck
                imnames += imname
                joints += joint
                joints_2ds += joints_2d
                error_accel += accel

        if vertex_errors[0] is not None:
            vertex_error = np.mean(np.array(vertex_errors))
        else:
            vertex_error = -1
        error = np.mean(np.array(errors))
        error_pa = np.mean(np.array(error_pas))
        abs_pck = np.mean(np.array(abs_pcks))
        pck = np.mean(np.array(pcks))
        accel = np.mean(np.array(error_accel))

        return vertex_error, error, error_pa, accel

class Eval_OcMotion(Eval_Dataset):
    def __init__(self, dataset_dir, smpl=None, evaltool=None, align_cam=False, dataset_name='', dtype='float32'):
        super(Eval_Dataset, self).__init__()
        # self.dataset_dir = os.path.join(dataset_dir, 'Eval_Human36M10FPS')
        # self.annots = load_pkl(os.path.join(self.dataset_dir, 'annot.pkl'))
        self.dataset_dir = os.path.join(dataset_dir, 'OcMotion')
        self.annots = load_pkl(os.path.join(self.dataset_dir, 'test.pkl'))
        self.smpl = smpl
        self.evaltool = evaltool
        self.align_cam = align_cam



    def evaluate(self, results_dir):
        vertex_errors, errors, error_pas, abs_pcks, pcks, imnames, joints, joints_2ds, error_cam_p, error_cam_r, error_accel = [], [], [], [], [], [], [], [], [], [], []
        
        for idx, seq in enumerate(self.annots):
            if idx != 0:
                continue
            _, seq_name, cam_name, img_name = seq[0][0]['img_path'].split('/')
            seq_dir = os.path.join(results_dir, 'results', seq_name)
            gt_meshes, gt_joints = [], []
            meshes = []
            print('processing %s' %(seq_name))
            for frame in seq[0]:
                _, _, _, img_name = frame['img_path'].split('/')
                index = img_name.split('.')[0]
                
                gt_param = frame['0']
                gt_joints.append(np.array(gt_param['lsp_joints_3d'], dtype=np.float32))
                gt_pose = torch.from_numpy(np.array(gt_param['pose'], dtype=np.float32)).reshape(-1, 72)
                gt_betas = torch.from_numpy(np.array(gt_param['betas'], dtype=np.float32)).reshape(-1, 10)
                gt_transl = torch.from_numpy(np.array(gt_param['trans'], dtype=np.float32)).reshape(-1, 3)
                # gt_scale = torch.from_numpy(np.array(gt_param['scale'], dtype=np.float32)).reshape(-1, 1)
                gt_scale = torch.ones((1, 1))

                gt_model_output = self.smpl(betas=gt_betas, body_pose=gt_pose[:,3:], global_orient=gt_pose[:,:3], transl=gt_transl, scale=gt_scale, return_verts=True, return_full_pose=False)

                gt_meshes.append(gt_model_output.vertices.numpy())

                
                f_dir = os.path.join(seq_dir, index + '.pkl')
                param = load_pkl(f_dir)['person00']

                pose = torch.from_numpy(param['pose']).reshape(-1, 72)
                betas = torch.from_numpy(param['betas']).reshape(-1, 10)
                transl = torch.from_numpy(param['transl']).reshape(-1, 3)
                scale = torch.from_numpy(param['scale']).reshape(-1, 1)

                model_output = self.smpl(betas=betas, body_pose=pose[:,3:], global_orient=pose[:,:3], transl=transl, scale=scale, return_verts=True, return_full_pose=False)

                meshes.append(model_output.vertices.numpy())
                
            output = np.concatenate(meshes)
            gt_joints = np.array(gt_joints)
            gt_output = np.concatenate(gt_meshes)
            assert output.shape[0] == gt_output.shape[0]
            vertex_error, error, error_pa, abs_pck, pck, imname, joint, joints_2d, accel = self.evaltool(output, {'gt_mesh':gt_output, 'gt_3d':gt_joints})

            # print(error, error_pa)
            vertex_errors += vertex_error
            errors += error
            error_pas += error_pa
            abs_pcks += abs_pck
            pcks += pck
            imnames += imname
            joints += joint
            joints_2ds += joints_2d
            error_accel += accel

        if vertex_errors[0] is not None:
            vertex_error = np.mean(np.array(vertex_errors))
        else:
            vertex_error = -1
        error = np.mean(np.array(errors))
        error_pa = np.mean(np.array(error_pas))
        abs_pck = np.mean(np.array(abs_pcks))
        pck = np.mean(np.array(pcks))
        accel = np.mean(np.array(error_accel))

        return vertex_error, error, error_pa, accel

class Eval_MPI3DHP(Eval_Dataset):
    def __init__(self, dataset_dir, smpl=None, evaltool=None, align_cam=False, dataset_name='', dtype='float32'):
        super(Eval_MPI3DHP, self).__init__()

        self.dataset_dir = os.path.join(dataset_dir, 'Eval_MPI3DHP')
        self.annots = load_pkl(os.path.join(self.dataset_dir, 'annot/test.pkl'))
        self.joint_mapper = [0, 1, 2, 3, 6, 7, 8, 12, 13, 14, 15, 17, 18, 19, 25, 26, 27]
        self.smpl = smpl
        self.evaltool = evaltool
        self.align_cam = align_cam

    def evaluate(self, results_dir):
        vertex_errors, errors, error_pas, abs_pcks, pcks, imnames, joints, joints_2ds, error_cam_p, error_cam_r, error_accel = [], [], [], [], [], [], [], [], [], [], []
        total_pair = []
        seq_last = None
        count = 1
        for index in self.annots:
            ant = self.annots[index]
            name = ant['img_path']
            _, seq, _, img_name = name.split('\\')

            if seq != seq_last:
                if seq_last is not None:
                    total_pair.append([meshes, gt_joints])
                meshes = []
                gt_joints = []
                vis = []
                seq_last = seq
                count = 1

            img_ind = img_name.split('.')[0].split('_')[1][1:]

            f_dir = os.path.join(results_dir, seq, '%s.pkl' %img_ind)
            if not os.path.exists(f_dir):
                print('Not complete data !!!')
                break

            param = load_pkl(f_dir)
            if len(param) < 1:
                vis.append(0)
                continue
            else:
                param=param['person00']
                vis.append(1)

            pose = torch.from_numpy(param['pose']).reshape(-1, 72)
            betas = torch.from_numpy(param['betas']).reshape(-1, 10)
            transl = torch.from_numpy(param['transl']).reshape(-1, 3)
            scale = torch.from_numpy(param['scale']).reshape(-1, 1)

            model_output = self.smpl(betas=betas, body_pose=pose[:,3:], global_orient=pose[:,:3], transl=transl, scale=scale, return_verts=True, return_full_pose=False)

            meshes.append(model_output.vertices.numpy())
            gt_joints.append(np.array(ant['mpi_joints_3d']))

        total_pair.append([meshes, gt_joints, vis])

        for data in total_pair:

            output = np.concatenate(data[0])
            data = np.concatenate(data[1])
            assert output.shape[0] == data.shape[0]
            vertex_error, error, error_pa, abs_pck, pck, imname, joint, joints_2d, accel = self.evaltool(output, {'gt_3d':data})

            # print(error, error_pa)
            vertex_errors += vertex_error
            errors += error
            error_pas += error_pa
            abs_pcks += abs_pck
            pcks += pck
            imnames += imname
            joints += joint
            joints_2ds += joints_2d
            error_accel += accel

        if vertex_errors[0] is not None:
            vertex_error = np.mean(np.array(vertex_errors))
        else:
            vertex_error = -1
        error = np.mean(np.array(errors))
        error_pa = np.mean(np.array(error_pas))
        abs_pck = np.mean(np.array(abs_pcks))
        pck = np.mean(np.array(pcks))
        accel = np.mean(np.array(error_accel)) # sampled in every 10 frame. The frame rate is 25

        return vertex_error, error, error_pa, accel

class Eval_Campus(Eval_Dataset):
    def __init__(self, dataset_dir, smpl=None, evaltool=None, align_cam=False, dataset_name='', dtype='float32'):
        super(Eval_Campus, self).__init__()
        import scipy.io as scio
        self.regressor = torch.tensor(np.load('data/J_regressor_halpe.npy'), dtype=torch.float32)
        self.dataset_name = dataset_name
        if self.dataset_name == 'Campus':
            self.dataset_dir = os.path.join(dataset_dir, 'Campus_Clip')
            self.eval_range = [i for i in range(350, 470)]
            self.eval_range += [i for i in range(650, 750)]
        elif self.dataset_name == 'Shelf':
            self.dataset_dir = os.path.join(dataset_dir, 'Shelf_Clip')
            self.eval_range = [i for i in range(300, 600)]
        else:
            sys.exit(0)

        actorsGT = scio.loadmat(os.path.join (self.dataset_dir, 'actorsGT.mat' ))
        self.actor3D = actorsGT['actor3D'][0]

        self.joint_mapper = [0, 1, 2, 3, 6, 7, 8, 12, 13, 14, 15, 17, 18, 19, 25, 26, 27]
        self.smpl = smpl
        self.evaltool = evaltool
        self.align_cam = align_cam

    def is_right(self, model_start_point, model_end_point, gt_strat_point, gt_end_point, alpha=0.5):
        bone_lenth = np.linalg.norm ( gt_end_point - gt_strat_point )
        start_difference = np.linalg.norm ( gt_strat_point - model_start_point )
        end_difference = np.linalg.norm ( gt_end_point - model_end_point )
        return ((start_difference + end_difference) / 2) <= alpha * bone_lenth

    def evaluate(self, results_dir):
        # from core.utils.visualization3d import Visualization
        # viz = Visualization()
        results = os.path.join(results_dir, 'results', self.dataset_name)
        cam_dirs = os.path.join(results_dir, 'cameras', self.dataset_name)
        sequences = os.listdir(results)
        rot_error, pos_error = 0., 0.

        if self.align_cam:
            gt_cam_dir = os.path.join(self.dataset_dir, 'camparams', self.dataset_name, 'camparams.txt')
            pred_cam_dir = os.path.join(results_dir, 'camparams', self.dataset_name, '00900.txt')
            gt_extris, gt_intris = load_camera_para(gt_cam_dir)
            pred_extris, pred_intris = load_camera_para(pred_cam_dir)

            transform, cam_scale, rot_error, pos_error, anligned_cam, cam_id = self.align_camera_nonlinear(pred_extris, gt_extris, est_scale=False, viz=False)

        check_result = np.zeros ( (len ( self.actor3D[0] ), len ( self.actor3D ), 10), dtype=np.int32 )
        accuracy_cnt = 0
        error_cnt = 0

        files = os.listdir(results)
        for f in files:
            img_id = int(f.split('.')[0])
            if img_id not in self.eval_range:
                continue
            f_dir = os.path.join(results, f)

            param = load_pkl(f_dir)
            params = []
            if len(param) < 1:
                continue
            else:
                for i in range(4):
                    id_name = 'person%02d' %i
                    if id_name in param.keys():
                        params.append(param[id_name])
                    else:
                        params.append(None)

            for pid in range ( len ( self.actor3D ) ):
                if self.actor3D[pid][img_id][0].shape == (1, 0) or self.actor3D[pid][img_id][0].shape == (0, 0):
                    continue

                if params[pid] is None:
                    check_result[img_id, pid, :] = -1
                    print('Cannot get any pose in img: %d' %img_id)
                    continue

                pose = torch.from_numpy(params[pid]['pose']).reshape(-1, 72)
                betas = torch.from_numpy(params[pid]['betas']).reshape(-1, 10)
                transl = torch.from_numpy(params[pid]['transl']).reshape(-1, 3)
                scale = torch.from_numpy(params[pid]['scale']).reshape(-1, 1)

                model_output = self.smpl(betas=betas, body_pose=pose[:,3:], global_orient=pose[:,:3], transl=transl, scale=scale, return_verts=True, return_full_pose=False)

                if self.align_cam:
                    verts = model_output.vertices.detach().cpu().numpy()[0]
                    verts = self.transform_with_cam(verts, transform, cam_scale)
                    verts = torch.tensor(verts, dtype=torch.float32)
                else:
                    verts = model_output.vertices[0]
                model_pose = torch.matmul(self.regressor, verts).numpy()
                model_pose = model_pose[[16,14,12,11,13,15,10,8,6,5,7,9,18,17]]

                # Shelf VPoser gt cam
                # old regressor
                model_pose[12] = model_pose[12] + (model_pose[13] - model_pose[12]) * np.array([0.4, 0.5, 0.3])
                model_pose[13] = model_pose[13] + (model_pose[13] - model_pose[12]) * np.array([-0.4, -0.5, -0.3])
                ######################################
                
                # model_pose[12] = model_pose[12] + (model_pose[13] - model_pose[12]) * np.array([0.4, 0.5, 0.3])
                # model_pose[13] = model_pose[13] + (model_pose[13] - model_pose[12]) * np.array([-0.4, -0.5, -0.2])
                # model_pose[12] = model_pose[12] + (model_pose[13] - model_pose[12]) * np.array([0.0, 0.0, 0.3])
                # model_pose[13] = model_pose[13] + (model_pose[13] - model_pose[12]) * np.array([-0.5, -1.5, -0.2])

                model_pose = model_pose[:14]
                gt_pose = self.actor3D[pid][img_id][0]

                if self.align_cam:
                    R, t, _ = self.rigid_transform_3D(model_pose, gt_pose)
                    transform = np.eye(4)
                    transform[:3,:3] = R
                    transform[:3,3] = t
                    model_pose = out_pos = (np.matmul(transform, np.insert(model_pose, 3, 1, axis=1).T).T)[:,:3]

                # from core.utils.visualization3d import Visualization
                # viz = Visualization()
                # viz.visualize_points(model_pose, (0,0,1))
                # viz.visualize_points(gt_pose, (1,0,0))
                # while True:
                #     viz.show()

                bones = [[0, 1], [1, 2], [3, 4], [4, 5], [6, 7], [7, 8], [9, 10], [10, 11], [12, 13]]
                for i, bone in enumerate ( bones ):
                    start_point, end_point = bone
                    if self.is_right ( model_pose[start_point], model_pose[end_point], gt_pose[start_point],
                                gt_pose[end_point] ):
                        check_result[img_id, pid, i] = 1
                        accuracy_cnt += 1
                    else:
                        check_result[img_id, pid, i] = -1
                        error_cnt += 1
                gt_hip = (gt_pose[2] + gt_pose[3]) / 2
                model_hip = (model_pose[2] + model_pose[3]) / 2
                if self.is_right ( model_hip, model_pose[12], gt_hip, gt_pose[12] ):
                    check_result[img_id, pid, -1] = 1
                    accuracy_cnt += 1
                else:
                    check_result[img_id, pid, -1] = -1
                    error_cnt += 1


        bone_group = OrderedDict (
            [('Head', np.array ( [8] )), ('Torso', np.array ( [9] )), ('Upper arms', np.array ( [5, 6] )),
            ('Lower arms', np.array ( [4, 7] )), ('Upper legs', np.array ( [1, 2] )),
            ('Lower legs', np.array ( [0, 3] ))] )

        total_avg = np.sum ( check_result > 0 ) / np.sum ( np.abs ( check_result ) )
        person_wise_avg = np.sum ( check_result > 0, axis=(0, 2) ) / np.sum ( np.abs ( check_result ), axis=(0, 2) )

        bone_wise_result = OrderedDict ()
        bone_person_wise_result = OrderedDict ()
        for k, v in bone_group.items ():
            bone_wise_result[k] = np.sum ( check_result[:, :, v] > 0 ) / np.sum ( np.abs ( check_result[:, :, v] ) )
            bone_person_wise_result[k] = np.sum ( check_result[:, :, v] > 0, axis=(0, 2) ) / np.sum (
                np.abs ( check_result[:, :, v] ), axis=(0, 2) )

        tb = PrettyTable ()
        tb.field_names = ['Bone Group'] + ['Actor {}'.format(i) for i in range(3)] + ['Average']
        list_tb = [tb.field_names]
        for k, v in bone_person_wise_result.items():
            this_row = [k] + [np.char.mod ( '%.2f', i * 100 ) for i in v[:3]] + [np.char.mod ( '%.2f', np.sum ( v[:3] ) * 100 / len ( v[:3] ) )]
            list_tb.append ( [float ( i ) if isinstance ( i, type ( np.array ( [] ) ) ) else i for i in this_row] )
            tb.add_row ( this_row )

        this_row = ['Total'] + [np.char.mod ( '%.2f', i * 100 ) for i in person_wise_avg[:3]] + [np.char.mod ( '%.2f', np.sum ( person_wise_avg[:3] ) * 100 / len ( person_wise_avg[:3] ) )]
        tb.add_row ( this_row )
        list_tb.append ( [float ( i ) if isinstance ( i, type ( np.array ( [] ) ) ) else i for i in this_row] )
        print ( tb )

        print('Camera Pos Error: %f mm  Camera Rot Error: %f deg' %(pos_error * 1000.0, rot_error * (180/np.pi)))
        return 0, 0, 0, 0

class Eval_MHHI(Eval_Dataset):
    def __init__(self, dataset_dir, smpl=None, evaltool=None, align_cam=False, dataset_name='', dtype='float32'):
        super(Eval_MHHI, self).__init__()
        self.dataset_dir = os.path.join(dataset_dir, 'Eval_DoubleB')
        self.marker = os.path.join(self.dataset_dir, 'marker/marker.dat')
        # set error parameters
        self.thres = 10000
        self.dataset_name = 'doubleB'
        self.smpl = smpl
        self.evaltool = evaltool
        self.align_cam = align_cam

    def evaluate(self, results_dir):
        
        results = os.path.join(results_dir, 'results/doubleB')
        cam_dirs = os.path.join(results_dir, 'cameras/doubleB')
        rot_error, pos_error = 0., 0.

        if self.align_cam:
            gt_cam_dir = os.path.join(self.dataset_dir, 'camparams', self.dataset_name, 'camparams.txt')
            pred_cam_dir = os.path.join(results_dir, 'camparams', self.dataset_name, '00385.txt')
            gt_extris, gt_intris = load_camera_para(gt_cam_dir)
            pred_extris, pred_intris = load_camera_para(pred_cam_dir)

            transform, cam_scale, rot_error, pos_error, aligned_cam, cam_id = self.align_camera_nonlinear(pred_extris, gt_extris, est_scale=True, viz=True)


        files = os.listdir(results)
        f_dir = os.path.join(results, files[1])
        param = load_pkl(f_dir)
        params = []
        id_name = 'person00'
        if id_name in param.keys():
            params.append(param[id_name])
        else:
            params.append(None)

        pose = torch.from_numpy(params[0]['pose']).reshape(-1, 72)
        betas = torch.from_numpy(params[0]['betas']).reshape(-1, 10)
        transl = torch.from_numpy(params[0]['transl']).reshape(-1, 3)
        scale = torch.from_numpy(params[0]['scale']).reshape(-1, 1)

        model_output = self.smpl(betas=betas, body_pose=pose[:,3:], global_orient=pose[:,:3], transl=transl, scale=scale, return_verts=True, return_full_pose=False)

        # read files
        if self.align_cam:
            verts = model_output.vertices.detach().cpu().numpy()[0].astype(float)
            verts = self.transform_with_cam(verts, transform, cam_scale)
        else:
            verts = model_output.vertices.detach().cpu().numpy()[0].astype(float)
        o_lines = verts
        mf = open(self.marker)
        m_lines = mf.readlines()

        # calculate the location of the corresponding points of min distance
        mod_loc = self.find_cp(m_lines, o_lines)

        # calculate the distance of corresponding points
        mcd_set = self.find_cd(m_lines, o_lines, mod_loc)

        ave_set = []

        gt_verts = []
        pred_verts = []

        # calculate the mean value of all frames
        for ind, f in tqdm(enumerate(files), total=len(files)):
            f_dir = os.path.join(results, f)
            param = load_pkl(f_dir)
            params = []
            id_name = 'person00'
            if id_name in param.keys():
                params.append(param[id_name])
            else:
                params.append(None)

            pose = torch.from_numpy(params[0]['pose']).reshape(-1, 72)
            betas = torch.from_numpy(params[0]['betas']).reshape(-1, 10)
            transl = torch.from_numpy(params[0]['transl']).reshape(-1, 3)
            scale = torch.from_numpy(params[0]['scale']).reshape(-1, 1)

            model_output = self.smpl(betas=betas, body_pose=pose[:,3:], global_orient=pose[:,:3], transl=transl, scale=scale, return_verts=True, return_full_pose=False)

            # read files
            if self.align_cam:
                verts = model_output.vertices.detach().cpu().numpy()[0].astype(float)
                verts = self.transform_with_cam(verts, transform, cam_scale)
            else:
                verts = model_output.vertices.detach().cpu().numpy()[0].astype(float)
            eof_lines = verts

            efm_line = m_lines[ind]

            # calculate the mean value of each frame
            e_ave, gt_vert, pred_vert = self.cal_cd(efm_line, eof_lines, mod_loc, mcd_set, self.thres)
            gt_verts.append(gt_vert)
            pred_verts.append(pred_vert)
            ave_set.append(e_ave)

        # calculate the standard deviation
        res_mv = np.mean(ave_set)
        res_sd = np.std(ave_set)
        # print res_sd, ave_set
        gt_verts = np.array(gt_verts)
        pred_verts = np.array(pred_verts)
        accel = 0 #self.evaltool.compute_error_accel(gt_verts, pred_verts)

        # write results
        # np.savez(join(res_dir, 'res_.npz'), mean_val=res_mv, stan_dev=res_sd)
        # np.savetxt(join(res_dir, 'res_8v.txt'), [res_mv, res_sd])
        print('Mean: %f  Std:  %f' %(res_mv, res_sd))
        mf.close()
        print('Camera Pos Error: %f mm  Camera Rot Error: %f deg' %(pos_error * 1000.0, rot_error * (180/np.pi)))
        return 0, 0, 0, accel

    def cal_cd(self, efm_line, eof_lines, mod_loc, mcd_set, thres):

        m_d = np.array(efm_line.split()[4:]).reshape(-1, 3).astype(float)
        epd_set = []
        marker = []
        est_verts = []
        for ind in range(38):

            # the distanc of two corresponding points
            a = m_d[ind]-mcd_set[ind]
            b = eof_lines[mod_loc[ind]]*1000

            # calculate euclidean distance
            epd = np.linalg.norm(a-b)

            # error parameters
            if epd <= thres:		
                epd_set.append(epd)
                marker.append(a)
                est_verts.append(b)

        return np.mean(epd_set), np.array(marker), np.array(est_verts)


    def find_cp(self, m_lines, o_lines):

        m_d = np.array(m_lines[0].split()[4:]).reshape(-1, 3).astype(float)

        mod_loc = []

        for mnd in range(0, 38):

            mod_set = []
            for ind, line in enumerate(o_lines):

                if ind < 6890:
                    o_d = line*1000
                    mo_d = np.linalg.norm(m_d[mnd]-o_d)
                    mod_set.append(mo_d)

            # find the location of the corresponding points of the min distance
            loc = mod_set.index(min(mod_set))
            mod_loc.append(loc)

        return mod_loc

    def find_cd(self, m_lines, o_lines, mod_loc):

        m_d = np.array(m_lines[0].split()[4:]).reshape(-1, 3).astype(float)

        mcd_set = []

        # obtain the distance of corresponding points
        for ind in range(0, 38):
            mcd = m_d[ind] - o_lines[mod_loc[ind]]*1000
            mcd_set.append(mcd)

        return mcd_set

class ShelfGT:
    def __init__(self, actor3D) -> None:
        self.actor3D = actor3D
        self.actor3D = self.actor3D[:3]

    def __getitem__(self, index):
        results = []
        for pid in range(len(self.actor3D)):
            gt_pose = self.actor3D[pid][index-2][0]
            if gt_pose.shape == (1, 0) or gt_pose.shape == (0, 0):
                continue
            keypoints3d = convert_shelf_shelfgt(gt_pose)
            results.append({'id': pid, 'keypoints3d': keypoints3d})
        return results

def convert_shelf_shelfgt(keypoints):
    gt_hip = (keypoints[2] + keypoints[3]) / 2
    gt = np.vstack((keypoints, gt_hip))
    return gt

def convert_shelf1(keypoints3d):
    shelf15 = np.zeros((15, 3))
    shelf15[:14] = keypoints3d
    shelf15[-1] = (keypoints3d[2] + keypoints3d[3]) / 2
    return shelf15


class Eval_Shelf_Cam(Eval_Dataset):
    def __init__(self, dataset_dir, smpl=None, evaltool=None, align_cam=False, dataset_name='', dtype='float32'):
        super(Eval_Shelf_Cam, self).__init__()

        self.gt_cam = os.path.join(dataset_dir, 'Shelf_Clip/camparams/Shelf/camparams.txt')
        self.gt_extris, self.gt_intris = load_camera_para(self.gt_cam)
        self.images_dir = os.path.join(dataset_dir, 'Shelf_Clip/\images/Shelf')
        self.keypoints_dir = os.path.join(dataset_dir, 'Shelf_Clip/keypoints_gt/Shelf')
        self.align_cam = align_cam
        self.dateset = FittingData(os.path.join(dataset_dir, 'Shelf_Clip'), frames=701, num_people=4)

    def evaluate(self, results_dir):
        # from core.utils.visualization3d import Visualization
        # viz = Visualization()
        pred_extris, pred_intris = load_camera_para(results_dir)

        pred_focal = np.concatenate((pred_intris[:,0,0], pred_intris[:,1,1]), axis=0) 
        gt_focal = np.concatenate((self.gt_intris[:,0,0], self.gt_intris[:,1,1]), axis=0)
        focal_error = np.mean(abs(pred_focal - gt_focal) / gt_focal)

        transform, cam_scale, rot_error, pos_error, aligned_cam, cam_id = self.align_camera_nonlinear(pred_extris, self.gt_extris, est_scale=True, viz=False)

        filter_joints_idx = [5,6,7,8,9,10,11,12,13,14,15,16]
        loss = []
        for i, data in enumerate(self.dateset):
            keypoints = data['keypoints']
            for idx in tqdm(range(self.dateset.num_people), total=self.dateset.num_people):
                for f in range(self.dateset.frames):
                    pack = [[keypoints[k][f][idx][filter_joints_idx], aligned_cam[i], pred_intris[k]] for i, k in enumerate(cam_id) if (keypoints[k][f][idx] is not None and keypoints[k][f][idx][:,2].max() > 0.2)]
                    if len(pack) < 2: # do not process single view case
                        continue
                    keps = np.array([p[0] for p in pack])
                    cam_extris = np.array([p[1] for p in pack])
                    cam_intris = np.array([p[2] for p in pack])
                    rec_joints3d = recompute3D(cam_extris, cam_intris, keps.copy())

                    for kp, extri, intri in zip(keps, cam_extris, cam_intris):
                        reproj_pose, _ = joint_projection(rec_joints3d, extri, intri, np.zeros((1,)), False)
                        reproj_loss = np.mean(np.linalg.norm(reproj_pose-kp[:,:2], axis=-1) * kp[:,-1])
                        loss.append(reproj_loss)
        loss = np.mean(np.array(loss))

        print('Focal Error: %f  Camera Pos Error: %f mm  Camera Rot Error: %f deg Re-Projection Loss: %f pixel' %(focal_error * 100., pos_error * 1000.0, rot_error * (180/np.pi), loss))
        return 0, 0, 0, 0

class Eval_Panoptic_Cam(Eval_Dataset):
    def __init__(self, dataset_dir, smpl=None, evaltool=None, align_cam=False, dataset_name='', dtype='float32'):
        super(Eval_Panoptic_Cam, self).__init__()

        self.gt_cam = os.path.join(dataset_dir, 'Panoptic31View/camparams/Panoptic/camparams.txt')
        self.gt_extris, self.gt_intris = load_camera_para(self.gt_cam)
        self.images_dir = os.path.join(dataset_dir, 'Panoptic31View/images/Panoptic')
        self.keypoints_dir = os.path.join(dataset_dir, 'Panoptic31View/keypoints/Panoptic')
        self.align_cam = align_cam
        self.dateset = FittingData(os.path.join(dataset_dir, 'Panoptic31View'), frames=750, num_people=5)

    def evaluate(self, results_dir):
        # from core.utils.visualization3d import Visualization
        # viz = Visualization()
        pred_extris, pred_intris = load_camera_para(results_dir)

        pred_focal = np.concatenate((pred_intris[:,0,0], pred_intris[:,1,1]), axis=0) 
        gt_focal = np.concatenate((self.gt_intris[:,0,0], self.gt_intris[:,1,1]), axis=0)
        focal_error = np.mean(abs(pred_focal - gt_focal) / gt_focal)

        transform, cam_scale, rot_error, pos_error, aligned_cam, cam_id = self.align_camera_nonlinear(pred_extris, self.gt_extris, est_scale=True, viz=False)
        # rot_error, pos_error = 0, 0
        # aligned_cam = self.gt_extris

        filter_joints_idx = [5,6,7,8,9,10,11,12,13,14,15,16]
        loss = []
        for i, data in enumerate(self.dateset):
            keypoints = data['keypoints']
            for idx in tqdm(range(self.dateset.num_people), total=self.dateset.num_people):
                for f in range(self.dateset.frames):
                    pack = [[keypoints[k][f][idx][filter_joints_idx], aligned_cam[i], pred_intris[k]] for i, k in enumerate(cam_id) if (keypoints[k][f][idx] is not None and keypoints[k][f][idx][:,2].max() > 0.2)]
                    if len(pack) < 2: # do not process single view case
                        continue
                    keps = np.array([p[0] for p in pack])
                    cam_extris = np.array([p[1] for p in pack])
                    cam_intris = np.array([p[2] for p in pack])
                    rec_joints3d = recompute3D(cam_extris, cam_intris, keps.copy())

                    for kp, extri, intri in zip(keps, cam_extris, cam_intris):
                        reproj_pose, _ = joint_projection(rec_joints3d, extri, intri, np.zeros((1,)), False)
                        reproj_loss = np.mean(np.linalg.norm(reproj_pose-kp[:,:2], axis=-1) * kp[:,-1])
                        loss.append(reproj_loss)
        loss = np.mean(np.array(loss))

        print('Focal Error: %f  Camera Pos Error: %f mm  Camera Rot Error: %f deg Re-Projection Loss: %f pixel' %(focal_error * 100., pos_error * 1000.0, rot_error * (180/np.pi), loss))
        return 0, 0, 0, 0
