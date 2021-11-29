# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems and the Max Planck Institute for Biological
# Cybernetics. All rights reserved.
#
# Contact: ps-license@tuebingen.mpg.de

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import sys
import os

import time

import numpy as np

import torch
import torch.nn as nn

from core.utils import module_utils
import cv2


@torch.no_grad()
class FittingMonitor():
    def __init__(self, summary_steps=1, visualize=False,
                 maxiters=100, ftol=2e-09, gtol=1e-05,
                 body_color=(1.0, 1.0, 0.9, 1.0),
                 model_type='smpl',
                 **kwargs):
        # super(FittingMonitor, self).__init__()

        self.maxiters = maxiters
        self.ftol = ftol
        self.gtol = gtol

        self.visualize = visualize
        self.summary_steps = summary_steps
        self.body_color = body_color
        self.model_type = model_type
        
    # def __enter__(self):
    #     self.steps = 0
    #     if self.visualize:
    #         self.mv = MeshViewer(body_color=self.body_color)
    #     return self

    # def __exit__(self, exception_type, exception_value, traceback):
    #     if self.visualize:
    #         self.mv.close_viewer()

    def set_colors(self, vertex_color):
        batch_size = self.colors.shape[0]

        self.colors = np.tile(
            np.array(vertex_color).reshape(1, 3),
            [batch_size, 1])

    def run_fitting(self, optimizer, closure, params, body_model,
                    use_vposer=True, pose_embedding=None, vposer=None, use_motionprior=False,
                    **kwargs):
        ''' Helper function for running an optimization process
            Parameters
            ----------
                optimizer: torch.optim.Optimizer
                    The PyTorch optimizer object
                closure: function
                    The function used to calculate the gradients
                params: list
                    List containing the parameters that will be optimized
                body_model: nn.Module
                    The body model PyTorch module
                use_vposer: bool
                    Flag on whether to use VPoser (default=True).
                pose_embedding: torch.tensor, BxN
                    The tensor that contains the latent pose variable.
                vposer: nn.Module
                    The VPoser module
            Returns
            -------
                loss: float
                The final loss value
        '''
        #append_wrists = self.model_type == 'smpl' and use_vposer
        prev_loss = None
        print('\n')
        for n in range(self.maxiters):
            loss, loss_dict = optimizer.step(closure)
            if torch.isnan(loss).sum() > 0:
                print('NaN loss value, stopping!')
                break

            if torch.isinf(loss).sum() > 0:
                print('Infinite loss value, stopping!')
                break

            if n > 0 and prev_loss is not None and self.ftol > 0:
                loss_rel_change = module_utils.rel_change(prev_loss, loss.item())

                if loss_rel_change <= self.ftol:
                    break

            if all([torch.abs(var.grad.view(-1).max()).item() < self.gtol
                    for var in params if var.grad is not None]):
                break

            if self.visualize and n % self.summary_steps == 0:
                body_pose = vposer.decode(
                    pose_embedding, output_type='aa').view(
                        1, -1) if use_vposer else None

                if append_wrists:
                    wrist_pose = torch.zeros([body_pose.shape[0], 6],
                                             dtype=body_pose.dtype,
                                             device=body_pose.device)
                    body_pose = torch.cat([body_pose, wrist_pose], dim=1)
                model_output = body_model(
                    return_verts=True, body_pose=body_pose)
                vertices = model_output.vertices.detach().cpu().numpy()

                self.mv.update_mesh(vertices.squeeze(),
                                    body_model.faces)

            prev_loss = loss.item()
            # print('stage fitting loss: ', prev_loss)
            print('stage fitting loss: %.1f' % prev_loss, loss_dict)
        return prev_loss

    def create_fitting_closure(self,
                               optimizer, body_models, camera=None,
                               gt_joints=None, loss=None,
                               joints_conf=None,
                               flags=None,
                               gt_joints3d=None,
                               joints3d_conf=None,
                               joint_weights=None,
                               return_verts=True, return_full_pose=False,
                               use_vposer=False, vposer=None,
                               use_motionprior=False,
                               pose_embeddings=None,
                               create_graph=False,
                               use_3d=False,
                               **kwargs):
        faces_tensor = body_models[0].faces_tensor.view(-1)

        # the vposer++ contains wrist
        append_wrists = False
        def fitting_func(backward=True):
            if backward:
                optimizer.zero_grad()

            body_model_outputs = []
            for model, pose_embedding in zip(body_models, pose_embeddings):
                if use_vposer:
                    body_pose = vposer.decode(
                        pose_embedding, output_type='aa').view(
                            pose_embedding.shape[0], -1)
                elif use_motionprior:
                    body_pose = vposer.decode(
                        pose_embedding, t=pose_embedding.shape[0]).view(
                            pose_embedding.shape[0], -1)
                else:
                    body_pose = None

                if append_wrists:
                    wrist_pose = torch.zeros([body_pose.shape[0], 6],
                                            dtype=body_pose.dtype,
                                            device=body_pose.device)
                    body_pose = torch.cat([body_pose, wrist_pose], dim=1)
                torch.cuda.empty_cache()
                body_model_output = model(return_verts=return_verts,
                                            body_pose=body_pose,
                                            return_full_pose=return_full_pose)
                body_model_outputs.append(body_model_output)
            
            total_loss, loss_dict = loss(body_model_outputs, flags, camera=camera, body_models=body_models,
                              gt_jointss=gt_joints,
                              body_model_faces=faces_tensor,
                              joints_confs=joints_conf,
                              gt_joints3d=gt_joints3d,
                              joints3d_conf=joints3d_conf,
                              joint_weights=joint_weights,
                              pose_embeddings=pose_embeddings,
                              use_vposer=use_vposer,
                              use_motionprior=use_motionprior,
                              **kwargs)

            if backward:
                total_loss.backward(retain_graph=False, create_graph=create_graph)
                # clip grad for GRU based motion prior
                # if use_motionprior:
                #     for emb in pose_embeddings:
                #         # nn.utils.clip_grad_value_(emb, 1e7)
                #         print(emb.grad.max())
                # if use_motionprior:
                #     nn.utils.clip_grad_norm_(pose_embeddings, 1e6, norm_type=2)
            # self.steps += 1
            # if self.visualize and self.steps % self.summary_steps == 0:
            #     model_output = body_model(return_verts=True,
            #                               body_pose=body_pose)
            #     vertices = model_output.vertices.detach().cpu().numpy()

            #     self.mv.update_mesh(vertices.squeeze(),
            #                         body_model.faces)

            return total_loss, loss_dict

        return fitting_func


def create_loss(loss_type='smplify', **kwargs):
    if loss_type == 'smplify':
        return SMPLifyLoss(**kwargs)
    else:
        raise ValueError('Unknown loss type: {}'.format(loss_type))


class SMPLifyLoss(nn.Module):

    def __init__(self, search_tree=None,
                 pen_distance=None, tri_filtering_module=None,
                 rho=100,
                 body_pose_prior=None,
                 shape_prior=None,
                 angle_prior=None,
                 use_joints_conf=True,
                 interpenetration=True, dtype=torch.float32,
                 data_weight=1.0,
                 body_pose_weight=0.0,
                 kinetic_weight=0.0,
                 shape_weight=0.0,
                 bending_prior_weight=0.0,
                 coll_loss_weight=0.0,
                 reduction='sum',
                 frame_length=16,
                 use_3d=False,
                 **kwargs):

        super(SMPLifyLoss, self).__init__()

        self.use_joints_conf = use_joints_conf
        self.angle_prior = angle_prior
        
        self.use_3d = use_3d

        self.robustifier = module_utils.GMoF(rho=rho)
        self.rho = rho

        self.body_pose_prior = body_pose_prior

        self.shape_prior = shape_prior

        self.fix_shape = kwargs.get('fix_shape')

        self.frames = frame_length
        self.index = [[j for j in range(i-3, i+3) if j >=0 and j < self.frames] for i in range(self.frames)]

        self.interpenetration = interpenetration
        if self.interpenetration:
            from sdf import SDF
            self.sdf = SDF()
            # self.search_tree = search_tree
            # self.tri_filtering_module = tri_filtering_module
            # self.pen_distance = pen_distance

        self.register_buffer('data_weight',
                             torch.tensor(data_weight, dtype=dtype))
        self.register_buffer('body_pose_weight',
                             torch.tensor(body_pose_weight, dtype=dtype))
        self.register_buffer('shape_weight',
                             torch.tensor(shape_weight, dtype=dtype))
        self.register_buffer('kinetic_weight',
                             torch.tensor(kinetic_weight, dtype=dtype))
        self.register_buffer('bending_prior_weight',
                             torch.tensor(bending_prior_weight, dtype=dtype))
        if self.interpenetration:
            self.register_buffer('coll_loss_weight',
                                 torch.tensor(coll_loss_weight, dtype=dtype))

    def reset_loss_weights(self, loss_weight_dict):
        for key in loss_weight_dict:
            if hasattr(self, key):
                weight_tensor = getattr(self, key)
                if 'torch.Tensor' in str(type(loss_weight_dict[key])):
                    weight_tensor = loss_weight_dict[key].clone().detach()
                else:
                    weight_tensor = torch.tensor(loss_weight_dict[key],
                                                 dtype=weight_tensor.dtype,
                                                 device=weight_tensor.device)
                setattr(self, key, weight_tensor)

    def get_bounding_boxes(self, vertices):
        num_people = vertices.shape[0]
        boxes = torch.zeros(num_people, 2, 3, device=vertices.device)
        for i in range(num_people):
            boxes[i, 0, :] = vertices[i].min(dim=0)[0]
            boxes[i, 1, :] = vertices[i].max(dim=0)[0]
        return boxes

    def forward(self, body_model_outputs, flags, camera, gt_jointss, joints_confs,
                body_model_faces, joint_weights,
                use_vposer=False, pose_embeddings=None, use_motionprior=False,
                gt_joints3d=None, joints3d_conf=None, body_models=None,
                **kwargs):

        start = time.time()
        loss_dict = {'reproj':0., 'prior':0., 'shape':0., 'kinetic':0.}
        total_loss = 0
        frames, num_joints = body_model_outputs[0].joints.shape[:2]
        num_person = len(body_model_outputs)
        num_views = len(camera)
        for body_model_output, gt_joints, joints_conf, pose_embedding, body_model, flag in zip(body_model_outputs, gt_jointss, joints_confs, pose_embeddings, body_models, flags.permute((1,0))):
        
            # project model to each view
            joints = body_model_output.joints.reshape(1, -1, 3)
            projected_joints = []
            for cam in camera:
                projected_joints_ = cam(joints)
                projected_joints.append(projected_joints_)
            
            weights = []
            for conf in joints_conf:
                # Calculate the weights for each joints
                weights_ = (joint_weights.repeat(conf.size(0), 1) * conf
                        if self.use_joints_conf else
                        joint_weights).reshape(1,-1,1)
                weights.append(weights_.repeat(1,1,2))

            # Calculate the distance of the projected joints from
            # the ground truth 2D detections
            gt_joints = gt_joints.reshape(len(projected_joints), -1, 2)
            joint_loss = 0.
            for i in range(len(gt_joints)):
                joint_diff = self.robustifier(gt_joints[i] - projected_joints[i])
                joint_loss_ = (torch.sum(weights[i] ** 2 * joint_diff) *
                            self.data_weight ** 2)
                joint_loss += joint_loss_
                if torch.isnan(joint_loss).sum():
                    print(1)
            joint_loss = joint_loss / num_views / frames
            loss_dict['reproj'] += int(joint_loss.data)

            # 3d loss
            joints3d_loss = 0.
            if self.use_3d:
                joints3d_conf = joints3d_conf.unsqueeze(dim=-1)
                diff3d = self.robustifier(gt_joints3d - body_model_output.joints)
                joints3d_loss = (torch.sum(joints3d_conf ** 2 * diff3d) *
                            self.data_weight ** 2)

            # Calculate the loss from the Pose prior
            if use_vposer:
                pprior_loss = (pose_embedding.pow(2).sum() *
                            self.body_pose_weight ** 2)
                #
                # index = [[j for j in range(i-3, i+3) if j >=0 and j < pose_embedding.size(0)] for i in range(pose_embedding.size(0))]
                for i, ind in enumerate(self.index):
                    mean_embedding = torch.mean(pose_embedding[ind], dim=0).reshape(1, -1)
                    diff_embedding = ((pose_embedding[i] - mean_embedding).pow(2).sum() * self.body_pose_weight ** 2)
                    pprior_loss += diff_embedding
            elif use_motionprior:
                # print(pose_embedding.max())
                # pprior_loss = 0
                # pprior_loss = (pose_embedding.pow(2).sum() *
                #                self.body_pose_weight ** 2)
                pprior_loss = (pose_embedding.pow(2).sum() *
                            (self.body_pose_weight / pose_embedding.size(0)) ** 2)
                # mean_embedding = torch.mean(pose_embedding, dim=0).reshape(1, -1).repeat(pose_embedding.size(0), 1)
                # diff_embedding = ((pose_embedding - mean_embedding).pow(2).sum() * self.body_pose_weight ** 2)
                # pprior_loss += diff_embedding
            else:
                pprior_loss = torch.sum(self.body_pose_prior(
                    body_model_output.body_pose,
                    body_model_output.betas)) * self.body_pose_weight ** 2
                if float(pprior_loss) > 5e4:
                    pprior_loss = 0.
                pprior_loss += body_model_output.body_pose.pow(2).sum() * (self.body_pose_weight * 4) ** 2
            pprior_loss = pprior_loss / frames
            loss_dict['prior'] += int(pprior_loss.data)

            shape_loss = 0.
            if not self.fix_shape:
                shape_loss = torch.sum(self.shape_prior(
                    body_model_output.betas)) * self.shape_weight ** 2
                # if use_vposer or use_motionprior:
                #     shape_loss = shape_loss * pose_embedding.size(0)
            loss_dict['shape'] += int(shape_loss.data)
            # Calculate the prior over the joint rotations. This a heuristic used
            # to prevent extreme rotation of the elbows and knees
            body_pose = body_model_output.full_pose[:, 3:66]

            angle_prior_loss = 0.
            # angle_prior_loss = torch.sum(
            #     self.angle_prior(body_pose)) * self.bending_prior_weight
            # if float(angle_prior_loss) > 1e4 and not use_vposer:
            #     angle_prior_loss = 0.

            global_smooth_loss = 0.
            # index = [[j for j in range(i-3, i+3) if j >=0 and j < pose_embedding.size(0)] for i in range(pose_embedding.size(0))]
            # for i, ind in enumerate(index):
            #     mean_trans = torch.mean(body_model.transl[ind], dim=0).reshape(1, -1)
            #     diff_trans = ((body_model.transl[i] - mean_trans).pow(2).sum() * 10 ** 2)
            #     global_smooth_loss += diff_trans
            #     mean_rots = torch.mean(body_model.global_orient[ind], dim=0).reshape(1, -1)
            #     diff_rots = ((body_model.global_orient[i] - mean_rots).pow(2).sum() * 10 ** 2)
            #     global_smooth_loss += diff_rots

            kinetic_loss = 0.
            velocity = body_model_output.joints[1:] -  body_model_output.joints[:-1]
            velocity = torch.norm(velocity, dim=-1).pow(2)
            kinetic_loss = velocity.sum() * self.kinetic_weight
            kinetic_loss += torch.norm(body_model_output.joints[0] -  body_model_output.joints[1], dim=-1).sum() * self.kinetic_weight
            kinetic_loss = kinetic_loss / frames
            loss_dict['kinetic'] += int(kinetic_loss.data)

            pen_loss = 0.0
            # Calculate the loss due to interpenetration
            if (self.interpenetration and self.coll_loss_weight.item() > 0):
                vertices = body_model_output.vertices
                boxes = self.get_bounding_boxes(vertices)
                boxes_center = boxes.mean(dim=1).unsqueeze(dim=1)
                boxes_scale = (1+0.2) * 0.5*(boxes[:,1] - boxes[:,0]).max(dim=-1)[0][:,None,None]

                with torch.no_grad():
                    vertices_centered = vertices - boxes_center
                    vertices_centered_scaled = vertices_centered / boxes_scale
                    assert(vertices_centered_scaled.min() >= -1)
                    assert(vertices_centered_scaled.max() <= 1)
                    assert(vertices.shape[0] == 1)
                    phi = self.sdf(body_model_faces.reshape(1, -1, 3).to(torch.int32), vertices_centered_scaled, grid_size=128)
                    assert(phi.min() >= 0)

                valid_people = vertices.shape[0]
                # Convert vertices to the format expected by grid_sample
                for i in range(valid_people):
                    weights = torch.ones(valid_people, 1, device=vertices.device)
                    # weights[i,0] = 0.
                    # Change coordinate system to local coordinate system of each person
                    vertices_local = (vertices - boxes_center[i].unsqueeze(dim=0)) / boxes_scale[i].unsqueeze(dim=0)
                    vertices_grid = vertices_local.view(1,-1,1,1,3)
                    # Sample from the phi grid
                    phi_val = nn.functional.grid_sample(phi[i][None, None], vertices_grid).view(valid_people, -1)
                    # ignore the phi values for the i-th shape
                    cur_loss = weights * phi_val
                    # if self.debugging:
                    #     import ipdb;ipdb.set_trace()
                    # # robustifier
                    # if self.robustifier:
                    #     frac = (cur_loss / self.robustifier) ** 2
                    #     cur_loss = frac / (frac + 1)

                    pen_loss += (self.coll_loss_weight * cur_loss.sum() / valid_people) ** 2
                    # print(pen_loss)
            # if (self.interpenetration and self.coll_loss_weight.item() > 0):
            #     batch_size = projected_joints.shape[0]
            #     triangles = torch.index_select(
            #         body_model_output.vertices, 1,
            #         body_model_faces).view(batch_size, -1, 3, 3)

            #     with torch.no_grad():
            #         collision_idxs = self.search_tree(triangles)

            #     # Remove unwanted collisions
            #     if self.tri_filtering_module is not None:
            #         collision_idxs = self.tri_filtering_module(collision_idxs)

            #     if collision_idxs.ge(0).sum().item() > 0:
            #         pen_loss = torch.sum(
            #             self.coll_loss_weight *
            #             self.pen_distance(triangles, collision_idxs))

            total_loss += (joint_loss + joints3d_loss + pprior_loss + shape_loss +
                        angle_prior_loss + pen_loss + global_smooth_loss + kinetic_loss)
        end = time.time()
        loss_dict['time'] = (end - start)
        total_loss = total_loss / len(body_model_outputs)
        for key in loss_dict:
            loss_dict[key] = loss_dict[key] / len(body_model_outputs)
        return total_loss, loss_dict