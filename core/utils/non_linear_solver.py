'''
 @FileName    : non_linear_solver.py
 @EditTime    : 2021-12-13 15:44:28
 @Author      : Buzhen Huang
 @Email       : hbz@seu.edu.cn
 @Description : 
'''

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os
import time
import numpy as np
import torch
from tqdm import tqdm
from core.optimizers import optim_factory
from core.utils import fitting

import cv2
import lpips
from random import randint
from core.gaussian.utils.loss_utils import l1_loss, l2_loss, ssim
from core.gaussian.scene import Scene
from core.gaussian.scene.gaussian_model import GaussianModel
from core.gaussian.gaussian_renderer import render
loss_fn_vgg = lpips.LPIPS(net='vgg').to(torch.device('cuda', torch.cuda.current_device()))

def non_linear_solver(
                    setting,
                    data,
                    dataset_obj,
                    args_gs,
                    batch_size=1,
                    data_weights=None,
                    body_pose_prior_weights=None,
                    kinetic_weights=None,
                    shape_weights=None,
                    coll_loss_weights=None,
                    use_joints_conf=False,
                    rho=100,
                    interpenetration=False,
                    loss_type='smplify',
                    visualize=False,
                    use_vposer=True,
                    use_motionprior=False,
                    interactive=True,
                    use_cuda=True,
                    **kwargs):

    device = setting['device']
    dtype = setting['dtype']
    vposer = setting['vposer']
    keypoints = data['keypoints']
    flags = data['flags']
    joint_weights = setting['joints_weight']
    models = setting['model']
    cameras = setting['cameras']
    pose_embeddings = setting['pose_embedding']
    dataset_gs = args_gs['dataset']
    opt = args_gs['opt']
    pipe = args_gs['pipe']

    assert (len(data_weights) ==
            len(body_pose_prior_weights) and len(shape_weights) ==
            len(body_pose_prior_weights) and len(coll_loss_weights) ==
            len(body_pose_prior_weights)), "Number of weight must match"
    
    # Load 2D keypoints
    keypoints = torch.tensor(keypoints, dtype=dtype, device=device)
    flags = torch.tensor(flags, dtype=dtype, device=device)
    gt_joints = keypoints[...,:2]
    joints_conf = keypoints[...,2]

    # Weights used for the pose prior and the shape prior
    opt_weights_dict = {'data_weight': data_weights,
                        'body_pose_weight': body_pose_prior_weights,
                        'shape_weight': shape_weights,
                        'kinetic_weight': kinetic_weights}
    if interpenetration:
        opt_weights_dict['coll_loss_weight'] = coll_loss_weights

    # Get loss weights for each stage
    keys = opt_weights_dict.keys()
    opt_weights = [dict(zip(keys, vals)) for vals in
                   zip(*(opt_weights_dict[k] for k in keys
                         if opt_weights_dict[k] is not None))]
    for weight_list in opt_weights:
        for key in weight_list:
            weight_list[key] = torch.tensor(weight_list[key],
                                            device=device,
                                            dtype=dtype)

    # Create fitting loss
    loss = fitting.create_loss(loss_type=loss_type,
                               joint_weights=joint_weights,
                               rho=rho,
                               use_joints_conf=use_joints_conf,
                               vposer=vposer,
                               pose_embedding=pose_embeddings,
                               body_pose_prior=setting['body_pose_prior'],
                               shape_prior=setting['shape_prior'],
                               angle_prior=setting['angle_prior'],
                               interpenetration=interpenetration,
                               dtype=dtype,
                               frame_length=dataset_obj.frames,
                               **kwargs)
    loss = loss.to(device=device)

    monitor = fitting.FittingMonitor(
            batch_size=batch_size, visualize=visualize, **kwargs)

    # Step 1: Optimize the cameras and motions
    final_loss_val = 0
    opt_start = time.time()
    for opt_idx, curr_weights in enumerate(tqdm(opt_weights, desc='Stage')):
        # Load all parameters for optimization
        body_params = []
        for model, pose_embedding in zip(models, pose_embeddings):
            body_param = list(model.parameters())
            body_params += list(
            filter(lambda x: x.requires_grad, body_param))
            if vposer is not None and opt_idx in [1,2,3]:
                body_params.append(pose_embedding)
        final_params = list(
            filter(lambda x: x.requires_grad, body_params))
        if opt_idx in [0,2,3]:
            for cam in cameras:
                if cam.translation.requires_grad:
                    final_params.append(cam.translation)
                if cam.rotation.requires_grad:
                    final_params.append(cam.rotation)
        body_optimizer, body_create_graph = optim_factory.create_optimizer(
            final_params, **kwargs)
        body_optimizer.zero_grad()

        loss.reset_loss_weights(curr_weights)

        closure = monitor.create_fitting_closure(
            body_optimizer, models,
            camera=cameras, gt_joints=gt_joints,
            joints_conf=joints_conf,
            flags=flags,
            joint_weights=joint_weights,
            loss=loss, create_graph=body_create_graph,
            use_vposer=use_vposer, vposer=vposer,
            use_motionprior=use_motionprior,
            pose_embeddings=pose_embeddings,
            return_verts=True, return_full_pose=True)

        if interactive:
            if use_cuda and torch.cuda.is_available():
                torch.cuda.synchronize()
            stage_start = time.time()
        final_loss_val = monitor.run_fitting(
            body_optimizer,
            closure, final_params,
            models,
            pose_embeddings=pose_embeddings, vposer=vposer, cameras=cameras,
            use_vposer=use_vposer, use_motionprior=use_motionprior)

        if interactive:
            if use_cuda and torch.cuda.is_available():
                torch.cuda.synchronize()
            elapsed = time.time() - stage_start
            if interactive:
                tqdm.write('Stage {:03d} done after {:.4f} seconds'.format(
                    opt_idx, elapsed))

    ## Gaussian iteration
    # initialize
    gaussians = GaussianModel(dataset_gs.sh_degree, dataset_gs.smpl_type, dataset_gs.actor_gender)
    scene = Scene(dataset_gs, gaussians, setting, dataset_obj)
    gaussians.training_setup(opt)
    bg_color = [1, 1, 1] if dataset_gs.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)
    viewpoint_stack = None
    first_iter = 0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    elapsed_time = 0

    # interate
    for iteration in range(first_iter, opt.iterations + 1):
        iter_start.record()

        # 更新学习率
        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # 开始时间
        start_time = time.time()

        # 随机选择一帧
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

        # 渲染
        render_pkg = render(viewpoint_cam, gaussians, pipe, background, iteration)
        image, alpha, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["render_alpha"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        # 保存渲染图片
        os.makedirs('output/3DOH/motion0/train_img', exist_ok=True)
        check_image = (image.permute(1, 2, 0).cpu().detach().numpy() * 255).astype(np.uint8)
        cv2.imwrite(f'output/3DOH/motion0/train_img/{iteration}.png', check_image)

        gt_image = viewpoint_cam.original_image.cuda()
        bkgd_mask = viewpoint_cam.bkgd_mask.cuda()
        bound_mask = viewpoint_cam.bound_mask.cuda()
        Ll1 = l1_loss(image.permute(1,2,0)[bound_mask[0]==1], gt_image.permute(1,2,0)[bound_mask[0]==1])
        mask_loss = l2_loss(alpha[bound_mask==1], bkgd_mask[bound_mask==1])

        # crop the object region
        x, y, w, h = cv2.boundingRect(bound_mask[0].cpu().numpy().astype(np.uint8))
        img_pred = image[:, y:y + h, x:x + w].unsqueeze(0)
        img_gt = gt_image[:, y:y + h, x:x + w].unsqueeze(0)
        # ssim loss
        ssim_loss = ssim(img_pred, img_gt)
        # lipis loss
        lpips_loss = loss_fn_vgg(img_pred, img_gt).reshape(-1)

        loss = Ll1 + 0.1 * mask_loss + 0.01 * (1.0 - ssim_loss) + 0.01 * lpips_loss

        # 后反馈loss
        loss.backward()

        # 结束时间
        end_time = time.time()

        # 计算迭代时间
        elapsed_time += (end_time - start_time)

        # 记录结束
        iter_end.record()

        # 密度优化
        with torch.no_grad():
            # 开始时间
            start_time = time.time()

            # 密度优化
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter],
                                                                     radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold,
                                                kl_threshold=0.4, t_vertices=viewpoint_cam.big_pose_world_vertex,
                                                iter=iteration)
                    # gaussians.densify_and_prune(opt.densify_grad_threshold, 0.01, scene.cameras_extent, 1)

                if iteration % opt.opacity_reset_interval == 0 or (
                        dataset_gs.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # 优化器更新参数
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

            # 更新进度条
            if iteration % 10 == 0:
                progress_bar.set_postfix({"loss": f"{loss.item()}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # 结束时间
            end_time = time.time()

            # 加上优化时间
            elapsed_time += (end_time - start_time)

    if interactive:
        if use_cuda and torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.time() - opt_start
        tqdm.write(
            'Body fitting done after {:.4f} seconds'.format(elapsed))
        tqdm.write('Body final loss val = {:.5f}'.format(
            final_loss_val))

        result = {}
        for idx, (model, pose_embedding) in enumerate(zip(models, pose_embeddings)):
            # Get the result of the fitting process
            model_result = {key: val.detach().cpu().numpy()
                            for key, val in model.named_parameters()}
            model_result['loss'] = final_loss_val
            model_result['pose_embedding'] = pose_embedding
            result['person%02d' %idx] = model_result

        # Get the optimized cameras
        rots, trans, intris = [], [], []
        for cam in cameras:
            rots.append(cam.rotation.detach().cpu().numpy())
            trans.append(cam.translation.detach().cpu().numpy())
            intri = np.eye(3)
            intri[0][0] = cam.focal_length_x.detach().cpu().numpy()
            intri[1][1] = cam.focal_length_y.detach().cpu().numpy()
            intri[:2,2] = cam.center.detach().cpu().numpy()
            intris.append(intri)
        result['cam_rots'] = rots
        result['cam_trans'] = trans
        result['intris'] = intris 
    return result
