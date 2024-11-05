'''
gaussian_loss.py is used to compute the loss of gs
'''

import torch
import torch.nn as nn
import numpy as np
import os
import lpips
loss_fn_vgg = lpips.LPIPS(net='vgg').to(torch.device('cuda', torch.cuda.current_device()))
import cv2
import lpips
from random import randint
from core.gaussian.utils.loss_utils import l1_loss, l2_loss, ssim
from core.gaussian.gaussian_renderer import render
loss_fn_vgg = lpips.LPIPS(net='vgg').to(torch.device('cuda', torch.cuda.current_device()))
from tqdm import tqdm
import time
from core.gaussian.scene import Scene

class GS3DLoss(nn.Module):
    def __init__(self,):
        super(GS3DLoss, self).__init__()

    def forward(self, opt, pipe, dataset_gs, gaussians, setting, dataset_obj, iterations, mode):
        # initialize
        scene = Scene(dataset_gs, gaussians, setting, dataset_obj, mode)
        gaussians.training_setup(opt)
        bg_color = [1, 1, 1] if dataset_gs.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        viewpoint_stack = None
        first_iter = 0
        progress_bar = tqdm(range(first_iter, iterations), desc="Training progress")
        first_iter += 1
        elapsed_time = 0

        # interate
        for iteration in range(first_iter, iterations + 1):
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
            viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))

            # 渲染
            render_pkg = render(viewpoint_cam, gaussians, pipe, background)
            image, alpha, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg[
                "render_alpha"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

            # 保存渲染图片
            os.makedirs('output/3DOH/motion18/train_img', exist_ok=True)
            check_image = (image.permute(1, 2, 0).cpu().detach().numpy() * 255).astype(np.uint8)
            cv2.imwrite(f'output/3DOH/motion18/train_img/{iteration}.png', check_image)

            gt_image = viewpoint_cam.original_image.cuda()
            bkgd_mask = viewpoint_cam.bkgd_mask.cuda()
            bound_mask = viewpoint_cam.bound_mask.cuda()
            Ll1 = l1_loss(image.permute(1, 2, 0)[bound_mask[0] == 1], gt_image.permute(1, 2, 0)[bound_mask[0] == 1])
            mask_loss = l2_loss(alpha[bound_mask == 1], bkgd_mask[bound_mask == 1])

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
                        gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent,
                                                    size_threshold,
                                                    kl_threshold=0.4, t_vertices=viewpoint_cam.big_pose_world_vertex,
                                                    iter=iteration)
                        # gaussians.densify_and_prune(opt.densify_grad_threshold, 0.01, scene.cameras_extent, 1)

                    if iteration % opt.opacity_reset_interval == 0 or (
                            dataset_gs.white_background and iteration == opt.densify_from_iter):
                        gaussians.reset_opacity()

                # 优化器更新参数
                if iteration < iterations:
                    gaussians.optimizer.step()
                    gaussians.optimizer.zero_grad(set_to_none=True)

                # 更新进度条
                if iteration % 10 == 0:
                    progress_bar.set_postfix({"loss": f"{loss.item()}"})
                    progress_bar.update(10)
                if iteration == iterations:
                    progress_bar.close()

                # 结束时间
                end_time = time.time()

                # 加上优化时间
                elapsed_time += (end_time - start_time)


        return loss