'''
 @FileName    : eval.py
 @EditTime    : 2021-07-07 19:43:10
 @Author      : Buzhen Huang
 @Email       : hbz@seu.edu.cn
 @Description : 
'''

import os
import numpy as np
import sys
sys.path.append('./')
from core.utils.eval_utils import *
import torch
from core import smplx


datasets = dict(
            Human36M=Eval_Human36M,
            OcMotion=Eval_OcMotion,
            MPI3DHP=Eval_MPI3DHP,
            Shelf=Eval_Campus,
            MHHI=Eval_MHHI,
            Campus=Eval_Campus,
        )

if __name__ == '__main__':
    
    eval_set = 'OcMotion'
    align_cam = False
    results = 'output'
    dataset_dir = 'data'
    
    model_params = dict(model_path='models/smpl/SMPL_NEUTRAL.pkl',
                        joint_mapper=None,
                        create_global_orient=True,
                        create_body_pose=True,
                        create_betas=True,
                        create_left_hand_pose=False,
                        create_right_hand_pose=False,
                        create_expression=False,
                        create_jaw_pose=False,
                        create_leye_pose=False,
                        create_reye_pose=False,
                        create_transl=True, #set transl in multi-view task  --Buzhen Huang 07/31/2019
                        create_scale=True,
                        batch_size=1,
                        dtype=torch.float32)

    smpl_model = smplx.create_scale(gender='neutral', **model_params)

    evaltool = HumanEval(eval_set, smpl=smpl_model)
    dataset = datasets[eval_set](dataset_dir, smpl=smpl_model, evaltool=evaltool, align_cam=align_cam, dataset_name=eval_set)

    vertex_error, error, error_pa, accel = dataset.evaluate(results)
    # vertex_error, error, error_pa, accel = dataset.evaluate_wo_order(results)

    print("Surface: %f, MPJPE: %f, PA-MPJPE: %f, Accel: %f" %(vertex_error, error, error_pa, accel))

    pass











