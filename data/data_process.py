import gc

import numpy as np
import pickle
import torch
import json
import os
import cv2
from utils.smpl_torch_batch import SMPLModel

def read_data():
    with open('data/3DOH/train.pkl', 'rb') as file:
        train_info = pickle.load(file)
    print('successfully read train.pkl!')

    with open('data/Eval_DoubleB/keypoints/doubleB/Camera00/00001_keypoints.json', 'r') as file:
        keypoints = json.load(file)
    print('successfully read 00001_keypoints.json!!')

    J_reg_halpe = np.load('data/J_regressor_halpe.npy')
    print('successfully read J_regressor_halpe.npy!!')

def get_keypoints():
    # PREPARE
    with open('data/3DOH/train.pkl', 'rb') as file:
        train_infos = pickle.load(file)
    print('successfully read train.pkl!')
    train_info = train_infos[0]
    jreg = np.load('data/J_regressor_halpe.npy').astype(np.float32)
    vertices = []
    for i in range(1200):
        vertices.append(np.load(f'data/3DOH/smpl_vertices/{i}.npy').astype(np.float32))
    # ITERATE
    for cam_idx, cam in enumerate(train_info):
        save_dir = f'data/3DOH/keypoints/motion0/Camera{cam_idx:02}'
        os.makedirs(save_dir, exist_ok=True)
        for pose_idx, pose in enumerate(cam):
            # GET VERTICE
            vertice = vertices[pose_idx]
            j3d = np.dot(jreg, vertice)
            # PROJECT
            pose_info = pose['0']
            extri = pose_info['extri']
            intri = pose_info['intri']
            K = intri
            R = extri[:3, :3]
            T = extri[:3, 3]
            j2d = np.dot(j3d, R.T) + T
            j2d = np.dot(j2d, K.T)
            j2d = j2d[:, :2] / j2d[:, 2:]
            j2d = list(np.column_stack((j2d, np.ones(j2d.shape[0]))).reshape(-1))
            # FORMAT
            people = []
            people_dict = {}
            people_dict['pose_keypoints_2d'] = j2d
            people.append(people_dict)
            keypoints = {
                'people': [{'pose_keypoints_2d': j2d}],
                'version': 1.1
            }
            # SAVE
            save_path = os.path.join(save_dir, f'{pose_idx:05}_keypoints.json')
            with open(save_path, 'w') as file:
                json.dump(keypoints, file, indent=4)

def get_cam():
    # PREPARE
    save_dir = 'data/OcMotion/camparams/0019'
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'camparams.txt')
    with open('data/OcMotion/test.pkl', 'rb') as file:
        train_infos = pickle.load(file)
    train_info = train_infos[3]

    # WRITE
    with open(save_path, 'w') as file:
        for cam_idx, cam in enumerate(train_info):
            cam_info = cam[0]['0']
            intri = cam_info['intri']
            extri = cam_info['extri']
            file.write(f"{cam_idx}\n")

            for row in intri:
                file.write(" ".join(map(str, row)) + "\n")

            file.write(f"0 0 \n")

            for row in extri[:3]:
                file.write(" ".join(map(str, row)) + "\n")

            file.write(f"\n")

def keypoints_vis():
    path = 'data/3DOH/keypoints/motion0/Camera00/00999_keypoints.json'
    with open(path, 'rb') as file:
        keypoints_info = json.load(file)
    keypoints = np.array(keypoints_info['people'][0]['pose_keypoints_2d'], dtype=np.float32).reshape(-1, 3)
    keypoints = keypoints[:, :2].astype(np.int32)

    h = 1536
    w = 2048
    image = np.zeros((h, w, 3), dtype=np.uint8)
    for point in keypoints:
        cv2.circle(image, tuple(point), 5, (0, 255, 0), -1)
    cv2.imshow('Keypoints', image)
    cv2.waitKey(0)  # 等待按键
    cv2.destroyAllWindows()  # 关闭窗口

def vertices_vis():
    # get camera info
    with open('data/3DOH/train.pkl', 'rb') as file:
        train_infos = pickle.load(file)
    print('successfully read train.pkl!')
    train_info = train_infos[0]
    cam_info = train_info[0][0]['0']
    intri = cam_info['intri']
    extri = cam_info['extri']
    K = intri
    R = extri[:3, :3]
    T = extri[:3, 3]

    # load smpl model
    smpl = SMPLModel(device=torch.device('cpu'), model_path='assets/SMPL_NEUTRAL.pkl')

    # make ouput dir
    os.makedirs('output/smpl_proj', exist_ok=True)

    # iterate
    for idx in range(1200):
        path = f'output/results/motion0/{idx:05d}.pkl'
        with open(path, 'rb') as file:
            smpl_params = pickle.load(file)
        sp = smpl_params['person00']
        s = torch.from_numpy(sp['betas'].astype(np.float32).reshape(-1, 10))
        p = torch.from_numpy(sp['pose'].astype(np.float32).reshape(-1, 72))
        t = torch.from_numpy(sp['transl'].astype(np.float32).reshape(-1, 3))
        v, j = smpl(s, p, t)
        v = v.detach().cpu().numpy().reshape(-1, 3).astype(np.float32)
        j = j.detach().cpu().numpy().astype(np.float32)

        projected = np.dot(v, R.T) + T
        projected = np.dot(projected, K.T)
        projected = projected[:, :2] / projected[:, 2:]
        projected = projected.astype(np.int32)

        h = 1536
        w = 2048
        image = np.zeros((h, w, 3), dtype=np.uint8)
        for point in projected:
            cv2.circle(image, tuple(point), 5, (0, 255, 0), -1)
        cv2.imwrite(f"output/smpl_proj/{idx:05d}.png", cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        # cv2.imshow('projected smpl vertices', image)
        # cv2.waitKey(0)  # 等待按键
        # cv2.destroyAllWindows()  # 关闭窗口

def get_smpl_params_vertices():
    # PREPARE
    os.makedirs('output/smpl_vertices', exist_ok=True)
    os.makedirs('output/smpl_params', exist_ok=True)

    # ITERATION
    for i in range(1200):
        path = f'output/results/motion0/{i:05d}.pkl'
        with open(path, 'rb') as file:
            smpl_params = pickle.load(file)
        sp = smpl_params['person00']
        s = torch.from_numpy(sp['betas'].astype(np.float32).reshape(-1, 10))
        p = torch.from_numpy(sp['pose'].astype(np.float32).reshape(-1, 72))
        t = torch.from_numpy(sp['transl'].astype(np.float32).reshape(-1, 3))
        smpl = SMPLModel(device=torch.device('cpu'), model_path='assets/SMPL_NEUTRAL.pkl')
        v, j = smpl(s, p, t)
        v = v.detach().cpu().numpy().reshape(-1, 3).astype(np.float32)
        j = j.detach().cpu().numpy().astype(np.float32)
        sp_save = {}
        sp_save['shapes'] = sp['betas'].astype(np.float32).reshape(-1, 10)
        sp_save['poses'] = sp['pose'].astype(np.float32).reshape(-1, 72)
        sp_save['Rh'] = sp_save['poses'][0][:3].reshape(-1, 3)
        sp_save['Th'] = sp['transl'].astype(np.float32).reshape(-1, 3)

        # SAVE
        np.save(f'output/smpl_vertices/{i}.npy', v)
        np.save(f'output/smpl_params/{i}.npy', sp_save)

def del_image():
    for idx in range(6):
        img_dir = f'data/3DOH/images/motion16/Camera{idx:02d}'
        imgs = os.listdir(img_dir)
        for img in imgs:
            img_path = os.path.join(img_dir, img)
            img_idx = int(img[:-4])
            if img_idx < 400 or img_idx > 499:
                os.remove(img_path)

def get_annots():
    # 读取plk文件，即smpl_params参数 39 * 6 * 1200
    pkl_path = 'data/OcMotion/test.pkl'
    with open(pkl_path, 'rb') as file:
        params = pickle.load(file)

    params = params[0]

    # 创建字典
    annots = {}
    cams = {}

    # 将cam info 存入字典
    K = []
    D = []
    dt = np.zeros((1, 5))
    R = []
    T = []
    for i in range(6):
        intri = params[i][0]['0']['intri']
        extri = params[i][0]['0']['extri']
        K.append(intri)
        R.append(extri[:3, :3])
        T.append((extri[:3, 3]).reshape(3, 1))
        D.append(dt)
    cams['K'] = K
    cams['R'] = R
    cams['T'] = T
    cams['D'] = D
    annots['cams'] = cams

    # 将imgs info 存入字典
    imgs = []
    for i in range(1200):
        img = []
        sub_ims = {}
        for j in range(6):
            path = os.path.join("images", "0013", "Camera{:02d}".format(j), "{:05d}.jpg".format(i))
            img.append(path)
        sub_ims['ims'] = img
        imgs.append(sub_ims)
    annots['ims'] = imgs
    np.save("data/OcMotion/annots.npy", annots)

    print(annots)

if __name__ == '__main__':
    # read_data()
    # get_keypoints()
    # get_cam()
    # keypoints_vis()
    # vertices_vis()
    # get_smpl_params_vertices()
    # del_image()
    get_annots()