import os
import pickle
import numpy as np
import cv2
from core.utils.render import Renderer


def project_to_img(joints, verts, faces, gt_joints, camera, image_path, img_folder, viz=False, path=None):
    exp = 1
    if len(verts) < 1:
        return
    for v, (cam, gt_joint_ids, img_path) in enumerate(zip(camera, gt_joints, image_path)):
        # if v > 0 and exp:
        #     break
        intri = np.eye(3)
        rot = cam.rotation.detach().cpu().numpy()
        trans = cam.translation.detach().cpu().numpy()
        intri[0][0] = cam.focal_length_x.detach().cpu().numpy()
        intri[1][1] = cam.focal_length_y.detach().cpu().numpy()
        intri[:2, 2] = cam.center.detach().cpu().numpy()
        rot_mat = cv2.Rodrigues(rot)[0]

        img = cv2.imread(os.path.join(img_folder, img_path))
        render = Renderer(resolution=(img.shape[1], img.shape[0]))
        img = render.render_multiperson(verts, faces, rot_mat.copy(), trans.copy(), intri.copy(), img.copy(),
                                        viz=False)

        img_out_file = os.path.join(path, img_path)
        if not os.path.exists(os.path.dirname(img_out_file)):
            os.makedirs(os.path.dirname(img_out_file))
        cv2.imwrite(img_out_file, img)
        render.renderer.delete()
        del render

def img2video():
    for idx in range(1):
        # 图像文件夹路径
        # image_folder = f'output/images/motion16/Camera{idx:02d}'  # 替换为你的图像文件夹路径
        # image_folder = 'output/images/motion16/Camera04'  # 替换为你的图像文件夹路径
        image_folder = 'output/render/gs'
        # video_name = f'origin{idx:02d}.avi'  # 输出视频文件名
        # video_name = 'addgs04.avi'  # 输出视频文件名
        video_name = 'gs.avi'  # 输出视频文件名

        # 获取图像文件名并排序
        images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
        images.sort()

        # 读取第一张图像以获取视频参数
        first_image = cv2.imread(os.path.join(image_folder, images[0]))
        height, width, layers = first_image.shape

        # 创建 VideoWriter 对象
        fourcc = cv2.VideoWriter_fourcc(*'XVID')  # 选择编码方式
        video = cv2.VideoWriter(video_name, fourcc, 15, (width, height))

        # 将图像写入视频
        for image in images:
            img_path = os.path.join(image_folder, image)
            video.write(cv2.imread(img_path))

        # 释放 VideoWriter 对象
        video.release()
        cv2.destroyAllWindows()

        print(f'第{idx}个视频已生成：{video_name}')

if __name__ == '__main__':
    # for i in range(100):
    #     path = f'output/3DOH/motion16/render_data/{i:03d}'
    #     with open(os.path.join(path, 'joints.pkl'), 'rb') as file:
    #         joints = pickle.load(file)
    #     with open(os.path.join(path, 'meshes.pkl'), 'rb') as file:
    #         meshes = pickle.load(file)
    #     with open(os.path.join(path, 'faces.pkl'), 'rb') as file:
    #         faces = pickle.load(file)
    #     with open(os.path.join(path, 'keyp_p.pkl'), 'rb') as file:
    #         keyp_p = pickle.load(file)
    #     with open(os.path.join(path, 'camera.pkl'), 'rb') as file:
    #         camera = pickle.load(file)
    #     with open(os.path.join(path, 'img_p.pkl'), 'rb') as file:
    #         img_p = pickle.load(file)
    #     dataset_obj_img_folder = 'data/3DOH/images'
    #     setting_img_folder = 'output/images'
    #     project_to_img(joints, meshes, faces, keyp_p, camera, img_p, dataset_obj_img_folder, viz=False, path=setting_img_folder)

    img2video()