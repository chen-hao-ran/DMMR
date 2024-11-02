import os

import pyrender
import numpy as np
import imageio
from core.utils.module_utils import load_camera_para
import trimesh

# 加载 OBJ 文件
mesh = trimesh.load('output/meshes/motion0/00299_00.obj')

mesh.show()

# 创建场景
scene = pyrender.Scene()

# 将 mesh 添加到场景中
pyrender_mesh = pyrender.Mesh.from_trimesh(mesh)
scene.add(pyrender_mesh)

# 创建相机
extri, intri = load_camera_para('data/3DOH/camparams/motion0/camparams.txt')
v = 0
fx = intri[v][0][0]
fy = intri[v][1][1]
H = 1536
W = 2048
xfov = 2 * np.arctan(W / 2 / fx)
yfov = 2 * np.arctan(H / 2 / fy)

camera = pyrender.PerspectiveCamera(yfov=yfov)
cam_pose = extri[v]
scene.add(camera, pose=cam_pose)

# 创建光源
light = pyrender.DirectionalLight(color=np.ones(3), intensity=1.0)
scene.add(light, pose=cam_pose)

# 创建离屏渲染器
renderer = pyrender.OffscreenRenderer(W, H)

# 渲染场景
color, depth = renderer.render(scene)

# 保存渲染结果为图像
os.makedirs('output/render/smpl', exist_ok=True)
imageio.imwrite('output/render/smpl/00001.png', color)

# 清理资源
renderer.delete()
