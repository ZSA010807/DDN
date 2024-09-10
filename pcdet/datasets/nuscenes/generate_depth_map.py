from os import path as osp
import os
import numpy as np
import pickle as pkl
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm

def remove_ego_points(points, center_radius=1.0):
    mask = ~((np.abs(points[:, 0]) < center_radius) & (np.abs(points[:, 1]) < center_radius))
    return points[mask]

def cart_to_hom(pts):
    pts_hom = np.hstack((pts, np.ones((pts.shape[0], 1), dtype=np.float32)))
    return pts_hom
def lidar_to_rect(pts_lidar, v2c):
    pts_lidar_hom = cart_to_hom(pts_lidar)
    pts_rect = np.dot(pts_lidar_hom, v2c.T)
    return pts_rect
def rect_to_img(pts_rect, p):
    pts_rect_hom = cart_to_hom(pts_rect)
    pts_2d_hom = np.dot(pts_rect_hom, p.T)
    # pts_rect_hom[:, 2][pts_rect_hom[:, 2] == 0] = 1e-9
    pts_img = (pts_2d_hom[:, 0:2].T / pts_rect_hom[:, 2]).T
    pts_rect_depth = pts_2d_hom[:, 2] - p.T[3,2]
    return pts_img, pts_rect_depth




def visualize_depth_map_with_colormap(depth_map):
    """
    使用颜色映射可视化深度图
    :param depth_map: 深度图 (height, width) 浮点数矩阵
    """
    plt.figure(figsize=(10, 8))

    # 显示深度图，使用 'plasma' 颜色映射
    plt.imshow(depth_map, cmap='plasma')

    # 添加颜色条
    plt.colorbar(label='Depth (meters)')

    # 显示图像
    plt.show()

def generate_map(u_coords, v_coords, depths, image_width, image_height, save_path=None):
    """
    生成类似于 KITTI 的稀疏深度图。

    参数：
    - u_coords (array-like): 点云投影到图像的 u（x）坐标。
    - v_coords (array-like): 点云投影到图像的 v（y）坐标。
    - depths (array-like): 点云在相机坐标系下的深度值（单位：米）。
    - image_width (int): 图像的宽度。
    - image_height (int): 图像的高度。
    - save_path (str, optional): 如果提供，则将深度图保存到指定路径。

    返回：
    - depth_map (numpy.ndarray): 生成的深度图。
    """
    # 初始化深度图，使用无穷大表示未填充的像素
    depth_map = np.full((image_height, image_width), np.inf, dtype=np.float32)

    # 将输入转换为 numpy 数组
    u = np.array(u_coords).astype(np.int32)
    v = np.array(v_coords).astype(np.int32)
    d = np.array(depths).astype(np.float32)

    # 过滤掉超出图像边界的点
    valid_mask = (u >= 0) & (u < image_width) & (v >= 0) & (v < image_height) & (d > 0)
    u = u[valid_mask]
    v = v[valid_mask]
    d = d[valid_mask]

    # 对每个像素，保留最小的深度值
    # 使用 numpy 的索引和聚合方法
    for i in range(len(u)):
        ui = u[i]
        vi = v[i]
        di = d[i]
        if di < depth_map[vi, ui]:
            depth_map[vi, ui] = di

    # 将无穷大的值设为 0，表示无效的深度
    depth_map[depth_map == np.inf] = 0

    # 如果需要保存为 KITTI 格式的 16 位 PNG（单位：毫米）
    if save_path:
        # 转换深度值范围，并转换为 16 位无符号整数
        depth_map_mm = (depth_map * 256).astype(np.uint16)
        cv2.imwrite(save_path, depth_map_mm)
def main():
    from pathlib import Path

    def count_files_in_directory(directory):
        # 使用 pathlib 列出文件夹中的所有文件
        file_count = len(list(Path(directory).glob('*')))

        return file_count

    # 示例使用
    directory = '/opt/data/private/work/code/CaDDN/data/nuscenes/v1.0-trainval/depth_2/depth_map_raw'
    print(f"文件数量: {count_files_in_directory(directory)}")

    info_path = '/opt/data/private/work/code/CaDDN/data/nuscenes/v1.0-trainval/mmdet3d_infos/nuscenes_infos_train.pkl'
    # depth_image = cv2.imread('/opt/data/private/work/code/CaDDN/data/nuscenes/v1.0-trainval/depth_2/depth_map_raw/00000.png', cv2.IMREAD_ANYDEPTH)
    # projected_depths = np.float32(depth_image / 256.0)
    # non_zero_count = np.count_nonzero(projected_depths)
    #
    # # 只选择深度图中非零的像素
    # non_zero_depths = projected_depths[projected_depths > 0]
    #
    # if non_zero_depths.size > 0:
    #     # 计算非零像素中的最小值和最大值
    #     min_value = np.min(non_zero_depths)
    #     max_value = np.max(non_zero_depths)
    # # visualize_depth_map_with_colormap(projected_depths)
    infos = pkl.load(open(info_path, 'rb'))
    im_height, im_width = (900, 1600)
    ROOT_DIR = (Path(__file__).resolve().parent / '../../../').resolve()
    lidar_dir = '/opt/data/private/work/code/CaDDN/data/nuscenes/v1.0-trainval/samples/LIDAR_TOP'
    file_path = ROOT_DIR / 'data' / 'nuscenes' / 'v1.0-trainval' / 'depth_2' / 'depth_map_raw'
    file_path.mkdir(parents=True, exist_ok=True)
    for i in tqdm(range(len(infos['infos'])), desc="Processing data", unit="file"):
        info = infos['infos'][i]
        sensor2lidar_rotation = info['cams']['CAM_FRONT']['sensor2lidar_rotation']  # 3*3
        sensor2lidar_translation = info['cams']['CAM_FRONT']['sensor2lidar_translation']  # 3,
        cam_intrinsic = info['cams']['CAM_FRONT']['cam_intrinsic']  # 3*3
        cam2img = np.hstack((cam_intrinsic, np.zeros((3, 1))))
        cam2lidar_4 = np.eye(4)
        cam2lidar = np.concatenate((sensor2lidar_rotation, sensor2lidar_translation.reshape(3, 1)), axis=1)
        cam2lidar_4[:3, :] = cam2lidar
        lidar2cam = np.linalg.inv(cam2lidar_4)[:3]
        lidarfile = os.path.basename(info['lidar_path'])
        lidarpath = osp.join(lidar_dir, lidarfile)
        points = np.fromfile(lidarpath, dtype=np.float32)
        points = points.reshape(-1, 5)
        points = points[:, :3]
        points = remove_ego_points(points)
        points_rect = lidar_to_rect(points, lidar2cam)
        pts_img, depth = rect_to_img(points_rect, cam2img)
        filename = f"{i:05}.png"
        save_path = file_path/filename
        generate_map(pts_img[:, 0], pts_img[:, 1], depth, im_width, im_height, save_path)

if __name__ == '__main__':
    main()