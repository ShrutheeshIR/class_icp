import numpy as np
import glob
import os
from file_utils import get_point_clouds
from class_icp import icp, fps_downsample
from plot_utils import display_registered_point_clouds
from eval_utils import compute_rre, compute_rte

def runner(dir_path):

    # l1_filenames = os.listdir(os.path.join(dir_path, "lidar_1")).sort(key=lambda x:float(x.split('.')))

    l1_filenames = sorted(glob.glob(os.path.join(dir_path, 'lidar_1/*')))

    l1_filenames = [os.path.basename(x) for x in l1_filenames]
    l1_basenames = np.array([float(l1_name.split('.')[0]) for l1_name in l1_filenames])

    l2_filenames = sorted(glob.glob(os.path.join(dir_path, 'lidar_2/*')))
    l2_filenames = [os.path.basename(x) for x in l2_filenames]
    l2_basenames = np.array([float(l2_name.split('.')[0]) for l2_name in l2_filenames])

    do_semantic = False
    display = True

    total_rre_error = 0.0
    total_rte_error = 0.0

    T_gt = np.loadtxt("l1l2_transform.txt")
    T_gt = np.eye(4)


    for index, filename in enumerate(l1_filenames):

        l1_basename = l1_basenames[index]
        l2_basename_index = np.argmin(np.abs(l2_basenames - l1_basename))

        l2_filename = l2_filenames[l2_basename_index]

        # print(filename, l2_filename)

        lidar_1_pc_path = os.path.join(dir_path, "lidar_1", filename)
        lidar_2_pc_path = os.path.join(dir_path, "lidar_2", l2_filename)


        source_pts_og, target_pts_og, source_pts_classes, target_pts_classes = get_point_clouds(lidar_1_pc_path, lidar_2_pc_path, do_semantic = do_semantic, display = display)

        source_pts = source_pts_og.copy()
        target_pts = target_pts_og.copy()

        # print(T_gt)
        display_registered_point_clouds(source_pts_og.T, target_pts_og.T, T_gt)

        # source_pts = fps_downsample(source_pts_og, 20000)
        # target_pts = fps_downsample(target_pts_og, 20000)

        T0 = np.eye(4)



        # T0 = initialize_ICP(source_pts, target_pts)

        T = icp(source_pts, target_pts, source_pts_classes, target_pts_classes, T0 = T0)
        print(T)
        print(T_gt)
        total_rre_error += compute_rre(T[:3, :3], T_gt[:3, :3])
        total_rte_error += compute_rte(T[:3, 3], T_gt[:3, 3])

        if display:
            display_registered_point_clouds(source_pts_og.T, target_pts_og.T, T)        

        print(total_rre_error, total_rte_error, index)

if __name__ == '__main__':
    runner("test_dir")