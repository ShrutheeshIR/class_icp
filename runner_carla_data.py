import numpy as np
import glob
import os
from file_utils import get_point_clouds
from class_icp import icp, fps_downsample
from plot_utils import display_registered_point_clouds
from eval_utils import compute_rre, compute_rte
import json
from tqdm import tqdm

def runner(dir_path):

    l1_directories = sorted(glob.glob(os.path.join(dir_path, 'lidar_1*')))
    l2_directories = [l1_dir.replace('lidar_1', 'lidar_2') for l1_dir in l1_directories]

    rte_error_all_dir = 0
    rre_error_all_dir = 0
    total_files = 0

    for (l1_directory, l2_directory) in tqdm(zip(l1_directories, l2_directories)):
        T_gt = np.linalg.inv(np.loadtxt(os.path.join(l1_directory, "l1l2_transform.txt")))


        l1_filenames = sorted(glob.glob(l1_directory + '/*.npy'))
        l1_filenames = [os.path.basename(x) for x in l1_filenames]
        l1_basenames = np.array([float(l1_name.split('.')[0]) for l1_name in l1_filenames])

        l2_filenames = sorted(glob.glob(l2_directory + '/*.npy'))
        l2_filenames = [os.path.basename(x) for x in l2_filenames]
        l2_basenames = np.array([float(l2_name.split('.')[0]) for l2_name in l2_filenames])

        # do_semantic = False
        display = False

        total_rre_error = 0.0
        total_rte_error = 0.0

        for do_semantic in [False, True]:

            for index, filename in tqdm(enumerate(l1_filenames), leave=False):
                if index > 25:
                    break
                l1_basename = l1_basenames[index]
                l2_basename_index = np.argmin(np.abs(l2_basenames - l1_basename))

                l2_filename = l2_filenames[l2_basename_index]

                lidar_1_pc_path = os.path.join(l1_directory, filename)
                lidar_2_pc_path = os.path.join(l2_directory, l2_filename)


                source_pts_og, target_pts_og, source_pts_classes, target_pts_classes = get_point_clouds(lidar_1_pc_path, lidar_2_pc_path, do_semantic = do_semantic, display = display)

                source_pts = source_pts_og.copy()
                target_pts = target_pts_og.copy()

                # print(T_gt)
                # display_registered_point_clouds(source_pts_og.T, target_pts_og.T, T_gt)

                T0 = np.eye(4)



                # T0 = initialize_ICP(source_pts, target_pts)

                T = icp(source_pts, target_pts, source_pts_classes, target_pts_classes, T0 = T0)
                # print(T)
                # print(T_gt)
                total_rre_error += compute_rre(T[:3, :3], T_gt[:3, :3])
                total_rte_error += compute_rte(T[:3, 3], T_gt[:3, 3])

                if display:
                    display_registered_point_clouds(source_pts_og.T, target_pts_og.T, T)
                # print(total_rre_error, total_rte_error, index)
        
            total_metrics = {'rte' : total_rte_error / float(index+1), 'rre' : total_rre_error / float(index+1), 'files' : (index + 1)}
            with open(os.path.join(l1_directory, 'metrics_%s.json'%do_semantic), 'w', encoding='utf-8') as f:
                json.dump(total_metrics, f, ensure_ascii=False, indent=4)        

            rte_error_all_dir += total_rte_error
            rre_error_all_dir += total_rre_error
            total_files += (index + 1)
        
        overall_metrics = {'rte' : rte_error_all_dir / float(total_files), 'rre' : rre_error_all_dir / float(total_files), 'files' : (total_files)}
        with open(os.path.join(dir_path, 'metrics_%s.json'%do_semantic), 'w', encoding='utf-8') as f:
            json.dump(overall_metrics, f, ensure_ascii=False, indent=4)        


if __name__ == '__main__':
    runner("data_dir")