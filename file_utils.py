import numpy as np
import open3d as o3d
from plot_utils import colormap_and_plot

def get_point_clouds(source_filename, target_filename, do_semantic = False, display = False):
    source_ply = np.load(source_filename)
    target_ply = np.load(target_filename)


    source_pts =  np.asarray(source_ply).T[:3]
    target_pts = np.asarray(target_ply).T[:3]

    source_pts[1] *= -1
    target_pts[1] *= -1

    if do_semantic:
        source_class_pts = np.asarray(source_ply)[:, -1].astype(np.int32)
        target_class_pts = np.asarray(target_ply)[:, -1].astype(np.int32)
    else:
        source_class_pts = np.zeros(len(source_pts.T)).astype(np.int32)
        target_class_pts = np.zeros(len(target_pts.T)).astype(np.int32)

    if display:
        src_color = np.ones(source_ply[:, :3].shape) * np.array([0.95, 0, 0]).reshape((1,3))
        target_color = np.ones(target_ply[:, :3].shape) * np.array([0, 0.95, 0]).reshape((1,3))

        pcd1 = o3d.geometry.PointCloud()
        pcd1.points = o3d.utility.Vector3dVector(source_ply[:, :3])
        pcd1.colors = o3d.utility.Vector3dVector(src_color)

        pcd2 = o3d.geometry.PointCloud()
        pcd2.points = o3d.utility.Vector3dVector(target_ply[:, :3])
        pcd2.colors = o3d.utility.Vector3dVector(target_color)

        o3d.visualization.draw_geometries([pcd1, pcd2])
        # stophere

    if display:
        colormap_and_plot(source_pts.T, source_class_pts)
        colormap_and_plot(target_pts.T, target_class_pts)

    return source_pts, target_pts, source_class_pts, target_class_pts
