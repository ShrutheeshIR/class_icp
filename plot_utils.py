import numpy as np
import open3d as o3d

def colormap_and_plot(source_ply, source_pts_classes):
    src_color = np.ones(source_ply[:, :3].shape) * np.array([0.95, 0, 0]).reshape((1,3))

    num_classes = np.unique(source_pts_classes)
    for c in num_classes:
        if c == 7:
            src_color[source_pts_classes == c] = np.random.rand(3) * 0.0
        else:
            src_color[source_pts_classes == c] = np.random.rand(3)
    

    pcd1 = o3d.geometry.PointCloud()
    pcd1.points = o3d.utility.Vector3dVector(source_ply[:, :3])
    pcd1.colors = o3d.utility.Vector3dVector(src_color)
    o3d.visualization.draw_geometries([pcd1])


def plot_point_clouds(source_ply, target_ply):
    src_color = np.ones(source_ply[:, :3].shape) * np.array([0.95, 0, 0]).reshape((1,3))
    target_color = np.ones(target_ply[:, :3].shape) * np.array([0, 0.95, 0]).reshape((1,3))

    pcd1 = o3d.geometry.PointCloud()
    pcd1.points = o3d.utility.Vector3dVector(source_ply[:, :3])
    pcd1.colors = o3d.utility.Vector3dVector(src_color)

    pcd2 = o3d.geometry.PointCloud()
    pcd2.points = o3d.utility.Vector3dVector(target_ply[:, :3])
    pcd2.colors = o3d.utility.Vector3dVector(target_color)

    o3d.visualization.draw_geometries([pcd1, pcd2])    

def display_registered_point_clouds(src, dst, T):
    """
    src : Nx3
    dst: Nx3
    """
    new_src = (T[:3, :3] @ src.T ).T + T[:3, 3] * np.array([1, 1, 1]).reshape((3,))
    plot_point_clouds(new_src, dst)