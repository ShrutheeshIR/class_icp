import numpy as np
import open3d as o3d
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.transform import Rotation as R
# from plot_utils import colormap_and_plot, plot_point_clouds
from utils_3d import toHomog, toInHomog

np.set_printoptions(suppress=True, precision=5)


def fps_downsample(points, number_of_points_to_sample):

    points = points.T
    selected_points = np.zeros((number_of_points_to_sample, 3))

    idx_selected = np.zeros((number_of_points_to_sample), dtype=np.int32)

    dist = np.ones(points.shape[0]) * np.inf # distance to the selected set
    for i in range(number_of_points_to_sample):
        # pick the point with max dist
        idx = np.argmax(dist)
        selected_points[i] = points[idx]
        dist_ = ((points - selected_points[i]) ** 2).sum(-1)
        dist = np.minimum(dist, dist_)
        idx_selected[i] = idx

    return selected_points.T, idx_selected


def compute_nearest_neighbour(source_pts, target_pts, source_pts_classes, target_pts_classes, num_max_points_per_class):
    """
    source_pts : Nx3
    target_pts : Nx3

    Returns : R, t
    """
    classes = np.unique(source_pts_classes)
    all_target_indices = []
    all_source_indices = []
    all_distances = []

    for src_class in classes:
        # if src_class != 7 and src_class != 1 and src_class != 8:
        #     continue
        
        source_indices = np.nonzero(source_pts_classes == src_class)[0]
        # if len(source_indices) > num_max_points_per_class:
        #     _, selected_indices = fps_downsample(source_pts[source_indices].T, num_max_points_per_class)
        #     source_indices = source_indices[selected_indices]
        
        source_class_pts = source_pts[source_indices]

        target_indices = np.nonzero(target_pts_classes == src_class)[0]

        # print(src_class, len(target_indices), len(source_indices))

        target_class_pts = target_pts[target_indices] # what if target has no points belonging to this class?

        if len(target_class_pts) < 1:
            continue
        
        all_source_indices.extend(source_indices)

        neigh = NearestNeighbors(n_neighbors = 1)
        neigh.fit(target_class_pts)
        distance, indices = neigh.kneighbors(source_class_pts, return_distance = True)
        # print(indices.ravel().shape)
        
        all_distances.extend(distance.reshape(-1).tolist())

        target_class_indices = indices.ravel().reshape(-1)

        target_global_indices = target_indices[target_class_indices]
        all_target_indices.extend(target_global_indices.tolist())

        # plot_point_clouds(source_pts[source_indices], target_pts[target_global_indices.tolist()])

    

    return np.array(all_distances), np.array(all_target_indices), np.array(all_source_indices)


def solve_Rt(source_pts, target_pts):
    """
    source_pts : 3xN
    target_pts : 3xN

    Returns : R, t
    """
    source_pts = source_pts.T
    target_pts = target_pts.T

    source_mean = np.mean(source_pts, axis = 0)
    target_mean = np.mean(target_pts, axis = 0)

    P_bar = source_pts - source_mean
    Q_bar = target_pts - target_mean

    H = np.dot(P_bar.T, Q_bar)
    U, _, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = np.dot(Vt.T, U.T)

    t = target_mean.T - np.dot(R, source_mean.T)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t.squeeze()
    return T




def icp_iteration(source_pts, target_pts, source_pt_classes, target_pt_classes, num_max_points_per_class):
    """
    source_pts : 3xN
    target_pts : 3xN

    T0 : initial_guess, 4x4 matrix
    """
    # distances, target_indices = compute_nearest_neighbour(source_pts.T, target_pts.T)
    distances, target_indices, source_indices = compute_nearest_neighbour(source_pts.T, target_pts.T, source_pt_classes, target_pt_classes, num_max_points_per_class)

    # print("D : ", np.mean(distances))
    T = solve_Rt(source_pts[:, source_indices], target_pts[:, target_indices])
    return T, np.mean(distances)


def icp(source_pts_og, target_pts_og, source_pt_classes, target_pt_classes, T0 = np.eye(4), num_iters = 5000, capped_points_no = 500):
    T = T0.copy()

    source_pts = source_pts_og.copy()
    target_pts = target_pts_og.copy()

    source_mean = np.mean(source_pts, axis = 1)
    target_mean = np.mean(target_pts, axis = 1)

    source_pts -= source_mean.reshape((3,1))
    target_pts -= target_mean.reshape((3,1))

    src = source_pts.copy()
    dst = target_pts.copy()

    source_pts = toInHomog(T @ toHomog(source_pts))
    prev_err = np.inf


    # classes = np.unique(source_pts_classes)
    # for src_class in classes:
    #     source_indices = np.nonzero(source_pts_classes == src_class)[0]
    #     if len(source_indices) > capped_points_no:
    #         _, selected_indices = fps_downsample(src[source_indices].T, capped_points_no)
    #         source_indices = source_indices[selected_indices]




    for _ in range(100):
        T, d = icp_iteration(src, dst, source_pt_classes, target_pt_classes, capped_points_no)
        src = toInHomog(T @ toHomog(src))
        # if d > prev_err:
        #     break
        prev_err = d
    
    T = solve_Rt(source_pts_og, src)
    T[:3, 3] += target_mean
    return T


def initialize_ICP(source_pts, target_pts):

    T_init = np.eye(4)
    T_init[:3, :3] = R.random().as_matrix()

    return T_init



if __name__ == '__main__':
    do_semantic = False

    source_pts_og, target_pts_og, source_pts_classes, target_pts_classes = get_point_clouds('l2.npy', 'l1.npy', do_semantic = do_semantic, display = True)

    source_pts = source_pts_og.copy()
    target_pts = target_pts_og.copy()

    # source_pts = fps_downsample(source_pts_og, 20000)
    # target_pts = fps_downsample(target_pts_og, 20000)

    T_gt = np.eye(4)

    # source_pts_classes = np.zeros((len(source_pts.T)), dtype=np.int32)
    # target_pts_classes = np.zeros((len(target_pts.T)), dtype=np.int32)
    T0 = initialize_ICP(source_pts, target_pts)

    T = icp(source_pts, target_pts, source_pts_classes, target_pts_classes, T0 = T0)
    print(T)
    rre = np.rad2deg(compute_rre(T[:3, :3], T_gt[:3, :3]))
    rte = compute_rte(T[:3, 3], T_gt[:3, 3])
    print(f"rre={rre}, rte={rte}")

    display_registered_point_clouds(source_pts_og.T, target_pts_og.T, T)
    # display_registered_point_clouds(source_pts_og.T, target_pts_og.T, np.eye(4))    