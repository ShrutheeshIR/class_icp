from scipy.spatial.transform import Rotation as R
import numpy as np
from transforms3d.euler import euler2mat




T = np.eye(4)
T[:3, 3] = np.array([0.25, -1.0, 0.04])
euler = np.array([5, -10, 30])
print(euler2mat(5,-10, 30).shape)
T[:3, :3] = euler2mat(5,-10, 30)

# T[:3, :3] = R.from_euler('xyz', euler, degrees=True).as_matrix()

# T = np.array([[0.8528684973716736, -0.48499056696891785, -0.19338934123516083, 0.25], [0.49240386486053467, 0.8702971339225769, -0.01101461797952652, 1.0], [0.1736481785774231, -0.0858316421508789, 0.981060266494751, 0.03999999910593033], [0.0, 0.0, 0.0, 1.0]])
# np.savetxt('l1l2_transform.txt', T)
T_new = T.copy()
# new_R = np.linalg.inv(T[:3, :3])
# T_new[:3, :3] = new_R
# T_new[:3, 3] *= -1
np.savetxt('l1l2_transform.txt', T_new)
# np.savetxt('l1l2_transform.txt', np.linalg.inv(T_new))
