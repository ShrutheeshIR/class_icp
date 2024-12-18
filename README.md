# class_icp

This work implements class-based Iterative Closest Point Matching (ICP) to estimate the relative transform between two sets of points, X and Y. This is a problem that often shows up in robotics, to estimate the pose of objects or self. 

Often the hardest step in ICP is performing correspondence matching. To improve this step, we leverage semantic data. After performing semantic segmentation, we can obtain the semantic class of each point in a point cloud. Using this class information, we perform matching by enforcing that the semantic classes must be equal for two points to match.

This improves the overall performance.

To run, look at `class_icp.py`