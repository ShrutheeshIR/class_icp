# class_icp

This work implements class-based Iterative Closest Point Matching (ICP) to estimate the relative transform between two sets of points, X and Y. This is a problem that often shows up in robotics, to estimate the pose of objects or self. 

Often the hardest step in ICP is performing correspondence matching. To improve this step, we leverage semantic data. After performing semantic segmentation, we can obtain the semantic class of each point in a point cloud. Using this class information, we perform matching by enforcing that the semantic classes must be equal for two points to match.

This improves the overall performance.

NOTE THAT THIS WORK DOES NOT CONTAIN THE LOGIC TO PERFORM SEMANTIC SEGMENTATION

### Brief description of code base


#### Library
1. The main lib resides in `class_icp.py`.
2. `file_utils.py` reads the points and their classes from given file names.
3. `eval_utils.py` for metrics
4. `utils_3d.py` to perform homogeneous transforms.
5. `plot_utils.py` contains utils to plot the point clouds in 3D


#### Running 

On custom data, run `runner.py`.
To integrate with ros and run on a current topic, run `ros_node.py`. Similarly to run on CARLA Self-driving data, run `runner_carla_data.py`


#### Docker
Setup the image using `docker_setup_pytorch.sh`