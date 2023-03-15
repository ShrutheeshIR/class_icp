
import sensor_msgs.point_cloud2 as pc2
import numpy as np
import rospy

class Lidar():
    def __init__(self, scan_topic="/robot_0/base_scan"):
        self.scan_sub = rospy.Subscriber(scan_topic, LaserScan, self.on_scan)
        self.laser_projector = LaserProjection()

    def on_scan(self, scan):
        rospy.loginfo("Got scan, projecting")
        cloud = self.laser_projector.projectLaser(scan)
        gen = pc2.read_points(cloud, skip_nans=True, field_names=("x", "y", "z", "o_id"))
        self.xyz_generator = gen