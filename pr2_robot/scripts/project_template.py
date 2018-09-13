#!/usr/bin/env python

# Import modules
import numpy as np
import sklearn
from sklearn.preprocessing import LabelEncoder
import pickle
from sensor_stick.srv import GetNormals
from sensor_stick.features import compute_color_histograms
from sensor_stick.features import compute_normal_histograms
from visualization_msgs.msg import Marker
from sensor_stick.marker_tools import *
from sensor_stick.msg import DetectedObjectsArray
from sensor_stick.msg import DetectedObject
from sensor_stick.pcl_helper import *

import rospy
import tf
from geometry_msgs.msg import Pose
from geometry_msgs.msg import Point
from std_msgs.msg import Float64
from std_msgs.msg import Int32
from std_msgs.msg import String
from pr2_robot.srv import *
from rospy_message_converter import message_converter
import yaml

world_number = 3

# Helper function to get surface normals
def get_normals(cloud):
    get_normals_prox = rospy.ServiceProxy('/feature_extractor/get_normals', GetNormals)
    return get_normals_prox(cloud).cluster

# Helper function to create a yaml friendly dictionary from ROS messages
def make_yaml_dict(test_scene_num, arm_name, object_name, pick_pose, place_pose):
    yaml_dict = {}
    yaml_dict["test_scene_num"] = test_scene_num.data
    yaml_dict["arm_name"]  = arm_name.data
    yaml_dict["object_name"] = object_name.data
    yaml_dict["pick_pose"] = message_converter.convert_ros_message_to_dictionary(pick_pose)
    yaml_dict["place_pose"] = message_converter.convert_ros_message_to_dictionary(place_pose)
    return yaml_dict

# Helper function to output to yaml file
def send_to_yaml(yaml_filename, dict_list):
    data_dict = {"object_list": dict_list}
    with open(yaml_filename, 'w') as outfile:
        yaml.dump(data_dict, outfile, default_flow_style=False)

def voxel_downsample(cloud, leaf_size):
    vox = cloud.make_voxel_grid_filter()
    vox.set_leaf_size(leaf_size, leaf_size, leaf_size)
    return vox.filter()

def passthrough_filter(cloud, filter_axis, axis_min, axis_max):
    passthrough = cloud.make_passthrough_filter()
    passthrough.set_filter_field_name(filter_axis)
    passthrough.set_filter_limits(axis_min, axis_max)
    return passthrough.filter()

def ransac_filter(cloud, max_distance):
    seg = cloud.make_segmenter()
    seg.set_model_type(pcl.SACMODEL_PLANE)
    seg.set_method_type(pcl.SAC_RANSAC)
    seg.set_distance_threshold(max_distance)
    return seg.segment()

def outlier_filter(cloud, inliers, negative, mean_k=50, x=1.0):
    extracted_inliers = cloud.extract(inliers, negative=negative)
    outlier_filter = extracted_inliers.make_statistical_outlier_filter()
    outlier_filter.set_mean_k(mean_k)
    outlier_filter.set_std_dev_mul_thresh(x)
    return outlier_filter.filter()

def statistical_filter(cloud, mean_k=50, x=1.0):
    outlier_filter = cloud.make_statistical_outlier_filter()
    outlier_filter.set_mean_k(mean_k)
    outlier_filter.set_std_dev_mul_thresh(x)
    return outlier_filter.filter()

def euclidean_clustering(cloud, tolerance, min_size, max_size):
    white_cloud = XYZRGB_to_XYZ(cloud)
    tree = white_cloud.make_kdtree()
    ec = white_cloud.make_EuclideanClusterExtraction()
    ec.set_ClusterTolerance(tolerance)
    ec.set_MinClusterSize(min_size)
    ec.set_MaxClusterSize(max_size)
    ec.set_SearchMethod(tree)
    return ec.Extract()

def color_clusters(cloud, cluster_indices):
    cluster_color = get_color_list(len(cluster_indices))
    color_cluster_point_list = []
    white_cloud = XYZRGB_to_XYZ(cloud)
    for j, indices in enumerate(cluster_indices):
        for i, indice in enumerate(indices):
            color_cluster_point_list.append([white_cloud[indice][0],
                white_cloud[indice][1],
                white_cloud[indice][2],
                rgb_to_float(cluster_color[j])])

    cluster_cloud = pcl.PointCloud_PointXYZRGB()
    cluster_cloud.from_list(color_cluster_point_list)
    return cluster_cloud


# Callback function for your Point Cloud Subscriber
def pcl_callback(pcl_msg):

# Exercise-2 TODOs:

    # TODO: Convert ROS msg to PCL data
    pcl_data = ros_to_pcl(pcl_msg)
    cloud_filtered = pcl_data

    # TODO: Statistical Outlier Filtering
    cloud_filtered = statistical_filter(cloud_filtered, 10, 0.001)

    # TODO: Voxel Grid Downsampling
    cloud_filtered = voxel_downsample(cloud_filtered, 0.01)

    # TODO: PassThrough Filter
    cloud_filtered = passthrough_filter(cloud_filtered, 'z', axis_min=0.6, axis_max=1.1)

    # TODO: RANSAC Plane Segmentation
    inliers, coefficients = ransac_filter(cloud_filtered, max_distance=0.01)

    # TODO: Extract inliers and outliers
    cloud_objects = outlier_filter(cloud_filtered, inliers, True)
    cloud_table = outlier_filter(cloud_filtered, inliers, False)

    # TODO: Euclidean Clustering
    cluster_indices = euclidean_clustering(cloud_objects, tolerance=0.05, min_size=100, max_size=2000)

    # TODO: Create Cluster-Mask Point Cloud to visualize each cluster separately
    cluster_cloud = color_clusters(cloud_objects, cluster_indices)

    # TODO: Convert PCL data to ROS messages
    # inlined!

    # TODO: Publish ROS messages
    pcl_objects_pub.publish(pcl_to_ros(cloud_objects))
    pcl_table_pub.publish(pcl_to_ros(cloud_table))
    pcl_cluster_pub.publish(pcl_to_ros(cluster_cloud))

# Exercise-3 TODOs:
    white_cloud = XYZRGB_to_XYZ(cloud_objects)

    # Classify the clusters! (loop through each detected cluster one at a time)
    detected_objects_labels = []
    detected_objects = []
    for index, pts_list in enumerate(cluster_indices):

        # Grab the points for the cluster
        pcl_cluster = cloud_objects.extract(pts_list)
        ros_cluster = pcl_to_ros(pcl_cluster)

        # Compute the associated feature vector
        chists = compute_color_histograms(ros_cluster, using_hsv=True)
        normals = get_normals(ros_cluster)
        nhists = compute_normal_histograms(normals)
        feature = np.concatenate((chists, nhists))

        # Make the prediction
        prediction = clf.predict(scaler.transform(feature.reshape(1,-1)))
        label = encoder.inverse_transform(prediction)[0]
        detected_objects_labels.append(label)

        # Publish a label into RViz
        label_pos = list(white_cloud[pts_list[0]])
        label_pos[2] += .4
        object_markers_pub.publish(make_label(label,label_pos, index))

        # Add the detected object to the list of detected objects.
        do = DetectedObject()
        do.label = label
        do.cloud = ros_cluster
        detected_objects.append(do)

    rospy.loginfo('Detected {} objects: {}'.format(len(detected_objects_labels), detected_objects_labels))

    # Publish the list of detected objects
    detected_objects_pub.publish(detected_objects)

    # Suggested location for where to invoke your pr2_mover() function within pcl_callback()
    # Could add some logic to determine whether or not your object detections are robust
    # before calling pr2_mover()
    try:
        pr2_mover(detected_objects)
    except rospy.ROSInterruptException:
        pass

# function to load parameters and request PickPlace service
def pr2_mover(detected_objects):
    dropbox = rospy.get_param('/dropbox')
    left_box_position = Point()
    right_box_position = Point()
    for box in dropbox:
        position = Point()
        position.x = box['position'][0]
        position.y = box['position'][1]
        position.z = box['position'][2]
        if box['name'] == 'left':
            left_box_position = position
        else:
            right_box_position = position

    test_scene_num = Int32()
    test_scene_num.data = world_number

    big_dict = []
    pick_list = rospy.get_param('/object_list')

    for item in pick_list:
        print 'Searching for item {} in {}'.format(item['name'], item['group'])

        # finds first object with matching name.
        for detected_object in detected_objects:
            if detected_object.label != item['name']:
                continue
            print 'Found {}'.format(detected_object.label)
            points_arr = ros_to_pcl(detected_object.cloud).to_array()
            centroid = np.mean(points_arr, axis=0)[:3]

            pick_pose = Pose()
            pick_pose.position.x = np.asscalar(centroid[0])
            pick_pose.position.y = np.asscalar(centroid[1])
            pick_pose.position.z = np.asscalar(centroid[2])

            object_name = String()
            object_name.data = item['name']

            place_pose = Pose()
            arm_name = String()

            if item['group'] == 'red':
                arm_name.data = 'left'
                place_pose = left_box_position
            else:
                arm_name.data = 'right'
                place_pose = right_box_position

            dict = make_yaml_dict(test_scene_num, arm_name, object_name, pick_pose, place_pose)
            big_dict.append(dict)
        # # Wait for 'pick_place_routine' service to come up
        # rospy.wait_for_service('pick_place_routine')
        #
        # try:
        #     pick_place_routine = rospy.ServiceProxy('pick_place_routine', PickPlace)
        #
        #     # TODO: Insert your message variables to be sent as a service request
        #     resp = pick_place_routine(TEST_SCENE_NUM, OBJECT_NAME, WHICH_ARM, PICK_POSE, PLACE_POSE)
        #
        #     print ("Response: ",resp.success)
        #
        # except rospy.ServiceException, e:
        #     print "Service call failed: %s"%e

    # TODO: Output your request parameters into output yaml file
    print 'writing yaml with {} items'.format(len(big_dict))
    send_to_yaml('output_{}.yaml'.format(world_number), big_dict)


if __name__ == '__main__':

    # TODO: ROS node initialization
    rospy.init_node('clustering', anonymous=True)

    # TODO: Create Subscribers
    pcl_sub = rospy.Subscriber('/pr2/world/points', pc2.PointCloud2, pcl_callback, queue_size=1)

    # TODO: Create Publishers
    pcl_objects_pub = rospy.Publisher('/pcl_objects', PointCloud2, queue_size=1)
    pcl_table_pub = rospy.Publisher('/pcl_table', PointCloud2, queue_size=1)
    pcl_cluster_pub = rospy.Publisher('/pcl_cluster', PointCloud2, queue_size=1)
    object_markers_pub = rospy.Publisher('/object_markers', Marker, queue_size=1)
    detected_objects_pub = rospy.Publisher('/detected_objects', DetectedObjectsArray, queue_size=1)

    # TODO: Load Model From disk
    model = pickle.load(open('/home/jeremy/catkin_ws/model_{}.sav'.format(world_number), 'rb'))
    clf = model['classifier']
    encoder = LabelEncoder()
    encoder.classes_ = model['classes']
    scaler = model['scaler']

    # Initialize color_list
    get_color_list.color_list = []

    # TODO: Spin while node is not shutdown
    while not rospy.is_shutdown():
        rospy.spin()
