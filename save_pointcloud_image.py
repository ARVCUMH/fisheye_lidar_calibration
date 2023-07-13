import os
from glob import glob
from tkinter import N
import numpy as np
from plyfile import PlyElement, PlyData
import matplotlib.image as mpimg
import csv

from image_utils import Image, CamModel
from pointcloud_utils import Visualizer, load_pc
from config import params


def save_pointcloud_with_image_data(pointcloud, cam_model, index, num_zeros=0):
    """ Save the pointcloud with the image data in a ply file

        :param pointcloud: pointcloud object
        :param cam_model:  camera model object
        :param index:      index of the image
        :param num_zeros:  number of zeros to add to the index
    """
    pointcloud.unfilter_lidar()
    pointcloud.get_spherical_coord()
    pointcloud.encode_values()
    lidar_fish = Image(image=pointcloud.image, cam_model=cam_model, points_values=pointcloud.pixel_values)
    lidar_fish.sphere_coord = pointcloud.spherical_coord
    lidar_fish.change2camera_ref_system()
    lidar_fish.sphere2fisheye()
    lidar_fish.check_image_limits()
    u, v =  lidar_fish.spherical_proj[1], lidar_fish.spherical_proj[0]
    if pointcloud.reflectivity is None:
        points = np.hstack((pointcloud.lidar3d[lidar_fish.filtered_index, :],
                            pointcloud.reflectivity[lidar_fish.filtered_index].reshape(-1, 1),
                            u.reshape(-1, 1),
                            v.reshape(-1, 1)))
    else:
        points = np.hstack((pointcloud.lidar3d[lidar_fish.filtered_index, :],
                            u.reshape(-1, 1),
                            v.reshape(-1, 1)))
    
    # Create the plydata object
    np.round(num_zeros)
    if num_zeros <= 0:
        num_zeros = 6
        
    filename = params.save_path + '/pointclouds/pointcloud{{:0{}d}}.ply'.format(num_zeros)
    names = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('r', 'f4'), ('u', 'f4'), ('v', 'f4')]
    if pointcloud.reflectivity is None:
        names = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('u', 'f4'), ('v', 'f4')]
    list_points = list(map(tuple, points))
    points_ply = np.array(list_points, dtype=names)
    points_ply = PlyElement.describe(points_ply, 'vertex')
    PlyData([points_ply], byte_order='>').write(filename.format(index))
    
    # PlyData.write(filename, points)

if __name__ == "__main__":
    
    # Load images and pointclouds
    images_path = params.load_images_path
    pointclouds_path = params.load_pointclouds_path
    imgs = sorted(glob(os.path.join(images_path, "*")), key=os.path.getmtime)
    pcls = sorted(glob(os.path.join(pointclouds_path, '*')), key=os.path.getmtime)
    
    # Define camera model from calibration file
    cam_model = CamModel(params.calibration_file)
        
    # Read CSV file with rotation and translation data
    with open(params.save_path + '/' + params.results_file + '.csv', newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        reader = list(reader)
        rotation = np.array(reader[0]).astype(np.float32)
        translation = np.array(reader[1]).astype(np.float32)
        
    if not os.path.exists(params.save_path + '/pointclouds'):
        os.makedirs(params.save_path + '/pointclouds')
    
    index = 0
    num_zeros = len(str(len(imgs)))
    for i, p in zip(imgs, pcls):
        # Read image and pointcloud
        image = mpimg.imread(i)
        # Convert image to uint8 format
        image = (image * 255).astype(np.uint8)
        points = load_pc(p)
        # Save pointcloud with image data
        points = Visualizer(points, image)
        points.define_transform_matrix(rotation, translation)
        points.transform_lidar_to_camera()
        save_pointcloud_with_image_data(points, cam_model, index, num_zeros)
        index += 1
        
    
    