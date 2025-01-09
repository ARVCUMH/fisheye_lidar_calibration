import os.path
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import csv

from segment_anything import sam_model_registry, SamPredictor

from calibration_utils import *  # Plane, get_corners, get_rotation_and_translation, kabsch_plot, results_plot, select_image_plane, select_lidar_plane, get_lidar_corners, get_camera_corners
from pointcloud_utils import load_pc, Visualizer
from image_utils import Image
from image_utils import CamModel
from config import params


def corner_finder(images_path=None, pointclouds_path=None, save_data_path=None, save_data_file=None):
    
    # Get images and pointclouds paths
    if images_path is None:
        images_path = params.images_path
    if pointclouds_path is None:
        pointclouds_path = params.pointclouds_path
    imgs = sorted(glob(os.path.join(images_path, "*")))
    pcls = sorted(glob(os.path.join(pointclouds_path, '*')))

    # Define camera model from calibration file
    cam_model = CamModel(params.calibration_file)
    
    # Get plane parameters
    planes_sizes = params.planes_sizes
    
    # assert that images and pointclouds have the same length
    assert len(imgs) == len(pcls), "Images and pointclouds have different length"
    
    if params.simulated:
        mask = None
    # Load SAM model
    else:
        sam_checkpoint = params.model_path
        # sam_checkpoint = None
        sam = sam_model_registry[params.model_type](checkpoint=sam_checkpoint)
        sam.to(device=params.device)
        mask = SamPredictor(sam)
    
    indexes = list(range(len(imgs)))
    # indexes = list(range(1))
    # if len(imgs) > 10:
    #     indexes = list(range(10))
    # else:
    #     indexes = list(range(len(imgs)))

    rotations = np.zeros((len(indexes), 3))
    translations = np.zeros((len(indexes), 3))
    kabsch_errors = np.zeros(len(indexes))
    kabsch_std = np.zeros(len(indexes))
    
    pointclouds_points = [[] for _ in range(len(indexes))]
    init_planes_points = [[[] for _ in range(len(planes_sizes))] for _ in range(len(indexes))]
    selections = [[[] for _ in range(len(planes_sizes))] for _ in range(len(indexes))]
    
    # while len(indexes) != 0:
        
    # Loop for selecting planes from the pointclouds and the images
    for i in indexes:
        
        # Read image and pointcloud
        image = mpimg.imread(imgs[i])
        # Convert image to uint8 format
        image = (image * 255).astype(np.uint8)
        points = load_pc(pcls[i])
            
        # Range in meters for the lidar points
        d_range = (0, 80)

        # Define Visualizer and Image objects
        vis = Visualizer(points, image)
        if vis.reflectivity is not None:
            vis.reflectivity_filter(params.reflectivity_threshold)
        vis.get_spherical_coord(lidar2camera=0)
        vis.encode_values(d_range=d_range)
        equirect_lidar = Image(image=image, cam_model=cam_model, spherical_image=params.spherical, points_values=vis.pixel_values)
        pointclouds_points[i] = vis.lidar3d
        
        # Select planes from the pointcloud and the image
        idplane = 1
        for plane_size in planes_sizes:
            plane = Plane(plane_size[0], plane_size[1], idplane)
                
            init_plane_points = select_lidar_plane(vis, equirect_lidar, plane)
            selection_data = select_image_plane(image, plane)
            # print(np.where(selection_data == 1)[0].shape)
            
            init_planes_points[i][idplane - 1] = init_plane_points
            selections[i][idplane - 1] = selection_data
            
            idplane += 1

    camera_corners2d = []
    camera_corners = []
    lidar_corners = []
    # Loop for getting corners coordinates from the pointclouds and the images
    for j in indexes:
        
        # Read image and pointcloud
        image = mpimg.imread(imgs[j])
        # Convert image to uint8 format
        image = (image[:, :, :3] * 255).astype(np.uint8)
        
        # Get corners coordinates
        idplane = 1
        for plane_size in planes_sizes:
            plane = Plane(plane_size[0], plane_size[1], idplane)
            init_plane_points = init_planes_points[j][idplane - 1]
            l_corners = get_lidar_corners(pointclouds_points[j], init_plane_points, plane)
            # print(np.amax(selections[j][idplane - 1]))
            c_corners, c_corners2d, _ = get_camera_corners(image, cam_model, plane, l_corners, selections[j][idplane - 1], mask)
            camera_corners.extend(c_corners)
            camera_corners2d.extend(c_corners2d)
            lidar_corners.extend(l_corners)
            idplane += 1

    camera_corners2d = np.array(camera_corners2d)  
    camera_corners = np.array(camera_corners)
    lidar_corners = np.array(lidar_corners)

    if params.save_data:
        if not os.path.exists(params.data_path):
            os.makedirs(params.data_path)
        # concatenate camera and lidar corners
        l_c_corners = np.concatenate((lidar_corners, camera_corners, camera_corners2d), axis=1)
        
        if save_data_path is None:
            save_data_path = params.data_path
        if save_data_file is None:
            save_data_file = params.data_file
        
        filename = save_data_path + '/' + save_data_file + '.csv'
        with open(filename, 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            for corner in l_c_corners:
                csvwriter.writerow(corner)
                
                
if __name__ == "__main__":
    corner_finder()