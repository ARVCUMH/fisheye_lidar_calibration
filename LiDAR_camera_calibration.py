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


if __name__ == "__main__":
    
    # Get images and pointclouds paths
    images_path = params.images_path
    pointclouds_path = params.pointclouds_path
    imgs = sorted(glob(os.path.join(images_path, "*")), key=os.path.getmtime)
    pcls = sorted(glob(os.path.join(pointclouds_path, '*')), key=os.path.getmtime)

    # Define camera model from calibration file
    cam_model = CamModel(params.calibration_file)
    
    # Get plane parameters
    planes_sizes = params.planes_sizes
    
    # assert that images and pointclouds have the same length
    assert len(imgs) == len(pcls), "Images and pointclouds have different length"
    
    # Load SAM model
    sam_checkpoint = "sam_vit_h_4b8939.pth"
    sam = sam_model_registry[params.model_type](checkpoint=sam_checkpoint)
    sam.to(device=params.device)
    mask = SamPredictor(sam)
    
    # indexes = list(range(2))
    indexes = list(range(len(imgs)))

    rotations = np.zeros((len(indexes), 3))
    translations = np.zeros((len(indexes), 3))
    kabsch_errors = np.zeros(len(indexes))
    kabsch_std = np.zeros(len(indexes))
    
    pointclouds_points = [[] for _ in range(len(indexes))]
    init_planes_points = [[[] for _ in range(len(planes_sizes))] for _ in range(len(indexes))]
    selections = [[[] for _ in range(len(planes_sizes))] for _ in range(len(indexes))]
    
    while len(indexes) != 0:
        
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
            vis.reflectivity_filter(params.reflectivity_threshold)
            vis.get_spherical_coord(lidar2camera=0)
            vis.encode_values(d_range=d_range)
            equirect_lidar = Image(image=image, cam_model=cam_model, points_values=vis.pixel_values)
            pointclouds_points[i] = vis.lidar3d
            
            # Select planes from the pointcloud and the image
            idplane = 1
            for plane_size in planes_sizes:
                plane = Plane(plane_size[0], plane_size[1], idplane)
                
                init_plane_points = select_lidar_plane(vis, equirect_lidar, plane)
                selection_data = select_image_plane(image, plane)
                
                init_planes_points[i][idplane - 1] = init_plane_points
                selections[i][idplane - 1] = selection_data
                idplane += 1
        
        # Loop for computing the rotation and translation between the camera and the lidar reference systems
        for j in indexes:
            
            # Read image and pointcloud
            image = mpimg.imread(imgs[j])
            # Convert image to uint8 format
            image = (image * 255).astype(np.uint8)
        
            camera_corners = []
            lidar_corners = []
            
            # Get corners coordinates
            idplane = 1
            for plane_size in planes_sizes:
                plane = Plane(plane_size[0], plane_size[1], idplane)
                pc = pointclouds_points[j]
                init_plane_points = init_planes_points[j][idplane - 1]
                l_corners = get_lidar_corners(pc, init_plane_points, plane)
                c_corners = get_camera_corners(image, cam_model, plane, l_corners, selections[j][idplane - 1], mask)
                camera_corners.extend(c_corners)
                lidar_corners.extend(l_corners)
                idplane += 1
                
            camera_corners = np.array(camera_corners)
            lidar_corners = np.array(lidar_corners)

            # Get rotation and translation between camera and lidar reference systems
            rotation, translation, mean_error, std_error = get_rotation_and_translation(camera_corners, 
                                                                                        lidar_corners, pc,
                                                                                        camera_model=cam_model)
            kabsch_errors[j] = mean_error
            kabsch_std[j] = std_error

            rotations[j] = rotation
            translations[j] = translation
            
        indexes = kabsch_plot(kabsch_errors, kabsch_std, label='Kabsch error: click to repeat and press enter')
    
    # Delete the selected indexes from bad results from the Kabsch algorithm
    delete_idx = kabsch_plot(kabsch_errors, kabsch_std, label='Kabsch error: click to delete and press enter')
    rotations = np.delete(rotations, delete_idx, axis=0)
    translations = np.delete(translations, delete_idx, axis=0)
    
    # Range in meters for the lidar points
    d_range = (0, 80)
    
    # Plot rotation and translation errors bars
    rotation, translation = results_plot(rotations, translations)
    for points, image in zip(pcls, imgs):
        points = load_pc(points)
        image = mpimg.imread(image)
        pointcloud = Visualizer(points, image)
        pointcloud.define_transform_matrix(rotation, translation)
        pointcloud.lidar_onto_image(cam_model=cam_model, fisheye=1, d_range=d_range)
        plt.show()

    if not os.path.exists(params.save_path):
        os.makedirs(params.save_path)
    
    if params.save_results:
        # Save rotation and translation results in a csv file
        results = np.array([rotation, translation])
        filename = params.save_path + '/' + params.results_file + '.csv'
        with open(filename, 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            for row in results:
                csvwriter.writerow(row)
        # np.savetxt(params.save_path + '/' + params.results_file, results, delimiter=",")
    