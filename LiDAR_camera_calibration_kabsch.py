import os.path
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import csv
import time

from calibration_utils import *  # Plane, get_corners, get_rotation_and_translation, kabsch_plot, results_plot, select_image_plane, select_lidar_plane, get_lidar_corners, get_camera_corners
from pointcloud_utils import load_pc, Visualizer
from image_utils import CamModel
from config import params


def show_and_save_results(camera_corners3d, lidar_corners, image_corners, imgs, pcls, real_rotation=None, real_translation=None):
    # Get rotation and translation between camera and lidar reference systems
    start = time.time()
    rotation, translation, mean_error, std_error, mean_pixel_error, std_pixel_error = get_rotation_and_translation(camera_corners3d, lidar_corners, image_corners, PointCloud(lidar_corners, image=mpimg.imread(imgs[0])))
    delta_time = time.time() - start
    print('Mean error: ', mean_error)
    print('Mean pixel error: ', mean_pixel_error)
    print('Standard deviation pixel error: ', std_pixel_error)
    print('Rotation: ', rotation)
    print('Translation: ', translation)
    
    # rotation = np.zeros(3)
    # # translation = np.array([0.1, -0.05, -0.16415])
    # translation = np.array([-0.08, 0, 0.16])
    
    # Range in meters for the lidar points
    d_range = (0, 80)
    
    # Plot rotation and translation errors bars
    # rotation, translation = results_plot(rotations, translations)

    if params.show_lidar_onto_image > 0:
        n = 4 * len(params.planes_sizes)
        if params.simulated:
            for i in range(1):  # range(len(pcls)):
                pointcloud = PointCloud(pcls[i])
                pointcloud.define_transform_matrix(real_rotation, real_translation)
                
                # Comment and uncomment for adding artificial transformation to the pointclouds
                inverse_transform_matrix = np.linalg.inv(pointcloud.transform_matrix)
                points = np.vstack([pointcloud.lidar3d.T, np.ones(pointcloud.lidar3d.shape[0])])
                pointcloud.lidar3d = np.vstack(np.dot(inverse_transform_matrix, points)[0:3, :].T)

                image = mpimg.imread(imgs[i])
                plt.imshow(image)
                pointcloud.define_transform_matrix(rotation, translation)
                pointcloud.get_spherical_coord(lidar2camera=True)
                xs, ys, zs = pointcloud.spherical_coord[0], pointcloud.spherical_coord[1], pointcloud.spherical_coord[2]
                long = np.arctan2(ys, xs)
                lat = np.arctan2(zs, np.linalg.norm([xs, ys], axis=0))
                x = (- long) * image.shape[1] / (2*np.pi) + image.shape[1] / 2 - 0.5  # -0.5 to center the points
                y = (-lat) * image.shape[0] / np.pi + image.shape[0] / 2 - 0.5  # -0.5 to center the points
                plt.scatter(x, y, s=3, c=pointcloud.reflectivity, cmap='jet')

                # icorners = PointCloud(camera_corners3d[0 + n*i:n + n*i])
                # icorners.get_spherical_coord(lidar2camera=0)
                # xs, ys, zs = icorners.spherical_coord[0], icorners.spherical_coord[1], icorners.spherical_coord[2]
                # long = np.arctan2(ys, xs)
                # lat = np.arctan2(zs, np.linalg.norm([xs, ys], axis=0))
                # x = (- long) * image.shape[1] / (2*np.pi) + image.shape[1] / 2
                # y = (-lat) * image.shape[0] / np.pi + image.shape[0] / 2
                # plt.scatter(x, y, s=10, c='r')
                n = len(params.planes_sizes) * 4
                x = image_corners[i*n:i*n+n, 0] - 0.5
                y = image_corners[i*n:i*n+n, 1] - 0.5
                plt.scatter(x, y, s=10, c='r')
                
                pcorners = PointCloud(lidar_corners[n*i:n + n*i])
                pcorners.define_transform_matrix(rotation, translation)
                pcorners.get_spherical_coord(lidar2camera=True)
                xs, ys, zs = pcorners.spherical_coord[0], pcorners.spherical_coord[1], pcorners.spherical_coord[2]
                long = np.arctan2(ys, xs)
                lat = np.arctan2(zs, np.linalg.norm([xs, ys], axis=0))
                x = (- long) * image.shape[1] / (2*np.pi) + image.shape[1] / 2 - 0.5  # -0.5 to center the points
                y = (-lat) * image.shape[0] / np.pi + image.shape[0] / 2 - 0.5  # -0.5 to center the points
                # print(np.array([x, y]).T)
                # print(image_corners[0 + n*i:n + n*i] - 0.5)
                # print('Mean pixel error: ', np.mean(np.linalg.norm(np.array([x, y]).T + 0.5 - image_corners[0 + n*i:n + n*i], axis=1)))
                plt.scatter(x, y, s=10, c='g')
                plt.show()

        else:
            for points, image, i in zip(pcls, imgs, range(len(pcls))):
                points = load_pc(points)
                image = mpimg.imread(image)
                pointcloud = Visualizer(points, image)
                pointcloud.define_transform_matrix(rotation, translation)
                pointcloud.lidar_corners = lidar_corners[0 + n*i:n + n*i]
                pointcloud.camera_corners = camera_corners3d[0 + n*i:n + n*i]
                pointcloud.lidar_onto_image(cam_model=cam_model, fisheye=params.show_lidar_onto_image - 1, d_range=d_range)
                plt.show()
    
    if params.save_results:
        if not os.path.exists(params.save_path):
            os.makedirs(params.save_path)
        # Save rotation and translation results in a csv file
        filename = params.save_path + '/' + params.results_file + '.csv'
        with open(filename, 'a', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)            
            if params.simulated:
                rotationerrors = real_rotation - rotation
                translationerrors = real_translation - translation
                rotationerror = np.mean(np.abs(rotationerrors))
                translationerror = np.linalg.norm(translationerrors)
                if np.mean(np.abs(real_translation)) != 0:
                    relative_translationerror = translationerror/np.linalg.norm(real_translation)
                else:
                    relative_translationerror = float('nan')
                if np.mean(np.abs(real_rotation)) != 0:
                    relative_rotationerror = rotationerror/np.linalg.norm(real_rotation)
                else:
                    relative_rotationerror = float('nan')
                csvwriter.writerow([real_rotation[0], real_rotation[1], real_rotation[2],
                                    real_translation[0], real_translation[1], real_translation[2],
                                    rotation[0], rotation[1], rotation[2],
                                    translation[0], translation[1], translation[2],
                                    rotationerrors[0], rotationerrors[1], rotationerrors[2],
                                    translationerrors[0], translationerrors[1], translationerrors[2],
                                    rotationerror, relative_rotationerror,
                                    translationerror, relative_translationerror,
                                    mean_pixel_error, std_pixel_error, mean_error, delta_time])        
            else:
                csvwriter.writerow([rotation[0], rotation[1], rotation[2],
                                    translation[0], translation[1], translation[2],
                                    mean_pixel_error, std_pixel_error, mean_error, delta_time])
        
        
def get_groundtruth(lidar_corners, real_rotation, real_translation):

    # Comment and uncomment for adding artificial transformation to the pointclouds
    pcloud = PointCloud(lidar_corners)
    pcloud.define_transform_matrix(real_rotation, real_translation)
    points = np.vstack([pcloud.lidar3d.T, np.ones(pcloud.lidar3d.shape[0])])
    inverse_transform_matrix = np.linalg.inv(pcloud.transform_matrix)
    lidar_corners = np.vstack(np.dot(inverse_transform_matrix, points)[0:3, :].T)
    # lidar_corners = np.vstack(np.dot(pcloud.transform_matrix, points)[0:3, :].T)
    
    return lidar_corners


def read_data_file(filename):            
    with open(filename, newline='') as f:
        reader = csv.reader(f)
        corners = []
        for row in reader:
            corner = [float(row[0]), float(row[1]), float(row[2]), float(row[3]), float(row[4]), float(row[5]), float(row[6]), float(row[7])]
            if np.abs(np.linalg.norm(corner[0:3]) - np.linalg.norm(corner[3:6])) < 0.2:
                corners.append(corner)
    corners=np.asarray(corners)
    
    lidar_corners = corners[:, :3]
    camera_corners3d = corners[:, 3:6]
    image_corners = corners[:, 6:]
    
    return lidar_corners, camera_corners3d, image_corners


if __name__ == "__main__":
    
    # Get images and pointclouds paths
    images_path = params.images_path
    pointclouds_path = params.pointclouds_path
    imgs = sorted(glob(os.path.join(images_path, "*")), key=os.path.getmtime)
    pcls = sorted(glob(os.path.join(pointclouds_path, '*')), key=os.path.getmtime)

    # Define camera model from calibration file
    cam_model = CamModel(params.calibration_file)
    
    if params.data_file != '':
        data_file = params.data_path + '/' + params.data_file + '.csv'
        lidar_corners, camera_corners3d, image_corners = read_data_file(data_file)
        if params.simulated:
            data = data_file.split('_')
            real_translation = [float(data[1]), float(data[3]), float(data[5])]
            real_rotation = [float(data[7]), float(data[9]), float(data[11])]
            lidar_corners_transformed = get_groundtruth(lidar_corners, real_rotation, real_translation)
            show_and_save_results(camera_corners3d, lidar_corners_transformed, image_corners, imgs, pcls, real_rotation, real_translation)
        else:
            show_and_save_results(camera_corners3d, lidar_corners, image_corners, imgs, pcls)  
    
    else:
        data_files = sorted(glob(os.path.join(params.data_path, '*')), key=os.path.getmtime)
        for data_file, img_files, pc_files in zip(data_files, imgs, pcls):
            img_files = sorted(glob(os.path.join(img_files, "*")), key=os.path.getmtime)
            pc_files = sorted(glob(os.path.join(pc_files, "*")), key=os.path.getmtime)
            lidar_corners, camera_corners3d, image_corners = read_data_file(data_file)
            if params.simulated:
                data = data_file.split('_')
                real_translation = [float(data[1]), float(data[3]), float(data[5])]
                real_rotation = [float(data[7]), float(data[9]), float(data[11].split('.')[0])]
                lidar_corners_transformed = get_groundtruth(lidar_corners, real_rotation, real_translation)
                show_and_save_results(camera_corners3d, lidar_corners_transformed, image_corners, img_files, pc_files, real_rotation, real_translation)
            else:
                show_and_save_results(camera_corners3d, lidar_corners, image_corners, img_files, pc_files)               

    # # Get rotation and translation between camera and lidar reference systems
    # rotation, translation, mean_error, mean_pixel_error = get_rotation_and_translation(camera_corners3d, lidar_corners, image_corners, PointCloud(lidar_corners, image=mpimg.imread(imgs[0])))
    # kabsch_errors = mean_error
    # kabsch_mean_pixel = mean_pixel_error
    # print('Mean error: ', mean_error)
    # print('Mean pixel error: ', mean_pixel_error)
    # print('Rotation: ', rotation)
    # print('Translation: ', translation)
    
    # # rotation = np.zeros(3)
    # # # translation = np.array([0.1, -0.05, -0.16415])
    # # translation = np.array([-0.08, 0, 0.16])
            
    # #     indexes = kabsch_plot(kabsch_errors, kabsch_std, label='Kabsch error: click to repeat and press enter')
    
    # # # Delete the selected indexes from bad results from the Kabsch algorithm
    # # delete_idx = kabsch_plot(kabsch_errors, kabsch_std, label='Kabsch error: click to delete and press enter')
    # # rotations = np.delete(rotations, delete_idx, axis=0)
    # # translations = np.delete(translations, delete_idx, axis=0)
    
    # # Range in meters for the lidar points
    # d_range = (0, 80)
    
    # # Plot rotation and translation errors bars
    # # rotation, translation = results_plot(rotations, translations)

    # if params.show_lidar_onto_image > 0:
    #     n = 4 * len(params.planes_sizes)
    #     if params.simulated:
    #         for i in range(1):  # range(len(pcls)):
    #             pointcloud = PointCloud(pcls[i])
    #             pointcloud.define_transform_matrix(real_rotation, real_translation)
                
    #             # Comment and uncomment for adding artificial transformation to the pointclouds
    #             inverse_transform_matrix = np.linalg.inv(pointcloud.transform_matrix)
    #             points = np.vstack([pointcloud.lidar3d.T, np.ones(pointcloud.lidar3d.shape[0])])
    #             pointcloud.lidar3d = np.vstack(np.dot(inverse_transform_matrix, points)[0:3, :].T)

    #             image = mpimg.imread(imgs[i])
    #             plt.imshow(image)
    #             pointcloud.define_transform_matrix(rotation, translation)
    #             pointcloud.get_spherical_coord(lidar2camera=True)
    #             xs, ys, zs = pointcloud.spherical_coord[0], pointcloud.spherical_coord[1], pointcloud.spherical_coord[2]
    #             long = np.arctan2(ys, xs)
    #             lat = np.arctan2(zs, np.linalg.norm([xs, ys], axis=0))
    #             x = (- long) * image.shape[1] / (2*np.pi) + image.shape[1] / 2 - 0.5  # -0.5 to center the points
    #             y = (-lat) * image.shape[0] / np.pi + image.shape[0] / 2 - 0.5  # -0.5 to center the points
    #             plt.scatter(x, y, s=3, c=pointcloud.reflectivity, cmap='jet')

    #             # icorners = PointCloud(camera_corners3d[0 + n*i:n + n*i])
    #             # icorners.get_spherical_coord(lidar2camera=0)
    #             # xs, ys, zs = icorners.spherical_coord[0], icorners.spherical_coord[1], icorners.spherical_coord[2]
    #             # long = np.arctan2(ys, xs)
    #             # lat = np.arctan2(zs, np.linalg.norm([xs, ys], axis=0))
    #             # x = (- long) * image.shape[1] / (2*np.pi) + image.shape[1] / 2
    #             # y = (-lat) * image.shape[0] / np.pi + image.shape[0] / 2
    #             # plt.scatter(x, y, s=10, c='r')
    #             n = len(params.planes_sizes) * 4
    #             x = image_corners[i*n:i*n+n, 0] - 0.5
    #             y = image_corners[i*n:i*n+n, 1] - 0.5
    #             plt.scatter(x, y, s=10, c='r')
                
    #             pcorners = PointCloud(lidar_corners[n*i:n + n*i])
    #             pcorners.define_transform_matrix(rotation, translation)
    #             pcorners.get_spherical_coord(lidar2camera=True)
    #             xs, ys, zs = pcorners.spherical_coord[0], pcorners.spherical_coord[1], pcorners.spherical_coord[2]
    #             long = np.arctan2(ys, xs)
    #             lat = np.arctan2(zs, np.linalg.norm([xs, ys], axis=0))
    #             x = (- long) * image.shape[1] / (2*np.pi) + image.shape[1] / 2 - 0.5  # -0.5 to center the points
    #             y = (-lat) * image.shape[0] / np.pi + image.shape[0] / 2 - 0.5  # -0.5 to center the points
    #             # print(np.array([x, y]).T)
    #             # print(image_corners[0 + n*i:n + n*i] - 0.5)
    #             # print('Mean pixel error: ', np.mean(np.linalg.norm(np.array([x, y]).T + 0.5 - image_corners[0 + n*i:n + n*i], axis=1)))
    #             plt.scatter(x, y, s=10, c='g')
    #             plt.show()

    #     else:
    #         for points, image, i in zip(pcls, imgs, range(len(pcls))):
    #             points = load_pc(points)
    #             image = mpimg.imread(image)
    #             pointcloud = Visualizer(points, image)
    #             pointcloud.define_transform_matrix(rotation, translation)
    #             pointcloud.lidar_corners = lidar_corners[0 + n*i:n + n*i]
    #             pointcloud.camera_corners = camera_corners3d[0 + n*i:n + n*i]
    #             pointcloud.lidar_onto_image(cam_model=cam_model, fisheye=params.show_lidar_onto_image - 1, d_range=d_range)
    #             plt.show()

    # if not os.path.exists(params.save_path):
    #     os.makedirs(params.save_path)
    
    # if params.save_results:
    #     # Save rotation and translation results in a csv file
    #     filename = params.save_path + '/' + params.results_file + '.csv'
    #     with open(filename, 'w', newline='') as csvfile:
    #         csvwriter = csv.writer(csvfile)
    #         # csvwriter.writerow(['Transformation matrix', pointcloud.transform_matrix])
    #         csvwriter.writerow(['Rotation', rotation[0], rotation[1], rotation[2]])
    #         csvwriter.writerow(['Translation', translation[0], translation[1], translation[2]])
    #         csvwriter.writerow(['MeanError', mean_error])
    #         csvwriter.writerow(['MeanPixelError', mean_pixel_error])
    #         if params.simulated:
    #             rotationerrors = rotation - real_rotation
    #             csvwriter.writerow(['RotationErrors', rotationerrors[0], rotationerrors[1], rotationerrors[2]])
    #             rotationerror = np.mean(np.abs(rotationerrors))
    #             csvwriter.writerow(['RotationError', rotationerror])
    #             translationerrors = translation - real_translation
    #             csvwriter.writerow(['TranslationErrors', translationerrors[0], translationerrors[1], translationerrors[2]])
    #             translationerror = np.linalg.norm(translationerrors)
    #             csvwriter.writerow(['TranslationError', translationerror])
    #     # np.savetxt(params.save_path + '/' + params.results_file, results, delimiter=",")
    