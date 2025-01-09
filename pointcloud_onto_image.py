import os.path
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import csv
import matplotlib.cm as cm

from config import params
from pointcloud_utils import Visualizer
from image_utils import Image, CamModel


def plot_pc_image_simulated(image, pointcloud, i, rotation, translation):
    plt.imshow(image)
    pointcloud.define_transform_matrix(rotation, translation)
    pointcloud.get_spherical_coord(True)
    xs, ys, zs = pointcloud.spherical_coord[0], pointcloud.spherical_coord[1], pointcloud.spherical_coord[2]
    long = np.arctan2(ys, xs)
    lat = np.arctan2(zs, np.linalg.norm([xs, ys], axis=0))
    x = (- long) * image.shape[1] / (2*np.pi) + image.shape[1] / 2
    y = (lat) * image.shape[0] / np.pi + image.shape[0] / 2
    # flip the image in the y-axis
    y = image.shape[0] - y
    plt.title('Image ' + str(i))
    plt.scatter(x-0.5, y-0.5, s=0.5, c='green')
    # print(np.amin(x-0.5), np.amax(x-0.5), np.amin(y-0.5), np.amax(y-0.5))
    # mini_x = np.array([x[np.argmax(pointcloud.lidar3d[:, 1])], x[np.argmin(pointcloud.lidar3d[:, 1])], x[np.argmax(pointcloud.lidar3d[:, 2])], x[np.argmin(pointcloud.lidar3d[:, 2])]])
    # mini_y = np.array([y[np.argmax(pointcloud.lidar3d[:, 1])], y[np.argmin(pointcloud.lidar3d[:, 1])], y[np.argmax(pointcloud.lidar3d[:, 2])], y[np.argmin(pointcloud.lidar3d[:, 2])]])
    # plt.scatter(mini_x, mini_y, s=5, c='r')
    # make the plot window bigger
    backend = plt.get_backend()
    if backend in ['TkAgg', 'Qt4Agg', 'Qt5Agg']:
        mng = plt.get_current_fig_manager()
        mng.window.showMaximized()
    elif backend == 'WXAgg':
        mng = plt.get_current_fig_manager()
        mng.frame.Maximize(True)
    elif backend == 'GTK3Agg':
        mng = plt.get_current_fig_manager()
        mng.window.fullscreen()

    plt.show()


def plot_pc_image(image, pointcloud, i, rotation, translation):
    plt.imshow(image.image)
    pointcloud.define_transform_matrix(rotation, translation)
    pointcloud.get_spherical_coord(True)
    xs, ys, zs = pointcloud.spherical_coord[0], pointcloud.spherical_coord[1], pointcloud.spherical_coord[2]
    image.sphere_coord = np.array([xs, ys, zs])
    image.change2camera_ref_system()
    image.sphere2fisheye()
    x, y = image.fisheye_coord[0], image.fisheye_coord[1]

    # delete points with negative coordinates
    mask = (x >= 0) & (y >= 0) & (x < image.image.shape[1]) & (y < image.image.shape[0])
    colours = cm.jet(image.points_values / np.amax(image.points_values))
    x, y, colours = x[mask], y[mask], colours[mask]
    cmap = 'jet'
    
    plt.title('Image ' + str(i))
    plt.scatter(x-0.5, y-0.5, s=0.03, c=colours, cmap=cmap)
    # print(np.amin(x-0.5), np.amax(x-0.5), np.amin(y-0.5), np.amax(y-0.5))
    # mini_x = np.array([x[np.argmax(pointcloud.lidar3d[:, 1])], x[np.argmin(pointcloud.lidar3d[:, 1])], x[np.argmax(pointcloud.lidar3d[:, 2])], x[np.argmin(pointcloud.lidar3d[:, 2])]])
    # mini_y = np.array([y[np.argmax(pointcloud.lidar3d[:, 1])], y[np.argmin(pointcloud.lidar3d[:, 1])], y[np.argmax(pointcloud.lidar3d[:, 2])], y[np.argmin(pointcloud.lidar3d[:, 2])]])
    # plt.scatter(mini_x, mini_y, s=5, c='r')
    # make the plot window bigger
    backend = plt.get_backend()
    if backend in ['TkAgg', 'Qt4Agg', 'Qt5Agg']:
        mng = plt.get_current_fig_manager()
        mng.window.showMaximized()
    elif backend == 'WXAgg':
        mng = plt.get_current_fig_manager()
        mng.frame.Maximize(True)
    elif backend == 'GTK3Agg':
        mng = plt.get_current_fig_manager()
        mng.window.fullscreen()

    plt.show()    


if __name__ == "__main__":
        
        # Get images and pointclouds paths
        images_path = params.images_path
        pointclouds_path = params.pointclouds_path
        imgs = sorted(glob(os.path.join(images_path, "*")))
        pcls = sorted(glob(os.path.join(pointclouds_path, '*')))

        # Define camera model from calibration file
        cam_model = CamModel(params.calibration_file)
        
        flag = False
        # if imgs[0] does not end in .png, then it is a directory
        if imgs[0].endswith('.png') is False:
            flag = True
    
        # indexes = list(range(1))
        indexes = list(range(len(imgs)))

        # if params.simulated:
        #     with open(params.ground_truth_file, newline='') as f:
        #         reader = csv.reader(f)
        #         # row one is rotation and row two is translation
        #         rotation = np.array([float(i) for i in next(reader)])
        #         translation = np.array([float(i) for i in next(reader)])
        
        # rotation = [0, 0, 0]
        # translation = [0, 0, 0]
        # rotation = [0.01188155220343893,-0.6374983787823745,-0.13089835596348143]
        # translation = [-0.059703185856596175,0.009314036627910305,0.1440442088452811]
        rotation = [0.09311604214286784,-0.7345086051159931,-0.2792859024494012]
        translation = [-0.07317915884214989,0.013365249479382401,0.13526401281259137]
        # rotation = [-0.4906162580864273,0.20528436907044503,0.2024056956163491]
        # translation = [-0.10098596812539153,0.02596603970813326,0.12371491268962882]
        # rotation = [-0.7663888665236177,0.4381860250904807,0.2000818393113821]
        # translation = [-0.09695261593447091,0.019264447355351044,0.1370679988071759]
        
        # Loop for selecting planes from the pointclouds and the images
        for i in indexes:
            if flag:
                images = sorted(glob(os.path.join(imgs[i], "*")))
                pointclouds = sorted(glob(os.path.join(pcls[i], '*')))
                for img, pc in zip(images, pointclouds):
                    # Read image and pointcloud
                    image = mpimg.imread(img)
                    pointcloud = Visualizer(pc, img)
                    pointcloud.reflectivity_filter(0.1)
                    if params.simulated:
                        plot_pc_image_simulated(image, pointcloud, i, rotation, translation)
                    else:
                        image_obj = Image(image, cam_model)
                        plot_pc_image(image_obj, pointcloud, i, rotation, translation)
            else:            
                # Read image and pointcloud
                image = mpimg.imread(imgs[i])
                pointcloud = Visualizer(pcls[i], imgs[i])
                pointcloud.reflectivity_filter(0.1)
                if params.simulated:
                    plot_pc_image_simulated(image, pointcloud, i, rotation, translation)
                else:
                    image_obj = Image(image, cam_model, points_values=pointcloud.depth)
                    plot_pc_image(image_obj, pointcloud, i, rotation, translation)