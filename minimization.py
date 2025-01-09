import numpy as np
from scipy.optimize import minimize
from glob import glob
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2

from pointcloud_utils import PointCloud
from image_utils import Image
from config import params


class Features:
    """Class to store 3D and 2D features"""
    def __init__(self, corners3d, corners2d):
        assert len(corners3d) == len(corners2d), "Number of 3D and 2D points must be the same"
        assert np.shape(corners3d)[1] == 3, "3D points must be in the form (n, 3)"
        assert np.shape(corners2d)[1] == 2, "2D points must be in the form (n, 2)"

        self.corners3d = corners3d
        self.corners2d = corners2d


def error_function_with_std(unknowns, *data):
    alpha, beta, gamma, x, y, z = unknowns
    corners3d = data[0].corners3d
    corners2d = data[0].corners2d

    pointcloud = PointCloud(corners3d)
    pointcloud.define_transform_matrix([alpha, beta, gamma], [x, y, z])
    pointcloud.get_spherical_coord(lidar2camera=True)
    # Rx = np.array([[1, 0, 0], [0, np.cos(alpha), -np.sin(alpha)], [0, np.sin(alpha), np.cos(alpha)]])
    # Ry = np.array([[np.cos(beta), 0, np.sin(beta)], [0, 1, 0], [-np.sin(beta), 0, np.cos(beta)]])
    # Rz = np.array([[np.cos(gamma), -np.sin(gamma), 0], [np.sin(gamma), np.cos(gamma), 0], [0, 0, 1]])
    # R = np.dot(Rz, np.dot(Ry, Rx))
    # translated_corners3d = np.dot(R, corners3d.T) + np.array([x, y, z]).reshape(3, 1)
    # pointcloud = PointCloud(translated_corners3d.T)
    # pointcloud.get_spherical_coord(lidar2camera=False)
    
    images_path = params.images_path
    imgs = sorted(glob(os.path.join(images_path, "*")), key=os.path.getmtime)
    # check if imgs is a directory
    if os.path.isdir(imgs[0]):
        imgs = sorted(glob(os.path.join(imgs[0], "*")), key=os.path.getmtime)
    
    # pointclouds_path = params.pointclouds_path
    # pcls = sorted(glob(os.path.join(pointclouds_path, '*')), key=os.path.getmtime)
    n = 2 * len(params.planes_sizes)
    image = Image(imgs[0], fov=185, spherical_image=params.spherical)
    image.eqr_coord = corners2d.T[:, :n]
    
    img = Image(imgs[0], fov=185, spherical_image=params.spherical)
    if len(data) > 1:
        im = data[1]
        scat = data[2]
        fig = data[3]
        im.set_data(img.image)
        pc = PointCloud(corners3d)
        pc.define_transform_matrix([alpha, beta, gamma], [x, y, z])
        pc.get_spherical_coord(lidar2camera=True)
        img.sphere_coord = pc.spherical_coord
        img.lidar_projection()
        x = np.concatenate((img.eqr_coord[0, :n], image.eqr_coord[0, :]))
        y = np.concatenate((img.eqr_coord[1, :n], image.eqr_coord[1, :])) 

        scat.set_offsets(np.array([x, y]).T)
        fig.canvas.draw()
        plt.pause(0.00001)
    
    image2 = Image(imgs[0], fov=185, spherical_image=params.spherical)
    image2.sphere_coord = pointcloud.spherical_coord
    image2.sphere2equirect()
    image2.norm2image(equirect=True)

    error = np.mean(np.linalg.norm(image2.eqr_coord - corners2d.T, axis=0))
    std = np.std(np.linalg.norm(image2.eqr_coord - corners2d.T, axis=0))
    # error = np.mean(np.linalg.norm(image.norm_coord.T - corners2d, axis=1))
    # error = np.sum(np.linalg.norm(image.norm_coord.T - corners2d, axis=1))
    # print(image2.eqr_coord.T)
    # print(image2.eqr_coord.T[:8])
    # print(corners2d[:8])
    print(error)
    # print(unknowns)
    return error, std


def error_function(unknowns, *data):
    alpha, beta, gamma, x, y, z = unknowns
    corners3d = data[0].corners3d
    corners2d = data[0].corners2d

    pointcloud = PointCloud(corners3d)
    pointcloud.define_transform_matrix([alpha, beta, gamma], [x, y, z])
    pointcloud.get_spherical_coord(lidar2camera=True)
    # Rx = np.array([[1, 0, 0], [0, np.cos(alpha), -np.sin(alpha)], [0, np.sin(alpha), np.cos(alpha)]])
    # Ry = np.array([[np.cos(beta), 0, np.sin(beta)], [0, 1, 0], [-np.sin(beta), 0, np.cos(beta)]])
    # Rz = np.array([[np.cos(gamma), -np.sin(gamma), 0], [np.sin(gamma), np.cos(gamma), 0], [0, 0, 1]])
    # R = np.dot(Rz, np.dot(Ry, Rx))
    # translated_corners3d = np.dot(R, corners3d.T) + np.array([x, y, z]).reshape(3, 1)
    # pointcloud = PointCloud(translated_corners3d.T)
    # pointcloud.get_spherical_coord(lidar2camera=False)
    
    images_path = params.images_path
    imgs = sorted(glob(os.path.join(images_path, "*")), key=os.path.getmtime)
    # check if imgs is a directory
    if os.path.isdir(imgs[0]):
        imgs = sorted(glob(os.path.join(imgs[0], "*")), key=os.path.getmtime)
    
    # pointclouds_path = params.pointclouds_path
    # pcls = sorted(glob(os.path.join(pointclouds_path, '*')), key=os.path.getmtime)
    n = 2 * len(params.planes_sizes)
    image = Image(imgs[0], fov=185, spherical_image=params.spherical)
    image.eqr_coord = corners2d.T[:, :n]
    
    img = Image(imgs[0], fov=185, spherical_image=params.spherical)
    if len(data) > 1:
        im = data[1]
        scat = data[2]
        fig = data[3]
        im.set_data(img.image)
        pc = PointCloud(corners3d)
        pc.define_transform_matrix([alpha, beta, gamma], [x, y, z])
        pc.get_spherical_coord(lidar2camera=True)
        img.sphere_coord = pc.spherical_coord
        img.lidar_projection()
        x = np.concatenate((img.eqr_coord[0, :n], image.eqr_coord[0, :]))
        y = np.concatenate((img.eqr_coord[1, :n], image.eqr_coord[1, :])) 

        scat.set_offsets(np.array([x, y]).T)
        fig.canvas.draw()
        plt.pause(0.00001)
    
    image2 = Image(imgs[0], fov=185, spherical_image=params.spherical)
    image2.sphere_coord = pointcloud.spherical_coord
    image2.sphere2equirect()
    image2.norm2image(equirect=True)

    error = np.mean(np.linalg.norm(image2.eqr_coord - corners2d.T, axis=0))
    # error = np.mean(np.linalg.norm(image.norm_coord.T - corners2d, axis=1))
    # error = np.sum(np.linalg.norm(image.norm_coord.T - corners2d, axis=1))
    # print(image2.eqr_coord.T)
    # print(image2.eqr_coord.T[:8])
    # print(corners2d[:8])
    print(error)
    # print(unknowns)
    return error


def ineq1(unknowns):
    alpha, beta, gamma, x, y, z = unknowns
    return -alpha + 180

def ineq2(unknowns):
    alpha, beta, gamma, x, y, z = unknowns
    return -beta + 180

def ineq3(unknowns):
    alpha, beta, gamma, x, y, z = unknowns
    return -gamma + 180

def ineq4(unknowns):
    alpha, beta, gamma, x, y, z = unknowns
    return alpha + 180

def ineq5(unknowns):
    alpha, beta, gamma, x, y, z = unknowns
    return beta + 180

def ineq6(unknowns):
    alpha, beta, gamma, x, y, z = unknowns
    return gamma + 180


def get_transformation_parameters(corners3d, corners2d, method, plot=False):
    constraints = [{'type': 'ineq', 'fun': ineq1},
                   {'type': 'ineq', 'fun': ineq2},
                   {'type': 'ineq', 'fun': ineq3},
                   {'type': 'ineq', 'fun': ineq4},
                   {'type': 'ineq', 'fun': ineq5},
                   {'type': 'ineq', 'fun': ineq6}]
    
    if plot:
        plt.ion()
        fig, ax = plt.subplots()
        images_path = params.images_path
        pointclouds_path = params.pointclouds_path
        imgs = sorted(glob(os.path.join(images_path, "*")))
        pcls = sorted(glob(os.path.join(pointclouds_path, '*')))
        img = Image(imgs[0], fov=185, spherical_image=True)
        im = ax.imshow(img.image)
        pc = PointCloud(pcls[0])
        pc.get_spherical_coord(lidar2camera=False)
        img.sphere_coord = pc.spherical_coord
        img.lidar_projection()
        scat = ax.scatter(img.eqr_coord[0, :], img.eqr_coord[1, :], c='r', s=10)  #, c=pc.depth, s=1, cmap='jet')

        solution = minimize(error_function, [0, 0, 0, 0, 0, 0], args=(Features(corners3d, corners2d), im, scat, fig), constraints=constraints, method=method, tol=1e-6)
        plt.ioff()
        plt.show()
        projection_error, projection_std = error_function_with_std(solution.x, Features(corners3d, corners2d))
    
    else:
        solution = minimize(error_function, [0, 0, 0, 0, 0, 0], args=Features(corners3d, corners2d), constraints=constraints, method=method)
        projection_error, projection_std = error_function_with_std(solution.x, Features(corners3d, corners2d))

        # pointcloud = PointCloud(corners3d)
        # alpha, beta, gamma, x, y, z = solution.x
        # pointcloud.define_transform_matrix([alpha, beta, gamma], [x, y, z])
        # pointcloud.get_spherical_coord(lidar2camera=True)
        # images_path = params.images_path
        # imgs = sorted(glob(os.path.join(images_path, "*")))
        # image = Image(imgs[0], fov=185, spherical_image=True)
        # image.sphere_coord = pointcloud.spherical_coord
        # image.sphere2equirect()
        # image.norm2image(equirect=True)
        # print(image.eqr_coord.T)
        # print(corners2d)
        # print('Mean pixel error:', np.mean(np.linalg.norm(image.eqr_coord - corners2d.T, axis=0)))

    return solution.x, projection_error, projection_std
