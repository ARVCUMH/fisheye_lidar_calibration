import cv2
import numpy as np
import pyransac3d as pyrsc
import mplcursors
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector
from scipy.optimize import fsolve, least_squares
from scipy import spatial
from scipy.spatial.transform import Rotation as R
from sklearn.cluster import KMeans
import time

from pointcloud_utils import PointCloud
from image_utils import Image
from config import params


class Plane:
    """ Class for defining reference plane dimensions. """
    def __init__(self, width, height, i=1, diagonal=None):
        self.width = width
        self.height = height
        self.diagonal = diagonal
        self.i = i
        if self.diagonal is None:
            self.diagonal = np.sqrt(self.width**2 +self.height**2)

def get_plane_points(points, i, plane_id):
    """ Find plane points from an initial seed selected on the screen with LassoSelector.

        :param points: array of lidar points
        :param i:      indexes of the plane points selected on the screen
        :plane_id:     Plane id

        :return: list of final point cloud plane points
    """
    # Find plane points from an initial seed
    points_kdtree = spatial.KDTree(points)
    indexes = []
    while True:
        idxs = points_kdtree.query_ball_point(points[i], params.radius_kdtree[plane_id - 1])
        if len(idxs) == 0:
            break
        new_indexes = []
        if idxs[0].__class__ == int:
            idxs = [idxs]
        for idx in idxs:            
            new_indexes.extend(idx)
        new_indexes = list(dict.fromkeys(new_indexes))  # filter repeated points
        i = [e for e in new_indexes if e not in indexes]
        indexes.extend(i)

    return points[indexes]


def equations(xyz, *data):
    """ Define 6 distance equations between checkerboard corners, 4 line equations where the corners are found and
        plane equations for getting corners 3D coordinates.

        :param xyz:  array with 12 coordinates from the same dimension from each corner
        :param data: data with the following attributes:
                   p0: spherical unit coordinates from top left corner
                   p1: spherical unit coordinates from top right corner
                   p2: spherical unit coordinates from bottom left corner
                   p3: spherical unit coordinates from bottom right corner
                   plane: plane object with dimension data

        :return: array of functions
    """

    # Get point values and plane dimension parameters
    c, plane = data
    p0, p1, p2, p3 = c[:, 0], c[:, 1], c[:, 2], c[:, 3]

    # CHECKERBOARD DIMENSIONS
    w, h, d = plane.width, plane.height, plane.diagonal

    x, y, z = xyz[:4], xyz[4:8], xyz[8:]

    # Line equations
    f1 = x[0]/p0[0] - y[0]/p0[1]
    f2 = x[0]/p0[0] - z[0]/p0[2]
    f3 = x[1]/p1[0] - y[1]/p1[1]
    f4 = x[1]/p1[0] - z[1]/p1[2]
    f5 = x[2]/p2[0] - y[2]/p2[1]
    f6 = x[2]/p2[0] - z[2]/p2[2]
    f7 = x[3]/p3[0] - y[3]/p3[1]
    f8 = x[3]/p3[0] - z[3]/p3[2]

    # Distance equations
    # f0 = (x[1] - x[0])**2 + (y[1] - y[0])**2 + (z[1] - z[0])**2 - w**2
    # f1 = (x[3] - x[2])**2 + (y[3] - y[2])**2 + (z[3] - z[2])**2 - w**2
    # f2 = (x[2] - x[0])**2 + (y[2] - y[0])**2 + (z[2] - z[0])**2 - h**2
    # f3 = (x[3] - x[1])**2 + (y[3] - y[1])**2 + (z[3] - z[1])**2 - h**2
    # f4 = (x[3] - x[0])**2 + (y[3] - y[0])**2 + (z[3] - z[0])**2 - d**2
    # f5 = (x[2] - x[1])**2 + (y[2] - y[1])**2 + (z[2] - z[1])**2 - d**2
    f9 = (x[1] - x[0])**2 + (y[1] - y[0])**2 + (z[1] - z[0])**2 - w**2
    f10 = (x[3] - x[2])**2 + (y[3] - y[2])**2 + (z[3] - z[2])**2 - w**2
    f11 = (x[2] - x[0])**2 + (y[2] - y[0])**2 + (z[2] - z[0])**2 - h**2
    f12 = (x[3] - x[1])**2 + (y[3] - y[1])**2 + (z[3] - z[1])**2 - h**2
    f13 = (x[3] - x[0])**2 + (y[3] - y[0])**2 + (z[3] - z[0])**2 - ((x[2] - x[1])**2 + (y[2] - y[1])**2 + (z[2] - z[1])**2)

    # Plane restriction equation
    f14 = np.linalg.det(np.matrix([[x[1]-x[0], y[1]-y[0], z[1]-z[0]], [x[2]-x[0], y[2]-y[0], z[2]-z[0]], [x[3]-x[0], y[3]-y[0], z[3]-z[0]]]))

    return np.array([f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13, f14])


def bounding_edge_corners(points, plane):
    """ Get the bounding edge corners with minimum error.
        The rectangle with minimum mse that bounds the 2D perimeter points.
        :param: points:    array of projected plane points
        :param: plane:     Plane object with the plane dimensions data

        :return: Base object with the [length, width], yaw, centroid [x,y] and fixed corner [x,y] from the base
    """

    # Get convex hull points only for x and y coordinates to get the perimeter
    hull_indexes = spatial.ConvexHull(points[:, :2]).vertices
    hull_points = points[hull_indexes, :2]
    perimeter = car2pol(hull_points)  # Perimeter in polar coordinates

    # Get the area of the rectangle for each orientation
    areas = []
    rotations = params.rotations
    for i in range(0, rotations):
        car_perimeter = pol2car(perimeter)  # Cartesian perimeter
        # Calculate area of the rectangle
        area = (np.amax(car_perimeter[:, 0]) - np.amin(car_perimeter[:, 0])) * (np.amax(car_perimeter[:, 1]) - np.amin(car_perimeter[:,  1]))
        areas.append(area)
        perimeter[:, 1] = perimeter[:, 1] + (np.pi / 2) / rotations  # New yaw

    min_area_index = np.argmin(areas)
    yaw = min_area_index * (np.pi / 2) / rotations  # Yaw of the rectangle with minimum area
    perimeter[:, 1] = perimeter[:, 1] + yaw - np.pi / 2  # Rotate the perimeter to get size and position
    car_p = pol2car(perimeter)  # Cartesian perimeter

    # Get four corners from rectangle orientation
    car_corners = np.array([[np.amin(car_p[:, 0]), np.amax(car_p[:, 1])], [np.amax(car_p[:, 0]), np.amax(car_p[:, 1])],
                            [np.amin(car_p[:, 0]), np.amin(car_p[:, 1])], [np.amax(car_p[:, 0]), np.amin(car_p[:, 1])]])

    # Augment the rectangle to get the real plane size
    x_dist, y_dist = np.amax(car_p[:, 0]) - np.amin(car_p[:, 0]), np.amax(car_p[:, 1]) - np.amin(car_p[:, 1])
    width, height = plane.width, plane.height
    h, w = height, width
    if x_dist < y_dist and width > height:
        h = width
        w = height
    elif width < height:
        h = width
        w = height

    car_corners[0] = [car_corners[0, 0] - (w - x_dist)/2, car_corners[0, 1] + (h - y_dist)/2]
    car_corners[1] = [car_corners[1, 0] + (w - x_dist)/2, car_corners[1, 1] + (h - y_dist)/2]
    car_corners[2] = [car_corners[2, 0] - (w - x_dist)/2, car_corners[2, 1] - (h - y_dist)/2]
    car_corners[3] = [car_corners[3, 0] + (w - x_dist)/2, car_corners[3, 1] - (h- y_dist)/2]
    pol_corners = car2pol(car_corners)
    pol_corners[:, 1] = pol_corners[:, 1] - yaw  # rotate back to the true position
    corners = pol2car(pol_corners)

    # Get centroid from the corners' mean
    centroid = [np.mean(corners[:, 0]), np.mean(corners[:, 1])]

    return centroid, corners


def pol2car(points):
    """ Converts 2D points in polar coordinates to cartesian coordinates.
        :param: points: Array of 2D points in polar coordinates

        :return: Array of 2D points in cartesian coordinates
    """
    rho = points.T[0]
    phi = points.T[1]
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    cartesians = np.vstack([x, y]).T
    return cartesians

def car2pol(points):
    """ Converts 2D points in cartesian coordinates to polar coordinates.
        :param: points: Array of 2D points in cartesian coordinates

        :return: Array of 2D points in polar coordinates
    """
    x = points.T[0]
    y = points.T[1]
    rho = np.sqrt(x ** 2 + y ** 2)
    phi = np.arctan2(y, x)
    polars = np.vstack([rho, phi]).T
    return polars


def corner_finder(points, plane):
    """ Steps to find 4 plane corners given the hand-selected plane points and the plane dimensions.
            1. Find a plane equation with ransac and project points to founded plane.
            2. Find the center of the plane.
            3. Knowing the distance center-to-corner, trace a line from center to the farthest plane point
               and get two opposite corners
            4. Knowing the width and height of the plane, get the other two corners from all possible solutions.

        :param points: array of hand-selected plane points.
        :param plane:  Plane object containing dimension information of the plane

        :return: array of 4 corners founded.
    """

    # Get plane equation from using Ransac
    rsc_plane = pyrsc.Plane()
    plane_eq, _ = rsc_plane.fit(points, thresh=0.01, maxIteration=20000)
    A, B, C, D = plane_eq
    if C > 0:
        A, B, C, D = -A, -B, -C, -D
    normal = np.array([A, B, C])

    # Project plane points onto the plane
    proj_points = []
    for p in points:
        k = - (A*p[0] + B*p[1] + C*p[2] + D) / (A**2 + B**2 + C**2)
        pp = [k*A + p[0], k*B + p[1], k*C + p[2]]
        proj_points.append(pp)
    proj_points = np.array(proj_points)

    # Get the center of mass (fix as a plane center)
    center = []
    for i in range(proj_points.shape[1]):  # for x, y, z
        center.append(np.sum(proj_points[:, i]) / proj_points.shape[0])

    # Define v1 as a vector normal to x-axis and plane normal
    v1 = np.cross(np.array([1, 0, 0]), normal)

    # Transform proj_points to a new coordinate system with center as origin and normal as z-axis
    M = np.eye(4)
    M[:3, 0] = v1 / np.linalg.norm(v1)
    M[:3, 2] = normal / np.linalg.norm(normal)
    M[:3, 1] = np.cross(M[:3, 2], M[:3, 0])
    M[:3, 3] = np.array(center)
    # Add a fourth dimension to the points with value 1
    proj_points_ = np.vstack([proj_points.T, np.ones(proj_points.shape[0])])
    proj_points_ = np.dot(np.linalg.inv(M), proj_points_)[0:3, :].T

    # Get center and corners in new coordinate system
    center, corners = bounding_edge_corners(proj_points_, plane)

    # Transform corners to original coordinate system adding a third dimension (z=0)
    corners = np.hstack([corners, np.zeros((corners.shape[0], 1))])
    # Add a fourth dimension to the points with value 1
    corners = np.vstack([corners.T, np.ones(corners.shape[0])])
    # Transform corners to original coordinate system
    corners = np.dot(M, corners)[0:3, :].T

    return corners


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.3])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.3])
    h, w = mask.shape[:2]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25) 

def on_release(pos1, pos2):
    global input_box
    input_box = np.array([pos1.xdata, pos1.ydata, pos2.xdata, pos2.ydata]).astype(int)
    plt.close()


def select_lidar_plane(visualizer, equirect_lidar, plane):
    """ Get the transformation matrix from the plane to the equirectangular image.
        :param visualizer:     Visualizer object containing the fisheye image and the lidar point cloud
        :parar equirect_lidar: Image object containing the fisheye image and the lidar point cloud
        :param plane:          Plane object containing dimension information of the plane
    
        :return: array of initial plane points.
    """

    # Plot equirectangular image and point cloud
    equirect_lidar.sphere_coord = visualizer.spherical_coord
    equirect_lidar.lidar_projection(pixel_points=0)
    
    if params.simulated:
        # cluster points in groups
        kmeans = KMeans(n_clusters=len(params.planes_sizes), random_state=0).fit(equirect_lidar.eqr_coord.T)
        
        # get the cluster with the most points if plane is 0, the second most if plane is 1, etc
        counts = np.bincount(kmeans.labels_)
        sorted_indices = np.argsort(counts)[::-1]

        # Create a mapping from old labels to new labels
        mapping = np.zeros_like(sorted_indices)
        mapping[sorted_indices] = np.arange(len(sorted_indices))

        # Apply the mapping to the labels
        new_labels = mapping[kmeans.labels_]
        plane_index_points = np.where(new_labels == plane.i - 1)[0]
    
    else:
        # Select plane points from plotted lidar
        fig, ax = plt.subplots(1)
        mng = plt.get_current_fig_manager()
        mng.resize(*mng.window.maxsize())
        ax.imshow(np.zeros((equirect_lidar.eqr_image.shape[0], equirect_lidar.eqr_image.shape[1])))
        ax.set_title('Select plane number ' + str(plane.i))
        pts = ax.scatter(x=equirect_lidar.eqr_coord[0], y=equirect_lidar.eqr_coord[1], c=equirect_lidar.points_values,
                        s=0.05, cmap=visualizer.cmap)
        plane_point = np.round(plt.ginput(timeout=0))[0]
        pixel_window = 6
        plane_index_points = np.where((equirect_lidar.eqr_coord[0] > plane_point[0] - pixel_window ) &
                                    (equirect_lidar.eqr_coord[0] < plane_point[0] + pixel_window ) &
                                    (equirect_lidar.eqr_coord[1] > plane_point[1] - pixel_window ) &
                                    (equirect_lidar.eqr_coord[1] < plane_point[1] + pixel_window ))[0]
        plt.close(fig)
    
    return plane_index_points


def select_image_plane(image, plane):
    """ Select image plane from fisheye image.
        :param image: fisheye image
        :param plane: Plane object containing dimension information of the plane
        
        :return: if detection mode is manual, array of 2D points of the plane 
        :return: if detection mode is automatic, input data for mask detection 
    """
    # Plot fisheye image and get 2D calibration pattern corners manually
    if params.corner_detection_mode == "manual":
        mng = plt.get_current_fig_manager()
        mng.resize(*mng.window.maxsize())
        plt.imshow(image)
        print("Zoom in and select 4 checkerboard corners: top left, top right, bottom left, bottom right from plane " + str(plane.i) + " of the image")
        plt.title("Zoom in and select 4 checkerboard corners: top left, top right, bottom left, bottom right from plane " + str(plane.i) + " of the image")
        image2d_points = plt.ginput(5, timeout=0)
        image2d_points = np.round(image2d_points[1:]).astype(int)
        plt.close()
        
        return image2d_points
    
    # Extract corners coordinates automatically from fisheye image
    elif params.corner_detection_mode == "automatic":
        # Segment Anything parameters
        input_point = None
        global input_box
        input_box = None
        
        if params.simulated:
            mask = np.zeros((image.shape[0], image.shape[1]))
            colour = np.array(params.planes_colours[plane.i - 1])
            colour = colour / np.amax(colour)
            # threshold the image coordinates in 1
            binarized_image = np.zeros((image.shape[0], image.shape[1], 4))
            binarized_image[np.where(image > 1)] = 1
            # get the coordinates of the points in the images which are equal to colour
            mask[np.where((binarized_image[:, :, 0] == colour[0]) & (binarized_image[:, :, 1] == colour[1]) & (binarized_image[:, :, 2] == colour[2]))] = 1
            return mask.astype(np.uint8)
            
        else:
            # Plot fisheye image and click on the plane to get corners
            fig, ax = plt.subplots()
            mng = plt.get_current_fig_manager()
            mng.resize(*mng.window.maxsize())
            ax.imshow(image)
            
            # Define Segment Anything parameters
            if params.selection_mode == "points":
                plt.title("Select points from the plane " + str(plane.i) + " to segment it. Press a key to continue")
                point = []
                while True:
                    p = plt.ginput(1, timeout=0)
                    if not p:
                        plt.close()
                        break  # Press a key to continue
                    point.append(p[0])
                point = np.round(point).astype(int)
                input_point = np.array(point)
                
                return input_point
                
            elif params.selection_mode == "box":
                plt.title("Draw a box around the plane " + str(plane.i) + " to segment it")
                rs = RectangleSelector(ax, onselect=on_release, useblit=True, button=[1], minspanx=5, minspany=5)
                plt.show()
            
                return input_box
            

def reorder_corners(corners):
    """ Reorder corners to match the order of the corners in the lidar point cloud.
    
        :param corners: array of 4 corners founded.
        
        :return: array of 4 corners reordered.
    """
    # Lidar format: [x, y, z]
    if corners.shape[1] == 3:
        m = corners[:, 1] + corners[:, 2]
        reordered_corners = np.zeros((4, 3))
        reordered_corners[0, :] = corners[np.argmax(m)]
        reordered_corners[3, :] = corners[np.argmin(m)]
        reordered_corners_del = np.delete(corners, [np.argmin(m), np.argmax(m)], axis=0)
        reordered_corners[1, :] = reordered_corners_del[np.argmin(reordered_corners_del[:, 1])]
        reordered_corners[2, :] = reordered_corners_del[np.argmax(reordered_corners_del[:, 1])]
    # Image format: [x, y]
    else:
        m = corners[:, 0] + corners[:, 1]
        reordered_corners = np.zeros((4, 2))
        reordered_corners[3, :] = corners[np.argmax(m)][0] + 0.5, corners[np.argmax(m)][1] + 0.5
        reordered_corners[0, :] = corners[np.argmin(m)][0] - 0.5, corners[np.argmin(m)][1] - 0.5
        reordered_corners_del = np.delete(corners, [np.argmin(m), np.argmax(m)], axis=0)
        mi = np.argmin(reordered_corners_del[:, 0])
        ma = np.argmax(reordered_corners_del[:, 0])
        reordered_corners[2, :] = reordered_corners_del[mi][0] - 0.5, reordered_corners_del[mi][1] + 0.5
        reordered_corners[1, :] = reordered_corners_del[ma][0] + 0.5, reordered_corners_del[ma][1] - 0.5
    
    return reordered_corners


def get_lidar_corners(points, plane_index_points, plane):
    """ Get the lidar 3d corners from the initial plane points.
        :param points:             array with 3d lidar filtered points
        :param plane_index_points: array of initial plane points.
        :param plane:              Plane object containing dimension information of the plane
        
        :return: array with 3d lidar corners
    """
    if params.simulated:
        plane_points = points[plane_index_points]
    else:
        # Find plane points from an initial seed
        plane_points = get_plane_points(points[:, :3], plane_index_points, plane.i)

    # Find corners from plane points founded with corner_finder
    lidar_corners3d = np.asarray(corner_finder(plane_points, plane))

    # Reorder corners
    lidar_corners3d = reorder_corners(lidar_corners3d)

    # plot 3D points. Plot plane_points and lidar_corners3d with different colors. Plot lines between corners
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(plane_points[:, 0], plane_points[:, 1], plane_points[:, 2], s=0.5, c="#2d40ff")
    ax.scatter(lidar_corners3d[:, 0], lidar_corners3d[:, 1], lidar_corners3d[:, 2], s=5, c="#fa2a00")
    ax.plot([lidar_corners3d[0, 0], lidar_corners3d[2, 0], lidar_corners3d[3, 0], lidar_corners3d[1, 0],
                lidar_corners3d[0, 0]],
            [lidar_corners3d[0, 1], lidar_corners3d[2, 1], lidar_corners3d[3, 1], lidar_corners3d[1, 1],
                lidar_corners3d[0, 1]],
            [lidar_corners3d[0, 2], lidar_corners3d[2, 2], lidar_corners3d[3, 2], lidar_corners3d[1, 2],
                lidar_corners3d[0, 2]], c="#fa2a00")
    ax.set_xlabel('X axis', fontsize=15), ax.set_ylabel('Y axis', fontsize=15)
    ax.set_zlabel('Z axis', fontsize=15), ax.set_aspect('equal')
    if params.save_plots:
        plt.savefig(params.plots_path + '/lidar_corners_plane_' + str(plane.i) + '_' + str(time.time()) + '.png', dpi=700)
    if params.show_lidar_plane:
        plt.show()
    plt.close()
    
    return lidar_corners3d


def get_lidar_contour(points, plane_index_points, plane):
    """ Get the plane contour from the initial plane points.
        :param points:              array with 3d lidar filtered points
        :param plane_index_poiints: array of initial plane points.
        :param plane:               Plane object containing dimension information of the plane
        
        :return: array with 3d lidar contour points
    """
    corners3d = get_lidar_corners(points, plane_index_points, plane)
    corners3d[2], corners3d[3] = corners3d[3].copy(), corners3d[2].copy()
    contour_points = []
    for i in range(4):
        p1 = corners3d[i]
        p2 = corners3d[(i + 1) % 4]

        # distance between p1 and p2
        num_points = 20
        t = np.linspace(0, 1, num_points)
        points = np.outer((1 - t), p1) + np.outer(t, p2)
        # add points arrays to the contour list
        for point in points:
            contour_points.append(point)
    contour_points = np.asarray(contour_points)
    # plt.plot(point[0], point[1], 'ro')

    return contour_points


def get_plane_mask(image, input_data, mask_predict=None):
    """ Get the plane mask from the image.
        :param image:           image in uint8 format
        :param input_data:      if detection mode is manual, array of 2D points of the plane 
                                if detection mode is automatic, input data for mask detection (points or box)
        :param mask_predict:    mask_predict object for predicting the mask of the plane
        
        :return: 2d array image sized with mask information, score of the mask and labels
    """
    if params.simulated:
        score = [1.0]
        if params.selection_mode == "points":
            # Find points in the neighborhood and add them to the plane if they are not black
            plane_points = np.zeros((image.shape[0], image.shape[1]))
            plane_points[input_data[:, 1], input_data[:, 0]] = 1
            directions = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (-1, -1), (-1, 1), (1, -1)]
            queue = [input_data[0]]
            while queue:
                point = queue.pop(0)
                x, y = point[1], point[0]
                for dx, dy in directions:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < image.shape[0] and 0 <= ny < image.shape[1]:
                        if np.any(image[nx, ny] != np.zeros(3)):
                            if plane_points[nx, ny] != 1:
                                plane_points[nx, ny] = 1
                                queue.append((ny, nx))
            mask = plane_points.astype(np.uint8) * 255
            input_point = input_data
            # print(input_point)
            input_label = np.ones(len(input_point))
            
        elif params.selection_mode == "box":
            plane_points = np.zeros((image.shape[0], image.shape[1]))
            # turn into one the points in the box that are not black in the image
            # # plot the box onto the image
            # plt.imshow(image)
            # plt.gca().add_patch(plt.Rectangle((input_data[0], input_data[1]), input_data[2] - input_data[0],
            #                                     input_data[3] - input_data[1], linewidth=1, edgecolor='r',
            #                                     facecolor='none'))
            # plt.show()
            window = image[input_data[1]:input_data[3], input_data[0]:input_data[2], :]
            # get the indexes from the points that are not black
            idx = np.where(window != [0, 0, 0])
            # convert to 1 the points from plane points that are not black
            plane_points[input_data[1] + idx[0], input_data[0] + idx[1]] = 1
            mask = plane_points.astype(np.uint8) * 255

    elif params.simulated != True:
        assert mask_predict is not None, "mask_predict must be provided in automatic and not simulated mode"
        if params.selection_mode == "points":
            input_point = input_data
            input_label = np.ones(len(input_point))  # Ones for foreground
            input_box = None
        elif params.selection_mode == "box":
            input_box = input_data
            input_point, input_label = None, None

        # Predict mask
        mask_predict.set_image(image)
        mask, score, logits = mask_predict.predict(point_coords=input_point,
                                                   point_labels=input_label,
                                                   box=input_box,
                                                   multimask_output=False)

        if params.dilation:
            # Convert the boolean values to uint8 values (0 for False, 255 for True)
            mask_uint8 = mask[0].astype(np.uint8) * 255
            ks = params.kernel_size
            kernel = np.ones((ks, ks), np.uint8)  # Define kernel for dilation
            # Apply dilation to the mask
            mask = cv2.dilate(mask_uint8, kernel, iterations=1)

    return mask, score, input_point, input_label


def corners2d_to_3d(image2d_points, lidar_corners3d, image, plane, camera_model=None):
    """ Get the camera 3d corners from the 2d corners in the image.
        :param image2d_points:  array of 2D points of the plane
        :param lidar_corners3d: array with 3d lidar corners
        :param image:           image array
        :param plane:           Plane object containing dimension information of the plane
        
        :return: array with 3d camera corners
        :return: array with 2d normalised camera corners
    """
    # Extract unit sphere corners coordinates from spherical projection or fisheye image
    if params.spherical:
        image_points = np.zeros(image2d_points.T.shape)
        image_points[0] = - (image2d_points.T[0] - image.shape[1] / 2)
        image_points[1] = - (image2d_points.T[1] - image.shape[0] / 2)
        image_points[0] = 2 * image_points[0] / image.shape[1]
        image_points[1] = image_points[1] / image.shape[0]
        long = np.pi * image_points[0]
        lat = np.pi * image_points[1]
        xs = np.cos(lat) * np.cos(long)
        ys = np.cos(lat) * np.sin(long)
        zs = np.sin(lat)        
        image3d_sphere = np.vstack([xs, ys, zs])
        image_points = image2d_points

        # long = np.arctan2(ys, xs)
        # lat = np.arctan2(zs, np.linalg.norm([xs, ys], axis=0))
        # xi = (- long) * image.shape[1] / (2*np.pi) + image.shape[1] / 2
        # yi = (-lat) * image.shape[0] / np.pi + image.shape[0] / 2
        # plt.imshow(image)
        # plt.scatter(xi, yi, s=10, c='r')
        # plt.show()

    else:
        assert camera_model is not None, "camera_model must be provided in fisheye mode"
        fish_image = Image(image, cam_model=camera_model)
        fish_image.fisheye_coord = np.flip(image2d_points.T, axis=0)
        fish_image.fisheye2sphere()
        image3d_sphere = fish_image.sphere_coord
        fish_image.sphere2equirect()
        fish_image.norm2image(equirect=True)
        image_points = fish_image.eqr_coord.T
    
    # if there is a value in image3d_sphere that is 0, replace it with 1e-10 to avoid division by zero
    image3d_sphere[image3d_sphere == 0] = 1e-10

    # Get 3D corners coordinates from camera reference system
    # Change initial solutions if needed, remain x positive to get the plane in front of the camera
    xyz0 = np.array(np.reshape(lidar_corners3d.T, 12))
    # xyz = fsolve(equations, xyz0, args=(image3d_sphere, plane))
    xyz = least_squares(equations, xyz0, args=(image3d_sphere, plane))
    camera_corners3d = np.reshape(xyz.x, (3, 4)).T

    return camera_corners3d, image_points

        
def get_camera_corners(image, camera_model, plane, lidar_corners3d, input_data, mask_predict=None):
    """ Get the camera 3d corners from the planes in the images.
        :param image:           image in uint8 format
        :param camera_model:    CamModel object
        :param plane:           Plane object containing dimension information of the plane
        :param lidar_corners3d: array with 3d lidar corners
        :param input_data:      if detection mode is manual, array of 2D points of the plane 
                                if detection mode is automatic, input data for mask detection (points or box)
        :param mask_predict:    mask_predict object for predicting the mask of the plane
        
        :return: array with 3d camera corners, array with 2d camera corners and mask
    """
    
    if params.corner_detection_mode == "manual":
        assert len(input_data) == 4, "4 corners must be provided in manual mode"
        image2d_points = input_data
        
    elif params.corner_detection_mode == "automatic":
        if params.simulated:
            mask = input_data
            score = [1.0]
        else:
            mask, score, input_point, input_label = get_plane_mask(image, input_data, mask_predict)
        
        # Find the contours of the mask image
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        contour_points = []
        for contour in contours:
            for point in contour:
                x, y = point[0]
                contour_points.append([x, y])
        contour_points = np.array(contour_points)
        
        # Get centroid of the mask
        centroid = np.mean(contour_points, axis=0)
        # Calculate the distance between the centroid and each contour point
        distances = np.sqrt(np.sum((contour_points - centroid)**2, axis=1))
        # Delete x% of the points that are closest to the centroid
        th = params.contour_distance_threshold
        contour_points = np.delete(contour_points, np.argsort(distances)[:int(len(contour_points) * th)], axis=0)
        # Create 4 groups of points that are close to each other (for four corners)
        kmeans = KMeans(n_clusters=4, random_state=0).fit(contour_points)
        cluster_labels = kmeans.labels_
        # Get point p1 from the cluster with label 0, p2 from the cluster with label 1, etc.
        g1 = contour_points[cluster_labels == 0]
        g2 = contour_points[cluster_labels == 1]
        g3 = contour_points[cluster_labels == 2]
        g4 = contour_points[cluster_labels == 3]
        
        # Find 4 mask points that maximize the distance between them as corners
        distance = 0

        # Run the RANSAC algorithm to find the four points with the maximum distance between them
        for i in range(params.ransac_iterations):
            # Randomly select four points from the four groups
            p1 = g1[np.random.randint(0, len(g1))]
            p2 = g2[np.random.randint(0, len(g2))]
            p3 = g3[np.random.randint(0, len(g3))]
            p4 = g4[np.random.randint(0, len(g4))]

            # Compute the distances between the four points
            distances = [np.sqrt(np.sum((p1 - p2)**2)),
                         np.sqrt(np.sum((p2 - p3)**2)),
                         np.sqrt(np.sum((p3 - p4)**2)),
                         np.sqrt(np.sum((p4 - p1)**2)),
                         np.sqrt(np.sum((p1 - p3)**2)),
                         np.sqrt(np.sum((p2 - p4)**2))]
            # distances = [np.sqrt(np.sum((p1 - p3)**2)),
            #              np.sqrt(np.sum((p2 - p4)**2))]
            sum_distances = np.sum(distances)

            # Check if the distance sumatory is greater than the previous one
            if sum_distances > distance:
                distance = sum_distances
                image2d_points = np.array([p1, p2, p3, p4]).squeeze()

        corners = []
        for image2d_point in image2d_points:
            # get the image window around the point
            window_size = 20
            window_mask = mask[int(image2d_point[1]) - window_size:int(image2d_point[1]) + window_size,
                               int(image2d_point[0]) - window_size:int(image2d_point[0]) + window_size]
            
            # window = image[int(image2d_point[1]) - window_size:int(image2d_point[1]) + window_size,
            #                int(image2d_point[0]) - window_size:int(image2d_point[0]) + window_size, :]
            # gray = cv2.cvtColor(window, cv2.COLOR_BGR2GRAY)
            # # find Harris corners
            # gray = np.float32(gray)
            # dst = cv2.cornerHarris(gray, 3, 3, 0.04)
            # corner = np.where(dst == np.amax(dst))[1][0], np.where(dst == np.amax(dst))[0][0]
            # corners.append([corner[0] + int(image2d_point[0]) - window_size,
            #                 corner[1] + int(image2d_point[1]) - window_size])

            # dst_idx = 0
            # while True:
            #     dst = cv2.cornerHarris(gray, 3, 3, 0.04)
            #     # get a list of dst values from the biggest to the smallest
            #     dst_list = np.sort(dst.flatten())[::-1]
            #     corner = np.where(dst == dst_list[dst_idx])[1][0], np.where(dst == dst_list[dst_idx])[0][0]
            #     # # plot the window and the corner
            #     # plt.imshow(window)
            #     # plt.scatter(corner[0], corner[1], s=10, c='r')
            #     # plt.show()
            #     if window_mask[corner] != 0:
            #         break
            #     else:
            #         dst_idx += 1
            
            # Find the contours of the mask image
            window_contours, hierarchy = cv2.findContours(window_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
            window_contour_points = []
            for window_contour in window_contours:
                for point in window_contour:
                    y, x = point[0]
                    window_contour_points.append([x, y])
            window_contour_points = np.array(window_contour_points)
            # create a mask with contour points
            window_contour_points_mask = np.zeros_like(window_mask)
            window_contour_points_mask[window_contour_points[:, 0], window_contour_points[:, 1]] = 1

            # from the window_mask, get the corner =! 0
            end_idx = window_size * 2 - 1
            window_mask_corners_idxs = [[0, 0], [0, end_idx], [end_idx, 0], [end_idx, end_idx]]
            window_mask_corners_values = [window_mask[0][0], window_mask[0][-1], window_mask[-1][0], window_mask[-1][-1]]
            for window_maks_corner_value, window_mask_corner_idxs in zip(window_mask_corners_values, window_mask_corners_idxs):
                if window_maks_corner_value != 0:
                    window_corner = window_mask_corner_idxs
            distance = 0
            # erase window contour points with 0 or window_size*2 - 1 in its coordinates
            window_contour_points = np.delete(window_contour_points, np.where(window_contour_points == 0)[0], axis=0)
            window_contour_points = np.delete(window_contour_points, np.where(window_contour_points == window_size * 2 - 1)[0], axis=0)
            for window_contour_point in window_contour_points:
                # calculate the distance between the window corner and the contour points
                d = np.linalg.norm(window_corner - window_contour_point)
                if d > distance:
                    distance = d
                    corner = [window_contour_point[1], window_contour_point[0]]
            # print(corner)
            mean_point = np.mean([corner, [window_size, window_size]], axis=0)
            d = 0
            corners.append([mean_point[0] + int(image2d_point[0]) - window_size,
                            mean_point[1] + int(image2d_point[1]) - window_size])
            # # show mask 
            # fig, ax = plt.subplots(1)
            # ax.imshow(window_contour_points_mask)
            # ax.scatter(mean_point[0], mean_point[1], s=10, c='r')
            # plt.show()
        
        # Reorder corners
        image2d_points = reorder_corners(np.asarray(corners)) 
            
        if True:    
            # Convert the uint8 values back to boolean values
            m = (mask == 255)
            plt.imshow(image)
            show_mask(m, plt.gca(), random_color=True)
            # Show clicked points if the mask is selected by points
            if params.selection_mode == "points" and params.simulated != True:
                show_points(input_point, input_label, plt.gca())
            # Plot corners 
            plt.scatter(image2d_points[:, 0], image2d_points[:, 1], s=0.05, c="#fa2a00")        
            plt.title(f"Mask 1, Score: {score[0]:.3f}", fontsize=18)
            plt.axis('off')
            if params.save_plots:
                plt.savefig(params.plots_path + '/segmented_plane_' + str(plane.i) + '_' + str(time.time()) + '.png', dpi=700)
            if params.show_segmented_plane:
                plt.show()
            plt.close()

    camera_corners3d, image2d_points = corners2d_to_3d(image2d_points + 0.5, lidar_corners3d, image, plane, camera_model)

    # # plot 3d camera_corners3d. Draw lines between corners
    # plt.figure()
    # ax = plt.axes(projection='3d')
    # # label each point as P0, P1, P2, P3
    # for i, txt in enumerate(["P0", "P1", "P2", "P3"]):
    #     ax.text(camera_corners3d[i, 0], camera_corners3d[i, 1], camera_corners3d[i, 2], txt)
    # ax.scatter(camera_corners3d[:, 0], camera_corners3d[:, 1], camera_corners3d[:, 2], s=5, c="#fa2a00")
    # ax.scatter(0, 0, 0, s=5, c="#2d40ff")
    # ax.plot([camera_corners3d[0, 0], camera_corners3d[2, 0], camera_corners3d[3, 0], camera_corners3d[1, 0],
    #             camera_corners3d[0, 0]],
    #         [camera_corners3d[0, 1], camera_corners3d[2, 1], camera_corners3d[3, 1], camera_corners3d[1, 1],
    #             camera_corners3d[0, 1]],
    #         [camera_corners3d[0, 2], camera_corners3d[2, 2], camera_corners3d[3, 2], camera_corners3d[1, 2],
    #             camera_corners3d[0, 2]], c="#fa2a00")
    # ax.set_xlabel('X axis', fontsize=15), ax.set_ylabel('Y axis', fontsize=15)
    # ax.set_zlabel('Z axis', fontsize=15), # ax.set_aspect('equal')
    # # plot infinite lines that pass through zero and lidar_corners3d
    # # colors red, green, blue, yellow
    # colors = ["#ff0000", "#00ff00", "#0000ff", "#ffff00"]
    # for i in range(4):
    #     ax.plot([0, image3d_sphere.T[i, 0]*7], [0, image3d_sphere.T[i, 1]*7], [0, image3d_sphere.T[i, 2]*7], c=colors[i])
    # plt.show()

    # corners = PointCloud(camera_corners3d)
    # corners.get_spherical_coord()
    # xs, ys, zs = corners.spherical_coord[0], corners.spherical_coord[1], corners.spherical_coord[2]
    # long = np.arctan2(ys, xs)
    # lat = np.arctan2(zs, np.linalg.norm([xs, ys], axis=0))
    # x = (- long) * image.shape[1] / (2*np.pi) + image.shape[1] / 2
    # y = (-lat) * image.shape[0] / np.pi + image.shape[0] / 2
    # plt.scatter(x, y, s=10, c='r')
    # plt.show()
    
    return camera_corners3d, image2d_points, mask


def get_rotation_and_translation(camera_corners3d, lidar_corners3d, image_corners, pointcloud):
    """ Estimate rotation and translation between camera and lidar reference systems
        :param camera_corners3d: array of 3D camera corners coordinates
        :param lidar_corners3d:  array of 3D lidar corners coordinates
        :param pointcloud:       PointCloud object
        :param camera_model:     CameraModel object containing camera parameters

        :return: array of euler angles of rotation
        :return: array of translation values
        :return: error between transformed lidar corners and camera corners
        :return: standard deviation of the error
        :return: mean error between transformed lidar corners and camera corners
    """

    # Estimate transform matrix between corners
    # print('\n3D camera corners coordinates: \n', camera_corners3d,
    #       '\n\n 3D lidar corners coordinates: \n', lidar_corners3d)

    # pointcloud.define_transform_matrix(np.array([0, 0, 0]), np.array([0.06, 0.11, -0.23]))
    # # frontal
    # r1 = [0.09311604214286784,-0.7345086051159931,-0.2792859024494012]
    # t1 = [-0.07317915884214989,0.013365249479382401,0.13526401281259137]
    # r2 = [0.01188155220343893,-0.6374983787823745,-0.13089835596348143]
    # t2 = [-0.059703185856596175,0.009314036627910305,0.1440442088452811]
    # # rear
    # r1 = [-0.7663888665236177,0.4381860250904807,0.2000818393113821]
    # t1 = [-0.09695261593447091,0.019264447355351044,0.1370679988071759]
    # r2 = [-0.4906162580864273,0.20528436907044503,0.2024056956163491]
    # t2 = [-0.10098596812539153,0.02596603970813326,0.12371491268962882]
    # r_mean = np.mean([r1, r2], axis=0)
    # t_mean = np.mean([t1, t2], axis=0)
    # pointcloud.define_transform_matrix(r_mean, t_mean)

    pointcloud.estimate_transform_matrix(camera_corners3d, lidar_corners3d)
    print('\nTransformation matrix: \n', pointcloud.transform_matrix)

    # print lidar_corners3d transformed to camera reference system
    lidar_corners3d_transformed = np.matmul(pointcloud.transform_matrix, np.vstack((lidar_corners3d.T, np.ones((1, lidar_corners3d.shape[0]))))).T[:, :3]
    # print('\nLidar corners coordinates transformed to camera reference system: \n', lidar_corners3d_transformed)

    # Get error between transformed lidar corners and camera corners
    err = np.linalg.norm(lidar_corners3d_transformed - camera_corners3d, axis=1)
    mean_error = np.mean(err)
    std_err = np.std(err)

    # Get reprojection error
    image = Image(pointcloud.image, fov=185, spherical_image=params.spherical)
    image.sphere_coord = lidar_corners3d_transformed.T
    image.sphere2equirect()
    image.norm2image(equirect=True)
    # print(image.eqr_coord.T, '\n', image_corners)
    pixels_error = np.linalg.norm(image.eqr_coord.T - image_corners, axis=1)
    mean_pixel_error = np.mean(pixels_error)
    std_pixel_error = np.std(pixels_error)

    euler = R.from_matrix(pointcloud.transform_matrix[:3, :3]).as_euler('xyz', degrees=True)
    # print('\nEuler angles from rotation matrix in degrees: \n', euler)
    # print('\nTranslation vector in meters: \n', pointcloud.transform_matrix[:3, 3])

    return euler, pointcloud.transform_matrix[:3, 3], mean_error, std_err, mean_pixel_error, std_pixel_error


def on_click(sel):
    # Get the index of the selected bar
    idx = sel.target.index
    on_click.selected_indices.append(idx)

def kabsch_plot(kabsch_errors, kabsch_std, label):
    """ Plotter function to plot and select Kabsch results
    
        :param kabsch_errors: array of Kabsch errors
        :param kabsch_std:    array of Kabsch errors standard deviation
        :param label:         string with the label of the plot
        
        :return: array of selected Kabsch errors
    """
    
    kabsch_mean = np.mean(kabsch_errors)
    
    # Print kabsch errors and standard deviation
    print('\nMean Kabsch error: ', kabsch_errors)
    print('\nKabsch error standard deviation: ', kabsch_std)

    # Plot kabsch errors with standard deviation bars and a line for the mean
    fig, ax = plt.subplots()
    plt.rc('axes', titlesize=12)
    plt.xticks(np.arange(len(kabsch_errors)))
    bars = ax.bar(np.arange(len(kabsch_errors)), kabsch_errors, yerr=kabsch_std, label='Mean Error', color='blue',
            ecolor='red', capsize=4)
    # bars = ax.bar(np.arange(len(kabsch_errors)), kabsch_errors, yerr=kabsch_std, label='Error de Kabsch', color='cornflowerblue',
    #         ecolor='red', capsize=4)
    ax.plot(np.arange(len(kabsch_errors)), np.ones(len(kabsch_errors)) * kabsch_mean,
            label='Mean error', color='green')
    # ax.plot(np.arange(len(kabsch_errors)), np.ones(len(kabsch_errors)) * kabsch_mean,
    #         label='Media de error de Kabsch', color='green')
    # plt.xlabel('Número de imagen')
    plt.xlabel('Image number')
    plt.ylabel('Error (m)')
    # Make xlabel and ylabel legend bigger
    ax.xaxis.label.set_size(18)
    ax.yaxis.label.set_size(18)
    # Make numbers in x and y axis bigger
    ax.tick_params(axis='both', labelsize=18)
    plt.title(label, size=18)
    plt.legend(prop={"size":18})
    
    on_click.selected_indices = []
    cursor = mplcursors.cursor(bars)
    cursor.connect('add', on_click)
    # Close plot when a key is pressed
    plt.waitforbuttonpress()
    plt.close()
    return on_click.selected_indices


def results_plot(rotations, translations):
    """ Plotter function to plot and select rotation and translation results
        
        :param rotations:    array of rotations
        :param translations: array of translations
        
        :return: array of the mean rotation
        :return: array of the mean translation
    """
    
    # Get the mean of the rotations and translations
    rotations = np.array(rotations)
    translations = np.array(translations)
    mean_rotation = np.mean(rotations, axis=0)
    mean_translation = np.mean(translations, axis=0)

    # Get the error of the rotations and translations
    rotations_error = rotations - np.repeat(mean_rotation, rotations.shape[0], axis=0).reshape(rotations.shape[1], rotations.shape[0]).T
    translations_error = translations - np.repeat(mean_translation, translations.shape[0], axis=0).reshape(rotations.shape[1], rotations.shape[0]).T

    # Print rotations and translations errors
    print("\nRotations: ", rotations)
    print("\nTranslations: ", translations)
    print("\nMean rotation: ", mean_rotation)
    print("\nMean translation: ", mean_translation)
    print("\nRotations errors: ", rotations_error)
    print("\nTranslations errors: ", translations_error)
    print("\nMean rotation error: ", np.std(rotations, axis=0))
    print("\nMean translation error: ", np.std(translations, axis=0))

    # Get the mean of the error of the rotations and translations
    mean_rotations_error = np.mean(abs(rotations_error), axis=0)
    mean_translation_error = np.mean(abs(translations_error), axis=0)
    
    plot_rotation = False
    
    # Plot rotation errors bars
    if plot_rotation:
        plt.figure()
        plt.xticks(np.arange(len(rotations_error)))
        plt.bar(np.arange(len(rotations_error)) - 0.3, abs(rotations_error[:, 0]), 0.3, label='x')
        plt.plot(np.arange(len(rotations_error)) - 0.3, np.ones(len(rotations_error)) * abs(mean_rotations_error[0]),
                    label='Mean x axis')
        plt.bar(np.arange(len(rotations_error)), abs(rotations_error[:, 1]), 0.3, label='y')
        plt.plot(np.arange(len(rotations_error)), np.ones(len(rotations_error)) * abs(mean_rotations_error[1]),
                    label='Mean y axis')
        plt.bar(np.arange(len(rotations_error)) + 0.3, abs(rotations_error[:, 2]), 0.3, label='z')
        plt.plot(np.arange(len(rotations_error)) + 0.3, np.ones(len(rotations_error)) * abs(mean_rotations_error[2]),
                    label='Mean z axis')
    else:
        # Get the mean rotation error for each row
        rotations_error = np.mean(abs(rotations_error), axis=1)
        mean_rotations_error = np.mean(abs(rotations_error))
        plt.figure()
        plt.xticks(np.arange(len(rotations_error)))
        plt.bar(np.arange(len(rotations_error)), rotations_error, label='std', color='blue')
        plt.plot(np.arange(len(rotations_error)),
                 np.ones(len(rotations_error)) * np.linalg.norm(mean_rotations_error),
                 label='Mean', color='red')
    plt.xlabel('Image number')
    plt.ylabel('Deviation from the mean (degrees)')
    # Make xlabel and ylabel legend bigger
    ax = plt.gca()
    ax.xaxis.label.set_size(18)
    ax.yaxis.label.set_size(18)
    plt.title('Deviation from the mean of rotations')
    ax.title.set_size(18)
    ax.tick_params(axis='both', labelsize=18)
    plt.legend(prop={"size":18})
    plt.show()
    
    # Plot translation errors bars
    plt.figure()
    plt.xticks(np.arange(len(translations_error)))
    plt.bar(np.arange(len(translations_error)),
            np.linalg.norm(translations_error, axis=1), label='std', color='blue')
    plt.plot(np.arange(len(translations_error)),
             np.ones(len(translations_error)) * np.linalg.norm(mean_translation_error),
             label='Mean', color='red')
    plt.xlabel('Image number')
    plt.ylabel('Standard deviation (m)')
    # Make xlabel and ylabel legend bigger
    ax = plt.gca()
    ax.xaxis.label.set_size(18)
    ax.yaxis.label.set_size(18)
    plt.title('Deviation from the mean of translations')
    ax.title.set_size(18)
    ax.tick_params(axis='both', labelsize=18)
    plt.legend(prop={"size":18})
    plt.show()

    # Print the results
    print("\nMean rotation error: ", mean_rotations_error)
    print("\nMean translation error: ", mean_translation_error)
    
    return mean_rotation, mean_translation


# Copy the exact same function as the one above but changing the labels to spanish
def results_plot_esp(rotations, translations):
    """ Plotter function to plot and select rotation and translation results
        
        :param rotations:    array of rotations
        :param translations: array of translations
        
        :return: array of the mean rotation
        :return: array of the mean translation
    """
    # Get the mean of the rotations and translations
    rotations = np.array(rotations)
    translations = np.array(translations)
    mean_rotation = np.mean(rotations, axis=0)
    mean_translation = np.mean(translations, axis=0)

    # Get the error of the rotations and translations
    rotations_error = rotations - np.repeat(mean_rotation, rotations.shape[0], axis=0).reshape(rotations.shape[1], rotations.shape[0]).T
    translations_error = translations - np.repeat(mean_translation, translations.shape[0], axis=0).reshape(rotations.shape[1], rotations.shape[0]).T

    # Print rotations and translations errors
    print("\nRotations: ", rotations)
    print("\nTranslations: ", translations)
    print("\nMean rotation: ", mean_rotation)
    print("\nMean translation: ", mean_translation)
    print("\nRotations errors: ", rotations_error)
    print("\nTranslations errors: ", translations_error)
    print("\nMean rotation error: ", np.std(rotations, axis=0))
    print("\nMean translation error: ", np.std(translations, axis=0))

    # Get the mean of the error of the rotations and translations
    mean_rotations_error = np.mean(abs(rotations_error), axis=0)
    mean_translation_error = np.mean(abs(translations_error), axis=0)

    # Plot rotation errors bars
    plt.figure()
    plt.xticks(np.arange(len(rotations_error)))
    plt.bar(np.arange(len(rotations_error)) - 0.3, abs(rotations_error[:, 0]), 0.3, label='x', color='red')
    plt.plot(np.arange(len(rotations_error)) - 0.3, np.ones(len(rotations_error)) * abs(mean_rotations_error[0]),
             label='Media eje x', color='red')
    plt.bar(np.arange(len(rotations_error)), abs(rotations_error[:, 1]), 0.3, label='y', color='green')
    plt.plot(np.arange(len(rotations_error)), np.ones(len(rotations_error)) * abs(mean_rotations_error[1]),
             label='Media eje y', color='green')
    plt.bar(np.arange(len(rotations_error)) + 0.3, abs(rotations_error[:, 2]), 0.3, label='z', color='blue')
    plt.plot(np.arange(len(rotations_error)) + 0.3, np.ones(len(rotations_error)) * abs(mean_rotations_error[2]),
             label='Media eje z', color='blue')
    plt.xlabel('Número de imagen')
    plt.ylabel('Desviación estándar (grados)')
    # Make xlabel and ylabel legend bigger
    ax = plt.gca()
    ax.xaxis.label.set_size(15)
    ax.yaxis.label.set_size(15)
    plt.title('Desviación estándar respecto la media de las rotaciones')
    ax.title.set_size(15)
    ax.tick_params(axis='both', labelsize=15)
    plt.legend(prop={"size":15})
    plt.show()
    
    # Plot translation errors bars
    plt.figure()
    plt.xticks(np.arange(len(translations_error)))
    plt.bar(np.arange(len(translations_error)),
            np.linalg.norm(translations_error, axis=1), label='Desviación estándar', color='cornflowerblue')
    plt.plot(np.arange(len(translations_error)),
             np.ones(len(translations_error)) * np.linalg.norm(mean_translation_error),
             label='Media', color='red')
    plt.xlabel('Número de imagen')
    plt.ylabel('Desviación estándar (m)')
    # Make xlabel and ylabel legend bigger
    ax = plt.gca()
    ax.xaxis.label.set_size(15)
    ax.yaxis.label.set_size(15)
    plt.title('Desviación estándar respecto la media de las traslaciones')
    ax.title.set_size(15)
    ax.tick_params(axis='both', labelsize=15)
    plt.legend(prop={"size":15})
    plt.show()

    # Print the results
    print("\nMean rotation error: ", mean_rotations_error)
    print("\nMean translation error: ", mean_translation_error)
    
    return mean_rotation, mean_translation
