import numpy as np
import os
import open3d as o3d
from plyfile import PlyData
import matplotlib.pyplot as plt
import matplotlib.image as mpi

from config import params
from image_utils import Image


def load_pc(path):
    """ Load PointCloud data from pcd or ply file. """
    extension = os.path.splitext(path)[1]
    if extension == ".pcd":
        p = o3d.io.read_point_cloud(path)
        return np.asarray(p.points, dtype=np.float32)
    if extension == ".ply":
        p = PlyData.read(path)
        fields = p.elements[0].data.dtype.names
        pointcloud = []
        for field in fields:
            # get the index of field
            idx = p.elements[0].data.dtype.names.index(field)
            if idx < 4:  # x, y and z fields
                pointcloud.append(p.elements[0].data[field])
        return np.asarray(pointcloud).T


def scale_to_255(a, mn, mx, dtype=np.uint8):
    """ Scales an array of values from specified min, max range to 0-255.
        Optionally specify the data type of the output (default is uint8).
    """
    return (((a - mn) / float(mx - mn)) * 255).astype(dtype)


class PointCloud:
    """ Class for point clouds and its methods. """

    v_res = np.deg2rad(params.lidar_vertical_resolution)
    h_res = np.deg2rad(params.lidar_horizontal_resolution)

    def __init__(self, lidar3d, image=None):
        """ Definition of point cloud attributes:
            :param lidar3d: Directory or array of points in LiDAR coordinates (x, y, z)/(x, y, z, r)
            :param image:   Directory or array of image points.
        """

        assert isinstance(lidar3d, str) or isinstance(lidar3d, np.ndarray), "lidar3d must be a directory pcd string or a matrix array. "
        if isinstance(lidar3d, str):
            self.lidar3d = load_pc(lidar3d)
            self.original_lidar3d = load_pc(lidar3d)
        else:
            self.lidar3d = lidar3d
            self.original_lidar3d = lidar3d

        if image is not None:
            assert isinstance(image, str) or isinstance(image, np.ndarray), "image must be a directory image string or a matrix array. "
            if isinstance(image, str):
                self.image = mpi.imread(image)
            else:
                self.image = image

        # Distance relative to origin when looked from top
        self.depth = np.sqrt(self.lidar3d[:, 0] ** 2 + self.lidar3d[:, 1] ** 2)
        self.original_depth = np.sqrt(self.original_lidar3d[:, 0] ** 2 + self.original_lidar3d[:, 1] ** 2)

        self.spherical_coord = None
        self.coord_img = None
        if self.lidar3d.shape[1] >= 4:
            self.reflectivity = self.lidar3d[:, 3]
            self.original_reflectivity = self.original_lidar3d[:, 3]
            self.lidar3d = self.lidar3d[:, :3]
            self.original_lidar3d = self.original_lidar3d[:, :3]
        else:
            self.reflectivity = None
            self.original_reflectivity = None
        self.transform_matrix = np.identity(4)
        if params.simulated == False:
            self.transform_matrix[2, 3] = -0.03618  # OUSTER TRANSFORM CORRECTION

    def __len__(self):
        return self.lidar3d.shape[0]

    def estimate_transform_matrix(self, lidar_pts, camera_pts):
        """ Estimate transform matrix between LiDAR and camera given 4

            :param lidar_pts:  array of 3D points from LiDAR coordinates
            :param camera_pts: array of 3D points from camera coordinates
        """

        assert lidar_pts.shape[0] == camera_pts.shape[0], 'There must be the same number of points.'

        N = lidar_pts.shape[0]  # total points

        centroid_A = np.mean(lidar_pts, axis=0)
        centroid_B = np.mean(camera_pts, axis=0)

        # center the points
        AA = lidar_pts - np.tile(centroid_A, (N, 1))
        BB = camera_pts - np.tile(centroid_B, (N, 1))

        # Get rotation matrix from svd
        H = np.dot(np.transpose(BB), AA)
        U, S, Vt = np.linalg.svd(H)
        R = np.dot(Vt.T, U.T)

        # special reflection case
        if np.linalg.det(R) < 0:
            Vt[2, :] *= -1
            R = np.dot(Vt.T, U.T)

        t = np.array(-R * np.matrix(centroid_B).T + np.matrix(centroid_A).T)

        self.transform_matrix = np.vstack([np.hstack([R, t]), np.array([0, 0, 0, 1])])

    def define_transform_matrix(self, r, t):
        """ Define transformation matrix manually. """

        # MATRIX EXTRACTED FROM LIDAR-CAMERA CALIBRATION PROCESS
        # t = [0.07842228, -0.04997922, -0.26663281]  # distances from camera to LiDAR
        # t = [ 0.13883695, -0.0543241, -0.27825683]  # distances from camera to LiDAR
        # t = [ 0.15881019, -0.04922911, -0.27629853]
        # r = [np.deg2rad(1.82945084) , np.deg2rad(-3.55761875), np.deg2rad(-1.2380633)]
        # r = [np.deg2rad(1.38220545) , np.deg2rad(- 2.24528872), np.deg2rad(- 0.98003899)]
        # r = [np.deg2rad(0.53921952) , np.deg2rad(-2.78353781), np.deg2rad(0.54010021)]

        r = [np.deg2rad(r[0]), np.deg2rad(r[1]), np.deg2rad(r[2])]
        Rx = np.array([[1, 0, 0, 0],
                       [0, np.cos(r[0]), -np.sin(r[0]), 0],
                       [0, np.sin(r[0]), np.cos(r[0]), 0],
                       [0, 0, 0, 1]])

        Ry = np.array([[np.cos(r[1]), 0, np.sin(r[1]), 0],
                       [0, 1, 0, 0],
                       [-np.sin(r[1]), 0, np.cos(r[1]), 0],
                       [0, 0, 0, 1]])

        Rz = np.array([[np.cos(r[2]), -np.sin(r[2]), 0, 0],
                       [np.sin(r[2]), np.cos(r[2]), 0, 0],
                       [0, 0, 1, 0],
                       [0, 0, 0, 1]])

        self.transform_matrix = np.dot(np.dot(Rz, Ry), Rx)
        self.transform_matrix[0][3] = t[0]
        self.transform_matrix[1][3] = t[1]
        self.transform_matrix[2][3] = t[2]
        # self.transform_matrix = [[-7.78983183e-04, -6.92083970e-02,  1.40196939e-01,  1.62669288e+00],
        #                          [-2.14531542e-01, -1.82117729e-02, -1.80218643e-02,  3.91008408e-01],
        #                          [-6.20808498e-03, -3.26799219e-01,  6.58804986e-01, -6.61006656e-02],
        #                          [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]]

    def transform_lidar_to_camera(self):
        """ Transforms the 3D LIDAR points to the camera 3D coordinates system. """

        points = np.vstack([self.lidar3d.T, np.ones(self.lidar3d.shape[0])])
        self.lidar3d = np.vstack(np.dot(self.transform_matrix, points)[0:3, :].T)
        self.depth = np.sqrt(self.lidar3d[:, 0] ** 2 + self.lidar3d[:, 1] ** 2)

    def distance_filter(self):
        """ Filter points by distance. """
        min_dist = 0.4
        self.lidar3d = self.lidar3d[self.depth > min_dist, :]
        if self.reflectivity is not None:
            self.reflectivity = self.reflectivity[self.depth > min_dist]
        self.depth = self.depth[self.depth > min_dist]

    def reflectivity_filter(self, percentage=0.05):
        """ Filter points by reflectivity. """
        min_reflectivity = percentage * 255
        if min_reflectivity != 0:
            self.lidar3d = self.lidar3d[self.reflectivity > min_reflectivity, :]
            self.depth = self.depth[self.reflectivity > min_reflectivity]
            self.reflectivity = self.reflectivity[self.reflectivity > min_reflectivity]

    def unfilter_lidar(self):
        self.lidar3d = self.original_lidar3d
        self.depth = self.original_depth
        self.reflectivity = self.original_reflectivity

    def projection(self, lidar2camera=1, projection_type=''):
        """ Takes points in a 3D space from LIDAR data and projects them onto a 2D space.
            The (0, 0) coordinates are in the middle of the 2D projection.
            :param: lidar2camera: flag to transform LiDAR points to camera reference system.
            :param: projection_type: Choose between 'cylindrical' or 'spherical'

            :return: ndarray of x and y coordinates of each point on the plane, being 0 the middle
        """

        if lidar2camera == 1:
            # Transform to camera 3D coordinates
            self.transform_lidar_to_camera()

        self.distance_filter()

        x_lidar = self.lidar3d[:, 0]
        y_lidar = self.lidar3d[:, 1]
        z_lidar = self.lidar3d[:, 2]

        # PROJECT INTO IMAGE COORDINATES
        if projection_type == 'cylindrical':
            x2d = np.arctan2(-y_lidar, x_lidar) / PointCloud.h_res
            y2d = (z_lidar / self.depth) / PointCloud.v_res
            # y2d = np.arctan2(z_lidar, self.depth) / v_res_rad
        else:  # spherical
            x2d = np.arctan2(-y_lidar, x_lidar) / PointCloud.h_res  # x2d = -np.arctan2(y_lidar, x_lidar) / h_res_rad
            y2d = np.arctan2(z_lidar, self.depth) / PointCloud.v_res

        return np.array([x2d, y2d]).T

    def get_spherical_coord(self, lidar2camera=True):
        """ Get coordinates from 3D LiDAR points onto the unit sphere.
            :param: lidar2camera: flag to transform LiDAR points to camera reference system.
        """

        if lidar2camera == True:
            # Transform to camera 3D coordinates
            self.transform_lidar_to_camera()

        # self.distance_filter()

        x_lidar = self.lidar3d[:, 0]
        y_lidar = self.lidar3d[:, 1]
        z_lidar = self.lidar3d[:, 2]

        # Project onto unit sphere
        n = np.linalg.norm(self.lidar3d, axis=1)
        x = x_lidar / n
        y = y_lidar / n
        z = z_lidar / n

        self.spherical_coord = np.vstack((x, y, z))


class Visualizer(PointCloud):
    """ PointCloud visualizer methods. """
    def __init__(self, lidar3d: np.ndarray, image: np.ndarray, value = 'depth', cmap = 'jet'):
        super().__init__(lidar3d, image)
        self.value = value
        self.cmap = cmap
        self.pixel_values = None
        self.lidar_corners = None
        self.camera_corners = None

    def encode_values(self, d_range=None):
        """ What data to use to encode the value for each pixel.
            :param d_range: If tuple is provided, it is used for clipping distance values to be within a min and max range
        """

        if self.value == 'reflectivity':
            assert self.reflectivity is not None, "There is no reflectivity data in point cloud data file."
            self.pixel_values = self.reflectivity  # Reflectivity
        elif self.value == 'height':
            self.pixel_values = self.lidar3d[:, 2]
        elif d_range is not None:
            self.pixel_values = -np.clip(self.depth, a_min=d_range[0], a_max=d_range[1])
        else:
            self.pixel_values = self.depth

    def lidar_to_panorama(self, projection_type='', d_range=None, saveto=None):
        """ Takes points in 3D space from LIDAR data and projects them to a 2D image and saves that image.
            :param: projection_type: Choose between 'cylindrical' or 'spherical'
            :param: d_range:         If tuple is provided, it is used for clipping distance values to be within a min and max range
            :param: saveto:          If a string is provided, it saves the image as that filename given
        """

        # GET 2D PROJECTED POINTS
        points2d = self.projection(lidar2camera=1, projection_type=projection_type)

        # SHIFT COORDINATES TO MAKE 0,0 THE MINIMUM
        x_img = points2d[:, 0]
        x_min = -2 * np.pi / PointCloud.h_res / 2  # Theoretical min x value based on sensor specs
        x_img -= x_min  # Shift
        x_max = int(2 * np.pi / PointCloud.h_res)  # Theoretical max x value after shifting

        y_img = points2d[:, 1]
        y_min = np.amin(y_img)  # min y value
        y_img -= y_min  # Shift
        y_max = int(np.amax(y_img))  # max x value

        self.encode_values(d_range=d_range)

        # CONVERT TO IMAGE ARRAY
        img = np.zeros([y_max + 1, x_max + 1], dtype=np.uint8)
        x, y = np.trunc(y_img).astype(np.int32), np.trunc(x_img).astype(np.int32)
        img[x, y] = scale_to_255(self.pixel_values, mn=np.amin(abs(self.pixel_values)), mx=np.amax(abs(self.pixel_values)))

        # PLOT THE IMAGE
        fig, ax = plt.subplots()
        ax.imshow(img, cmap=self.cmap)
        ax.xaxis.set_visible(False)  # Do not draw axis tick marks
        ax.yaxis.set_visible(False)  # Do not draw axis tick marks
        plt.xlim([0, x_max])  # prevent drawing empty space outside of horizontal FOV
        plt.ylim([0, y_max])  # prevent drawing empty space outside of vertical FOV

        # SAVE THE IMAGE
        if saveto is not None:
            img = np.flip(img, 0)
            mpi.imsave(saveto, img, cmap=self.cmap)

    def lidar_onto_image(self, cam_model=None, fisheye=0, d_range=None, saveto=None):
        """ Shows 3D LiDAR points onto its matched image obtained at the same of time. Optionally saves the result to specified filename.
            :param cam_model: Fisheye camera model loaded from calib.txt
            :param fisheye:   Project LiDAR onto fisheye image converted to spherical projection if 0
                              Project LiDAR onto fisheye image if 1.
            :param d_range:   If tuple is provided, it is used for clipping distance values to be within a min and max range
            :param saveto:    If a string is provided, it saves the image as this filename
        """

        self.unfilter_lidar()
        self.get_spherical_coord()
        self.encode_values(d_range=d_range)

        if fisheye == 0:
            # GET LIDAR PROJECTED ONTO SPHERICAL PROJECTION
            lidar_proj = Image(image=self.image, cam_model=cam_model, spherical_image=params.spherical, points_values=self.pixel_values)
            if params.spherical == False:
                lidar_proj.fish2equirect()
            lidar_proj.sphere_coord = self.spherical_coord
            lidar_proj.lidar_projection(pixel_points=0)
            plt.imshow(lidar_proj.eqr_image)
            plt.xticks([])
            plt.yticks([])
            plt.gca().set_aspect('equal', adjustable='box')
            plt.scatter(x=lidar_proj.eqr_coord[0], y=lidar_proj.eqr_coord[1], c=lidar_proj.points_values, s=0.05, cmap=self.cmap)

            # Plot lidar and camera corners
            if self.lidar_corners is not None:
                lcorners = PointCloud(self.lidar_corners, self.image)
                lcorners.transform_matrix = self.transform_matrix
                lcorners.get_spherical_coord()
                lidar_proj.sphere_coord = lcorners.spherical_coord
                lidar_proj.lidar_projection(pixel_points=0)
                plt.scatter(x=lidar_proj.eqr_coord[0], y=lidar_proj.eqr_coord[1], c='k', s=5)
                plt.plot([lidar_proj.eqr_coord[0, 0], lidar_proj.eqr_coord[0, 2], lidar_proj.eqr_coord[0, 3], lidar_proj.eqr_coord[0, 1], lidar_proj.eqr_coord[0, 0]],
                         [lidar_proj.eqr_coord[1, 0], lidar_proj.eqr_coord[1, 2], lidar_proj.eqr_coord[1, 3], lidar_proj.eqr_coord[1, 1], lidar_proj.eqr_coord[1, 0]],
                         'r', linewidth=1)
            if self.camera_corners is not None:
                ccorners = PointCloud(self.camera_corners, self.image)
                ccorners.get_spherical_coord()
                lidar_proj.sphere_coord = ccorners.spherical_coord
                lidar_proj.lidar_projection(pixel_points=0)
                plt.scatter(x=lidar_proj.eqr_coord[0], y=lidar_proj.eqr_coord[1], c='g', s=5)
                # plt.plot([lidar_proj.eqr_coord[0, 0], lidar_proj.eqr_coord[0, 2], lidar_proj.eqr_coord[0, 3], lidar_proj.eqr_coord[0, 1], lidar_proj.eqr_coord[0, 0]],
                #          [lidar_proj.eqr_coord[1, 0], lidar_proj.eqr_coord[1, 2], lidar_proj.eqr_coord[1, 3], lidar_proj.eqr_coord[1, 1], lidar_proj.eqr_coord[1, 0]],
                #          'g', linewidth=1)

        else:
            assert params.spherical == False, "The image must be a fisheye image. Check spherical parameter in config.yaml file."
            lidar_fish = Image(image=self.image, cam_model=cam_model, points_values=self.pixel_values)
            lidar_fish.sphere_coord = self.spherical_coord
            lidar_fish.change2camera_ref_system()
            lidar_fish.sphere2fisheye()
            lidar_fish.check_image_limits()
            plt.imshow(lidar_fish.image)
            plt.xticks([])
            plt.yticks([])
            u, v =  lidar_fish.spherical_proj[1], lidar_fish.spherical_proj[0]
            plt.scatter(x=u, y=v, c=lidar_fish.points_values, s=0.01, cmap=self.cmap)

            # Plot lidar and camera corners
            if self.lidar_corners is not None:
                lcorners = PointCloud(self.lidar_corners, self.image)
                lcorners.transform_matrix = self.transform_matrix
                lcorners.get_spherical_coord()
                lidar_fish.sphere_coord = lcorners.spherical_coord
                lidar_fish.change2camera_ref_system()
                lidar_fish.sphere2fisheye()
                lidar_fish.check_image_limits()
                u, v =  lidar_fish.spherical_proj[1], lidar_fish.spherical_proj[0]
                plt.scatter(x=u, y=v, c='r', s=5)
                plt.plot([u[0], u[2], u[3], u[1], u[0]], [v[0], v[2], v[3], v[1], v[0]], 'k', linewidth=1)
            if self.camera_corners is not None:
                ccorners = PointCloud(self.camera_corners, self.image)
                ccorners.get_spherical_coord(0)
                lidar_fish.sphere_coord = ccorners.spherical_coord
                lidar_fish.change2camera_ref_system()
                lidar_fish.sphere2fisheye()
                lidar_fish.check_image_limits()
                u, v =  lidar_fish.spherical_proj[1], lidar_fish.spherical_proj[0]
                plt.scatter(x=u, y=v, c='g', s=5)
                # plt.plot([u[0], u[2], u[3], u[1], u[0]], [v[0], v[2], v[3], v[1], v[0]], 'g', linewidth=1)

        if saveto is not None:
            plt.savefig(saveto, dpi=300, bbox_inches='tight')
