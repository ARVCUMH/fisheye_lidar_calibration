import numpy as np
import matplotlib.image as mpi
import matplotlib.cm as cm
import time
from scipy.spatial.transform import Rotation as R
from skimage.draw import line


class CamModel:
    """ Class for cam model (fisheye model). """
    def __init__(self, filename: str):
        self.filename = filename
        self.length_pol = 4
        self.pol = [0, 0, 0, 0]
        self.length_invpol = 4
        self.invpol = [0, 0, 0, 0]
        self.height = 1080
        self.width = 1920
        self.xc = self.height / 2
        self.yc = self.width / 2
        self.c = 1
        self.d = 0
        self.e = 0
        self.get_cam_model()

    def get_cam_model(self):
        """ Reads .txt file containing the camera calibration parameters exported from the Matlab toolbox. """
        with open(self.filename) as f:
            lines = [l for l in f]
            l = lines[2]
            data = l.split()
            self.length_pol = int(data[0])
            self.pol = [float(d) for d in data[1:]]

            l = lines[6]
            data = l.split()
            self.length_invpol = int(data[0])
            self.invpol = [float(d) for d in data[1:]]

            l = lines[10]
            data = l.split()
            self.xc = float(data[0])
            self.yc = float(data[1])

            l = lines[14]
            data = l.split()
            self.c = float(data[0])
            self.d = float(data[1])
            self.e = float(data[2])

            l = lines[18]
            data = l.split()
            self.height = int(data[0])
            self.width = int(data[1])

class Image:
    """ Class for images and its methods. """

    def __init__(self, image, cam_model: CamModel=None, fov=None, spherical_image: bool=False, xyxy=None, points_values=None):
        """ Definition of point cloud attributes:
            :param image:           Array of image points.
            :param cam_model:       CamModel object with camera model info.
            :param fov:             Field of view of the camera in degrees.
            :param spherical_image: Boolean to indicate if the image is a spherical projection or not.
            :param xyxy:            Array of equirectangular pixels coordinates.
            :param points_values:   Array of LiDAR values points to be projected
        """

        assert isinstance(image, str) or isinstance(image, np.ndarray), "image must be an image directory string or a matrix array. "
        if isinstance(image, str):
            self.image = mpi.imread(image)
        else:
            self.image = image
        if fov is None:
            assert isinstance(cam_model, CamModel), "CamModel object or fov must be given."
        self.cam_model = cam_model
        if fov is not None:
            self.FOV = np.deg2rad(fov)
        else:
            self.FOV = None
        self.xyxy = xyxy
        self.points_values = points_values
        self.rotm = R.from_euler('xyz', [np.pi / 2, 0, np.pi / 2]).as_matrix()
        self.eqr_coord = None
        self.norm_coord = None
        self.sphere_coord = None
        self.fisheye_coord = None
        if spherical_image:
            self.eqr_image = self.image
        # elif self.xyxy is not None:
        #     self.eqr_image = np.zeros([self.xyxy[3] - self.xyxy[1], self.xyxy[2] - self.xyxy[0], 3])
        else:
            self.eqr_image = np.zeros([self.image.shape[0], 2 * self.image.shape[0], 3])

    def square_fisheye(self, h_cut=0, v_cut=0):
        """ Cut fisheye image to a square image. """

        # Horizontal cut
        if h_cut != 0:
            self.image = self.image[:, h_cut:-h_cut, :]
            n_rows = np.round((self.image.shape[1] - self.image.shape[0]) / 2).astype(int)
            self.image = np.vstack((np.zeros([n_rows, self.image.shape[1], 3]), self.image))
            self.image = np.vstack((self.image, np.zeros([n_rows, self.image.shape[1], 3])))

        # Vertical cut
        if v_cut != 0:
            self.image = self.image[v_cut:-v_cut, :, :]
            n_columns = np.round((self.image.shape[0] - self.image.shape[1]) / 2).astype(int)
            self.image = np.hstack((np.zeros([self.image.shape[0], n_columns, 3]), self.image))
            self.image = np.hstack((self.image, np.zeros([self.image.shape[0], n_columns, 3])))

    def get_equirect(self):
        """ Get equirectangular pixels coordinates. """

        [u_out, v_out] = np.meshgrid(np.arange(0, self.eqr_image.shape[1]), np.arange(0, self.eqr_image.shape[0]))
        # if self.xyxy is None:
        #     [u_out, v_out] = np.meshgrid(np.arange(0, self.eqr_image.shape[1]), np.arange(0, self.eqr_image.shape[0]))
        # else: 
        #     [u_out, v_out] = np.meshgrid(np.arange(self.xyxy[0], self.xyxy[2]), np.arange(self.xyxy[1], self.xyxy[3]))
        u_out = np.ravel(u_out)
        v_out = np.ravel(v_out)
        self.eqr_coord = np.vstack((u_out, v_out))

    def image2norm(self):
        """ Normalize equirectangular image. """

        # Vertical scale factor from the normalized image to the original image
        ay = self.eqr_image.shape[0] / 2

        # Horizontal scale factor from the normalized image to the original image
        ax = self.eqr_image.shape[1] / 2

        A = np.array([[ax, 0], [0, ay]])
        D = np.array([[1, 0], [0, -1]])

        u = self.eqr_coord[0]
        v = self.eqr_coord[1]
        M = np.linalg.inv(A * D)
        x = M[0, 0] * (u - ax * np.ones(np.size(u))) + M[0, 1] * (v - ay * np.ones(np.size(u)))
        y = M[1, 0] * (u - ax * np.ones(np.size(u))) + M[1, 1] * (v - ay * np.ones(np.size(u)))

        self.norm_coord = np.vstack((x, y))

    def norm2image(self, equirect = False):
        """ Denormalize to get original equirectangular image. """

        if equirect:
            ay = self.eqr_image.shape[0] / 2
            ax = self.eqr_image.shape[1] / 2
        else:
            ay = self.image.shape[0] / 2
            ax = self.image.shape[1] / 2

        A = np.array([[ax, 0], [0, ay]])
        D = np.array([[1, 0], [0, -1]])

        x = self.norm_coord[0]
        y = self.norm_coord[1]
        M = np.multiply(A, D)
        u = M[0, 0] * x + M[0, 1] * y + ax * np.ones(np.size(y))
        v = M[1, 0] * x + M[1, 1] * y + ay * np.ones(np.size(y))

        # Extra line for converting last u coordinate out of boundaries
        u = np.clip(u, None, self.eqr_image.shape[1] - 1)

        if equirect:
            self.eqr_coord = np.vstack((u, v))
        else:
            self.fisheye_coord = np.vstack((u, v))

    def equirect2sphere(self):
        """ Get spherical coordinates from normalized x and y coordinates. """

        # Normalized equirectangular coordinates to longitude and latitude(long, lat)
        long = np.pi * ( - self.norm_coord[0])
        lat = self.norm_coord[1] * np.pi / 2

        # Longitude and latitude to the unit sphere coordinate system(Px, Py, Pz)
        Px = np.multiply(np.cos(lat), np.cos(long))
        Py = np.multiply(np.cos(lat), np.sin(long))
        Pz = np.sin(lat)
        self.sphere_coord = np.vstack((Px, Py, Pz))

    def sphere2equirect(self):
        """ Get normalized equirect coordinates from unit spherical coordinates. """

        # Unit sphere coordinate system(Px, Py, Pz)
        Px, Py, Pz = self.sphere_coord[0], self.sphere_coord[1], self.sphere_coord[2]

        # Unit sphere coordinates to longitude and latitude(long, lat)
        long = np.arctan2(Py, Px)
        lat = np.arctan2(Pz, np.linalg.norm([Px, Py], axis=0))

        # Normalized equirectangular coordinates
        x = - long / np.pi
        y = 2 * lat / np.pi
        self.norm_coord = np.vstack((x, y))

    def change2camera_ref_system(self):
        """ Change sphere coordinates to the camera reference system. """

        # rotm is the orientation of the camera frame with respect to the world frame
        if self.rotm.shape[1] == 3:
            self.sphere_coord = np.dot(self.rotm.transpose(), self.sphere_coord)
        else:
            self.sphere_coord = self.rotm[0:3, 0:3].transpose() * self.sphere_coord + self.rotm[0:3, 4]

    def sphere2fisheye(self):
        """ Get fisheye coordinates from spherical coordinates. """
        if self.cam_model is None:
            theta = np.arctan2(self.sphere_coord[1], self.sphere_coord[0])
            phi = np.arctan2(np.linalg.norm(self.sphere_coord[:2], axis=0), self.sphere_coord[2])
            rho = phi * (2/self.FOV)
            x = rho * np.cos(theta)
            y = rho * np.sin(theta)
            x = np.flip(x, axis=0)
            self.norm_coord = np.vstack((x, y))
            self.norm2image(equirect=False)
            
        else:
            # Transform into the camera model coordinate system
            r = R.from_euler('yzx', [np.pi, np.pi / 2, 0]).as_matrix()
            P = np.matmul(r, self.sphere_coord)

            # Calculate the projection on the fisheye image using the camera model
            self.world2cam(P)  # [rows, col]

    def world2cam(self, points3D):
        """ Projects 3D points on the image and returns the pixel coordinates. """

        proj = [[], []]
        norm = np.linalg.norm(np.array(points3D[:2]), axis=0)
        for k in range(np.shape(points3D)[1]):
            if norm[k] != 0:
                theta = np.arctan(points3D[2][k] / norm[k])
                invnorm = 1.0 / norm[k]
                rho = self.cam_model.invpol[0]
                t_i = 1.0

                for i in range(1, self.cam_model.length_invpol):
                    t_i *= theta
                    rho += t_i * self.cam_model.invpol[i]

                x = points3D[0][k] * invnorm * rho
                y = points3D[1][k] * invnorm * rho

                proj[0].append(x * self.cam_model.c + y * self.cam_model.d + self.cam_model.xc)
                proj[1].append(x * self.cam_model.e + y + self.cam_model.yc)
            else:
                proj[0].append(self.cam_model.xc)
                proj[1].append(self.cam_model.yc)

        self.fisheye_coord = np.array([proj[1], proj[0]])

    def fisheye2sphere(self):
        """ Get spherical coordinates from fisheye coordinates. """

        # Calculate the projection on the sphere using the camera model
        self.cam2world(np.round(self.fisheye_coord).astype(int))

        r = R.from_euler('yzx', [np.pi / 2, np.pi, 0]).as_matrix()
        self.sphere_coord = np.matmul(r, self.sphere_coord)

    def cam2world(self, points2D):
        """ Returns the 3D points projected on the sphere from the image pixels. """

        points3D = [[], [], []]
        invdet = 1.0 / (self.cam_model.c - self.cam_model.d * self.cam_model.e)
        for k in range(np.shape(points2D)[1]):
            xp = invdet * ((points2D[0][k] - self.cam_model.xc) - self.cam_model.d * (points2D[1][k] - self.cam_model.yc))
            yp = invdet * (-self.cam_model.e * (points2D[0][k] - self.cam_model.xc) + self.cam_model.c * (points2D[1][k] - self.cam_model.yc))
            zp = self.cam_model.pol[0]

            r = np.linalg.norm([xp, yp])
            r_i = 1.0

            for i in range(1, self.cam_model.length_pol):
                r_i *= r
                zp += r_i * self.cam_model.pol[i]

            invnorm = 1.0 / np.linalg.norm([xp, yp, zp])

            points3D[0].append(invnorm * xp)
            points3D[1].append(invnorm * yp)
            points3D[2].append(invnorm * zp)

        self.sphere_coord = np.array(points3D)

    def check_image_limits(self):
        """ Delete image and projection points which are out of limits. """

        proj = np.round(self.fisheye_coord).astype(int)
        limits = (proj >= 0) & (proj[0] < self.image.shape[1]) & (proj[1] < self.image.shape[0])
        if self.xyxy is not None:
            limits2 = (self.eqr_coord[0] >= self.xyxy[0]) & (self.eqr_coord[0] < self.xyxy[2]) & (self.eqr_coord[1] >= self.xyxy[1]) & (self.eqr_coord[1] < self.xyxy[3])
            limits = limits & limits2
        self.filtered_index = np.logical_and.reduce(limits)
        self.fisheye_coord = self.fisheye_coord[:, self.filtered_index]
        if self.eqr_coord is not None:
            self.eqr_coord = self.eqr_coord[:, self.filtered_index]
        if self.points_values is not None and len(self.points_values) == len(self.filtered_index):
            self.points_values = self.points_values[self.filtered_index]

    def fish2equirect(self):
        """ Converts a fisheye image into an equirectangular image. """

        u_o, v_o, u_i, v_i = self.fisheye2equirect(self)
        self.fromfisheye2equirect(self, u_o, v_o, u_i, v_i)

    def fisheye2equirect(self):
        self.get_equirect()
        self.image2norm()
        self.equirect2sphere()
        self.change2camera_ref_system()
        self.sphere2fisheye()
        self.check_image_limits()
        u_o, v_o = self.eqr_coord[0], self.eqr_coord[1]
        u_i, v_i = np.round(self.fisheye_coord[0]).astype(int), np.round(self.fisheye_coord[1]).astype(int)
        return u_o, v_o, u_i, v_i

    def fromfisheye2equirect(self, u_o, v_o, u_i, v_i):
        for tmp in range(v_o.shape[0]):
            self.eqr_image[v_o[tmp], u_o[tmp], :] = self.image[v_i[tmp], u_i[tmp], :3]


    def lidar_projection(self, pixel_points=False):
        """ Projects LiDAR points onto the equirectangular image.

            :param pixel_points: if != False, projects point cloud points onto the equirectangular projection as pixels.
                                 If the equirectangular image is wanted, fish2equirect has to be called before outside.
        """

        self.sphere2equirect()
        self.norm2image(equirect=True)
        if pixel_points != False:
            self.eqr_coord = np.round(self.eqr_coord).astype(int)
            self.points_values = cm.jet(self.points_values / np.amax(self.points_values))
            u_o, v_o = np.round(self.eqr_coord[0]).astype(int), np.round(self.eqr_coord[1]).astype(int)
            for tmp in range(v_o.shape[0]):
                self.eqr_image[v_o[tmp], u_o[tmp], :] = self.points_values[tmp, :3]

    def line_projection(self):
        """ Projects LiDAR points onto the equirectangular image. """

        self.sphere2equirect()
        self.norm2image(equirect=True)
        self.eqr_coord = np.round(self.eqr_coord).astype(int)
        self.points_values = cm.jet(self.points_values / np.amax(self.points_values))
        u_o, v_o = np.round(self.eqr_coord[0]).astype(int), np.round(self.eqr_coord[1]).astype(int)
        r, c = np.array([]), np.array([])
        values = np.array([[], [], []]).T
        for i in range(len(u_o) - 1):
            rr, cc = line(v_o[i], u_o[i], v_o[i+1], u_o[i+1])
            self.eqr_image[rr, cc, :] = self.points_values[i, :3]
            r, c = np.append(r, rr), np.append(c, cc)
            # append self.values to values as many times as the length of rr. shape of values is (len(rr), 3)
            values = np.vstack([values, np.tile(self.points_values[i, :3], (len(rr), 1))])

        return r, c, values
