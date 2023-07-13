import numpy as np
import matplotlib.image as mpi
import matplotlib.cm as cm
from scipy.spatial.transform import Rotation as R


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

    def __init__(self, image, cam_model: CamModel, points_values=None):
        """ Definition of point cloud attributes:
            :param image:         Array of image points.
            :param cam_model:     CamModel object with camera model info.
            :param points_values: Array of LiDAR values points to be projected
        """

        assert isinstance(image, str) or isinstance(image, np.ndarray), "image must be an image directory string or a matrix array. "
        if isinstance(image, str):
            self.image = mpi.imread(image)
        else:
            self.image = image
        self.cam_model = cam_model
        self.points_values = points_values
        self.rotm = R.from_euler('xyz', [np.pi / 2, 0, np.pi / 2]).as_matrix()
        self.eqr_coord = None
        self.norm_coord = None
        self.sphere_coord = None
        self.spherical_proj = None
        self.eqr_image = np.zeros([self.image.shape[0], 2 * self.image.shape[0], 3])

    def get_equirect(self):
        """ Get equirectangular pixels coordinates. """

        [u_out, v_out] = np.meshgrid(np.arange(0, self.eqr_image.shape[1]), np.arange(0, self.eqr_image.shape[0]))
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

    def norm2image(self):
        """ Denormalize to get original equirectangular image. """

        # Vertical scale factor from the normalized image to the original image
        ay = self.eqr_image.shape[0] / 2

        # Horizontal scale factor from the normalized image to the original image
        ax = self.eqr_image.shape[1] / 2

        A = np.array([[ax, 0], [0, ay]])
        D = np.array([[1, 0], [0, -1]])

        x = self.norm_coord[0]
        y = self.norm_coord[1]
        M = np.multiply(A, D)
        u = M[0, 0] * x + M[0, 1] * y + ax * np.ones(np.size(y))
        v = M[1, 0] * x + M[1, 1] * y + ay * np.ones(np.size(y))

        # Extra line for converting last u coordinate out of boundaries
        u = np.clip(u, None, self.eqr_image.shape[1] - 1)

        self.eqr_coord = np.vstack((u, v))

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

        self.spherical_proj = np.array(proj)

    def fisheye2sphere(self):
        """ Get spherical coordinates from fisheye coordinates. """

        # Calculate the projection on the sphere using the camera model
        self.cam2world(np.round(self.spherical_proj).astype(int))

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

        proj = np.round(self.spherical_proj).astype(int)
        limits = (proj >= 0) & (proj[0] < self.image.shape[0]) & (proj[1] < self.image.shape[1])
        self.filtered_index = np.logical_and.reduce(limits)
        self.spherical_proj = self.spherical_proj[:, self.filtered_index]
        if self.eqr_coord is not None:
            self.eqr_coord = self.eqr_coord[:, self.filtered_index]
        if self.points_values is not None and len(self.points_values) == len(self.filtered_index):
            self.points_values = self.points_values[self.filtered_index]

    def fish2equirect(self):
        """ Converts a fisheye image into an equirectangular image. """

        self.get_equirect()
        self.image2norm()
        self.equirect2sphere()
        self.change2camera_ref_system()
        self.sphere2fisheye()
        self.check_image_limits()
        u_o, v_o = self.eqr_coord[0], self.eqr_coord[1]
        u_i, v_i = np.round(self.spherical_proj[1]).astype(int), np.round(self.spherical_proj[0]).astype(int)
        for tmp in range(v_o.shape[0]):
            self.eqr_image[v_o[tmp], u_o[tmp], :] = self.image[v_i[tmp], u_i[tmp], :3]

    def lidar_projection(self, pixel_points=0):
        """ Projects LiDAR points onto the equirectangular image.

            :param pixel_points: if != 0, projects point cloud points onto the equirectangular projection as pixels.
                                 If the equirectangular image is wanted, fish2equirect has to be called before outside.
        """

        self.sphere2equirect()
        self.norm2image()
        if pixel_points != 0:
            self.eqr_coord = np.round(self.eqr_coord).astype(int)
            self.points_values = cm.jet(self.points_values / np.amin(self.points_values))
            u_o, v_o = np.round(self.eqr_coord[0]).astype(int), np.round(self.eqr_coord[1]).astype(int)
            for tmp in range(v_o.shape[0]):
                self.eqr_image[v_o[tmp], u_o[tmp], :] = self.points_values[tmp, :3]
