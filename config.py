import yaml


# Extract configuration parameters from config.yaml file
with open('config.yaml', 'r') as file:
    param_data = yaml.safe_load(file)


class PathParameters:
    def __init__(self):
        self.images_path = param_data['images_path']
        self.spherical = param_data['spherical']
        self.pointclouds_path = param_data['pointclouds_path']
        self.calibration_file = param_data['calibration_file']

class ExperimentParameters:
    def __init__(self):
        self.planes_sizes = param_data['planes_sizes']
        self.planes_colours = param_data['planes_colours']
        self.lidar_vertical_resolution = param_data['lidar_vertical_resolution']
        self.lidar_horizontal_resolution = param_data['lidar_horizontal_resolution']
        self.show_lidar_onto_image = param_data['show_lidar_onto_image']

class LidarParameters:
    def __init__(self):
        self.reflectivity_threshold = param_data['reflectivity_threshold']
        self.radius_kdtree = param_data['radius_kdtree']
        self.rotations = param_data['rotations']
        self.show_lidar_plane = param_data['show_lidar_plane']

class ImageParameters:
    def __init__(self):
        # self.simulation = param_data['simulation']
        self.simulated = param_data['simulated']
        self.corner_detection_mode = param_data['corner_detection_mode']
        self.selection_mode = param_data['selection_mode']
        self.model_path = param_data['model_path']
        self.model_type = param_data['model_type']
        self.device = param_data['device']
        self.dilation = param_data['dilation']
        self.kernel_size = param_data['kernel_size']
        self.contour_distance_threshold = param_data['contour_distance_threshold']
        self.ransac_iterations = param_data['ransac_iterations']
        self.show_segmented_plane = param_data['show_segmented_plane']
        
class SaveParameters:
    def __init__(self):
        self.save_path = param_data['save_path']
        self.results_file = param_data['results_file']
        self.save_results = param_data['save_results']
        self.data_path = param_data['data_path']
        self.data_file = param_data['data_file']
        self.save_data = param_data['save_data']
        self.plots_path = param_data['plots_path']
        self.save_plots = param_data['save_plots']
        self.load_images_path = param_data['load_images_path']
        self.load_pointclouds_path = param_data['load_pointclouds_path']
        self.outputs_file = param_data['outputs_file']


class Parameters(PathParameters, ExperimentParameters, LidarParameters, ImageParameters, SaveParameters):
    def __init__(self):
        super(Parameters, self).__init__()
        super(PathParameters, self).__init__()
        super(ExperimentParameters, self).__init__()
        super(LidarParameters, self).__init__()
        super(ImageParameters, self).__init__()
        super(SaveParameters, self).__init__()
        
params = Parameters()
