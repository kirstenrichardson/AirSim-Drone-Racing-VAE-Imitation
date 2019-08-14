import cv2
import numpy as np

import os
import sys
curr_dir = os.path.dirname(os.path.abspath(__file__))

# import airsim
airsim_path = os.path.join(curr_dir, '..', '..', 'airsim')
sys.path.insert(0, airsim_path)
import setup_path
import airsim
print(os.path.abspath(airsim.__file__))
from airsim.types import Pose, Vector3r, Quaternionr
import airsim.types
import airsim.utils
import time

#TODO: remove this normally -- just for debugging
# import gate regressor
models_path = os.path.join(curr_dir, '..', '..', 'gate_pose_regressor')
sys.path.insert(0, models_path)
import gate_pose_regressor.gate_regressor


# import utils
models_path = os.path.join(curr_dir, '..', '..', 'racing_utils')
sys.path.insert(0, models_path)
import racing_utils

GATE_YAW_RANGE = [-np.pi, np.pi]  # world theta gate
# GATE_YAW_RANGE = [-1, 1]  # world theta gate -- this range is btw -pi and +pi
UAV_X_RANGE = [-30, 30] # world x quad
UAV_Y_RANGE = [-30, 30] # world y quad
UAV_Z_RANGE = [-2, -3] # world z quad

UAV_YAW_RANGE = [-np.pi, np.pi]  #[-eps, eps] [-np.pi/4, np.pi/4]
eps = np.pi/10.0  # 18 degrees
UAV_PITCH_RANGE = [-eps, eps]  #[-np.pi/4, np.pi/4]
UAV_ROLL_RANGE = [-eps, eps]  #[-np.pi/4, np.pi/4]

R_RANGE = [0.1, 20]  # in meters
CAM_FOV = 90.0*0.85  # in degrees -- needs to be a bit smaller than 90 in fact because of cone vs. square


class PoseSampler:
    def __init__(self, num_samples, dataset_path, with_gate=True):
        self.num_samples = num_samples
        self.base_path = dataset_path
        self.csv_path = os.path.join(self.base_path, 'gate_training_data.csv')
        self.curr_idx = 0
        self.with_gate = with_gate
        self.client = airsim.MultirotorClient()

        self.client.confirmConnection()
        self.client.simLoadLevel('Soccer_Field_Easy')
        time.sleep(4)
        self.client = airsim.MultirotorClient()
        self.configureEnvironment()
        # TODO: remove this normally -- just for debugging
        # path_weights = '/home/rb/data/model_outputs/reg_5/reg_model_20.ckpt'
        path_weights = '/home/rb/data/model_outputs/cmvae_9/cmvae_model_15.ckpt'
        self.gate_regressor = gate_pose_regressor.gate_regressor.GateRegressor(regressor_type='cmvae', path_weights=path_weights)


    def update(self):
        '''
        convetion of names:
        p_a_b: pose of frame b relative to frame a
        t_a_b: translation vector from a to b
        q_a_b: rotation quaternion from a to b
        o: origin
        b: UAV body frame
        g: gate frame
        '''
        # create and set pose for the quad
        p_o_b, phi_base = racing_utils.geom_utils.randomQuadPose(UAV_X_RANGE, UAV_Y_RANGE, UAV_Z_RANGE, UAV_YAW_RANGE, UAV_PITCH_RANGE, UAV_ROLL_RANGE)
        self.client.simSetVehiclePose(p_o_b, True)
        # create and set gate pose relative to the quad
        p_o_g, r, theta, psi, phi_rel = racing_utils.geom_utils.randomGatePose(p_o_b, phi_base, R_RANGE, CAM_FOV)
        # p_o_g_new = racing_utils.geom_utils.debugRelativeOrientation(p_o_b, p_o_g, -np.pi/1.0)
        # self.client.simSetObjectPose(self.tgt_name, p_o_g_new, True)
        if self.with_gate:
            self.client.simSetObjectPose(self.tgt_name, p_o_g, True)
            # self.client.plot_tf([p_o_g], duration=20.0)
        # request quad img from AirSim
        image_response = self.client.simGetImages([airsim.ImageRequest('0', airsim.ImageType.Scene, False, False)])[0]
        # save all the necessary information to file
        self.writeImgToFile(image_response)
        self.writePosToFile(r, theta, psi, phi_rel)
        self.curr_idx += 1
        #TODO: just for debug
        image_response = self.client.simGetImages([airsim.ImageRequest('0', airsim.ImageType.Scene, False, False)])[0]
        img_1d = np.fromstring(image_response.image_data_uint8, dtype=np.uint8)  # get numpy array
        img_rgb = img_1d.reshape(image_response.height, image_response.width, 3)  # reshape array to 4 channel image array H X W X 3
        # img_bgr = racing_utils.dataset_utils.convert_rgb2bgr(img_rgb)
        img_resized = cv2.resize(img_rgb, (64, 64)).astype(np.float32)
        img_batch_1 = np.array([img_resized])
        # img_batch_1 = np.reshape(img_resized, (-1,) + img_resized.shape)
        gate_pose = self.gate_regressor.predict_gate_pose(img_batch_1, p_o_b)
        self.client.plot_tf([gate_pose], duration=20.0)


    # def update_debug(self):
    #     '''
    #     convetion of names:
    #     p_a_b: pose of frame b relative to frame a
    #     t_a_b: translation vector from a to b
    #     q_a_b: rotation quaternion from a to b
    #     o: origin
    #     b: UAV body frame
    #     g: gate frame
    #     '''
    #     # create and set pose for the quad
    #     # p_o_b, phi_base = racing_utils.geom_utils.randomQuadPose(UAV_X_RANGE, UAV_Y_RANGE, UAV_Z_RANGE, UAV_YAW_RANGE, UAV_PITCH_RANGE, UAV_ROLL_RANGE)
    #     p_o_b = airsim.Pose(Vector3r(0,0,-10), Quaternionr())
    #     self.client.simSetVehiclePose(p_o_b, True)
    #     # create and set gate pose relative to the quad
    #     alpha = CAM_FOV / 180.0 * np.pi / 2.0  # alpha is half of fov angle
    #     theta_range = [-alpha, alpha]
    #     r = 5.0
    #     # psi = np.pi/4 + np.pi/6
    #     # theta = -np.pi/9
    #     # for r in np.linspace(1,100, num=100):
    #     #     p_o_g, r, theta, psi, phi_rel = racing_utils.geom_utils.debugGatePoses(p_o_b, r, theta, psi)
    #     #     # self.client.simSetObjectPose(self.tgt_name, p_o_g, True)
    #     #     self.client.plot_tf([p_o_g], duration=20.0)
    #     #     # request quad img from AirSim
    #     #     image = self.client.simGetImages([airsim.ImageRequest('0', airsim.ImageType.Scene, False, False)])[0]
    #     #     # save all the necessary information to file
    #     #     self.writeImgToFile(image)
    #     #     self.writePosToFile(r, theta, psi, phi_rel)
    #     #     self.curr_idx += 1
    #     for theta in np.linspace(theta_range[0], theta_range[1], num=11):
    #         alpha_prime = np.arctan(np.cos(np.abs(theta)))
    #         psi_range = [np.pi / 2 - alpha_prime, np.pi / 2 + alpha_prime]
    #         for psi in np.linspace(psi_range[0], psi_range[1], num=11):
    #             p_o_g, r, theta, psi, phi_rel = racing_utils.geom_utils.debugGatePoses(p_o_b, r, theta, psi)
    #             # self.client.simSetObjectPose(self.tgt_name, p_o_g, True)
    #             self.client.plot_tf([p_o_g], duration=20.0)
    #             # request quad img from AirSim
    #             image = self.client.simGetImages([airsim.ImageRequest('0', airsim.ImageType.Scene, False, False)])[0]
    #             # save all the necessary information to file
    #             self.writeImgToFile(image)
    #             self.writePosToFile(r, theta, psi, phi_rel)
    #             self.curr_idx += 1

    def configureEnvironment(self):
        [self.client.simDestroyObject(gate_object) for gate_object in self.client.simListSceneObjects(".*[Gg]ate.*")]
        if self.with_gate:
            self.tgt_name = self.client.simSpawnObject("gate", "RedGate16x16", Pose(position_val=Vector3r(0,0,15)), 1.5)
            # self.tgt_name = self.client.simSpawnObject("gate", "CheckeredGate16x16", Pose(position_val=Vector3r(0,0,15)))
        else:
            self.tgt_name = "empty_target"

        if os.path.exists(self.csv_path):
            self.file = open(self.csv_path, "a")
        else:
            self.file = open(self.csv_path, "w")

    # write image to file
    def writeImgToFile(self, image_response):
        if len(image_response.image_data_uint8) == image_response.width * image_response.height * 3:
            img1d = np.fromstring(image_response.image_data_uint8, dtype=np.uint8)  # get numpy array
            img_rgb = img1d.reshape(image_response.height, image_response.width, 3)  # reshape array to 4 channel image array H X W X 3
            cv2.imwrite(os.path.join(self.base_path, 'images', str(self.curr_idx).zfill(len(str(self.num_samples))) + '.png'), img_rgb)  # write to png
        else:
            print('ERROR IN IMAGE SIZE -- NOT SUPPOSED TO HAPPEN')

    # writes your data to file
    def writePosToFile(self, r, theta, psi, phi_rel):
        data_string = '{0} {1} {2} {3}\n'.format(r, theta, psi, phi_rel)
        self.file.write(data_string)