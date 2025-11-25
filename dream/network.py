# Copyright (c) 2020 NVIDIA Corporation. All rights reserved.
# This work is licensed under the NVIDIA Source Code License - Non-commercial. Full
# text can be found in LICENSE.md

import os
from PIL import Image as PILImage

import numpy as np
import ruamel.yaml
import yaml
import torch
import torchvision.transforms as TVTransforms

import dream
import posediff

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

KNOWN_OPTIMIZERS = [
    "adam", 
    "sgd",
]  # the Stochastic Gradient Descent optimizer

def create_network_from_config_data(network_config_data):
    # Create the network
    dream_network = DreamNetwork(network_config_data)
    return dream_network
class DreamNetwork:
    def __init__(self, network_config):

        self.network_config = network_config

        self.parse_keypoints()

        # TBD: the following "getters" should all be methods
        self.manipulator_name = self.network_config["manipulator"]["name"]
        self.n_keypoints = len(self.keypoint_names)
        self.architecture_type = self.network_config["architecture"]["type"]

        # print robot info
        print("---------- `network.py`.  `DreamNetwork:__init()` ----------")
        print("  Manipulator: {}".format(self.manipulator_name))
        print("  Keypoint names: {}".format(self.keypoint_names))
        print("  Friendly keypoint names: {}".format(self.friendly_keypoint_names))
        print("  Architecture type: {}".format(self.architecture_type))

        self.image_normalization = self.network_config["architecture"]["image_normalization"]

        # Create architecture and loss
        with open(network_config["posediff_config"], 'r') as file:
            config = yaml.safe_load(file)
        self.model = posediff.PoseDiffModel(config)
        self.criterion = torch.nn.MSELoss()

        # Optimizer is created in a separate call, because this isn't needed unless we're training
        self.optimizer = None

    def trained_net_input_resolution(self):
        return tuple(self.network_config["training"]["config"]["net_input_resolution"])

    def trained_net_output_resolution(self):
        return tuple(self.network_config["training"]["config"]["net_output_resolution"])

    def image_preprocessing(self):
        return self.network_config["architecture"]["image_preprocessing"]
    
    def parse_keypoints(self):
        # Parse keypoint specification
        self.keypoint_names = []
        self.friendly_keypoint_names = []
        self.ros_keypoint_frames = []

        for kp_def in self.network_config["manipulator"]["keypoints"]:
            kp_name = kp_def["name"]

            friendly_kp_name = kp_def["friendly_name"] if "friendly_name" in kp_def else kp_name
            ros_kp_frame = kp_def["ros_frame"] if "ros_frame" in kp_def else kp_name
            
            self.keypoint_names.append(kp_name)
            self.friendly_keypoint_names.append(friendly_kp_name)
            self.ros_keypoint_frames.append(ros_kp_frame)

    def train(self, network_input_heads, target):
        assert self.optimizer, "Optimizer must be defined. Use enable_training() first."

        self.optimizer.zero_grad()

        loss = self.loss(network_input_heads, target)

        loss.backward()
        self.optimizer.step()

        return loss

    def loss(self, network_input_heads, target):
        target = target.reshape(network_input_heads.shape[0], -1)
        noise = torch.randn_like(target)
        pred = self.model(network_input_heads, target, noise)
        # Verify TODO
        if self.model.training:
            loss = self.criterion(pred, noise)
        else:
            loss = self.criterion(pred, target)

        return loss

    # image_raw_resolution: (width, height) in pixels
    # calls resolution_after_preprocessing under the hood, using network trained resolution as the reference resolution
    def net_resolutions_from_image_raw_resolution(
        self, image_raw_resolution, image_preprocessing_override=None
    ):

        # Input argument handling
        assert (
            len(image_raw_resolution) == 2
        ), 'Expected "image_raw_resolution" to have length 2, but it has length {}.'.format(
            len(image_raw_resolution)
        )

        image_preprocessing = (
            image_preprocessing_override
            if image_preprocessing_override
            else self.image_preprocessing()
        )

        # Calculate image resolution at network input layer, after preprocessing
        net_input_resolution = dream.image_proc.resolution_after_preprocessing(
            image_raw_resolution,
            self.trained_net_input_resolution(),
            image_preprocessing,
        )
        net_output_resolution = self.net_output_resolution_from_input_resolution(
            net_input_resolution
        )

        return net_input_resolution, net_output_resolution

    def net_output_resolution_from_input_resolution(self, net_input_resolution):

        # Input argument handling
        assert (
            len(net_input_resolution) == 2
        ), 'Expected "net_input_resolution" to have length 2, but it has length {}.'.format(
            len(net_input_resolution)
        )
        # Set default value
        net_output_resolution = net_input_resolution

        ##### For PoseDiff, model output is not image. Net in and out res are always the same #####

        # netin_width, netin_height = net_input_resolution

        # # Construct test input and send thru network to get the output
        # # This assumes there is only one input head, which is the RGB image
        # with torch.no_grad():
        #     net_input_as_tensor_batch = torch.zeros(
        #         2, 3, netin_height, netin_width
        #     ).cuda()
        #     target = torch.zeros(2, self.n_keypoints*3).cuda()
        #     noise = torch.zeros_like(target).cuda()
        #     net_output_as_tensor_batch = self.model(net_input_as_tensor_batch, target, noise)
        #     net_output_shape = net_output_as_tensor_batch[0][0].shape
        #     net_output_resolution = (net_output_shape[2], net_output_shape[1])

        return net_output_resolution

    # Wrapper for inferences from a PIL image directly
    # Returns keypoints in the input image (not necessarily network input) frame
    # Allows for an optional overwrite
    def keypoints_from_image(
        self, input_rgb_image_as_pil, image_preprocessing_override=None, debug=False
    ):
        # do preprocessing
        image_preprocessing = (
            image_preprocessing_override
            if image_preprocessing_override
            else self.image_preprocessing()
        )

        input_image_preproc_before_norm = dream.image_proc.preprocess_image(
            input_rgb_image_as_pil,
            self.trained_net_input_resolution(),
            image_preprocessing,
        )

        tensor_from_image_tform = TVTransforms.Compose(
            [
                TVTransforms.ToTensor(),
                TVTransforms.Normalize(
                    self.image_normalization["mean"], self.image_normalization["stdev"]
                ),
            ]
        )
        input_rgb_image_as_tensor = tensor_from_image_tform(
            input_image_preproc_before_norm
        )

        # Inference
        with torch.no_grad():
            input_rgb_image_as_tensor_batch = input_rgb_image_as_tensor.unsqueeze(
                0
            ).cuda()
            positions_batch = self.inference(
                input_rgb_image_as_tensor_batch
            ).cpu()

        positions = np.array(
            positions_batch[0], dtype=float
        )

        detection_result = {"positions": positions}
        if debug:
            detection_result["image_rgb_net_input"] = input_image_preproc_before_norm

        return detection_result

    # Inference is designed to return the best output of belief_maps and keypoints
    # This is an abstraction layer so even if multiple stages are used, this only produces one set of outputs (for the final stage)
    def inference(self, network_input):
        network_output = self.model(network_input)
        network_output = network_output.reshape(
            network_output.shape[0], -1, 3
        )
        return network_output

    def save_network_config(self, config_file_path, overwrite=False):

        if not overwrite:
            assert not os.path.exists(
                config_file_path
            ), 'Output file already exists in "{}".'.format(config_file_path)

        # Create saver
        data_saver = ruamel.yaml.YAML()
        data_saver.default_flow_style = False
        data_saver.explicit_start = False

        with open(config_file_path, "w") as f:
            # TBD - convert to ruamel.yaml.comments.CommentedMap to get rid of !!omap in yaml
            data_saver.dump(self.network_config, f)

    def save_network_params(self, network_params_path, overwrite=False):

        if not overwrite:
            assert not os.path.exists(
                network_params_path
            ), 'Output file already exists in "{}".'.format(network_params_path)

        # Save weights
        torch.save(self.model.state_dict(), network_params_path)

    def save_network(
        self, output_dir, output_filename_without_extension, overwrite=False
    ):

        dream.utilities.makedirs(output_dir, exist_ok=overwrite)

        network_config_dir = os.path.join(
            output_dir, output_filename_without_extension + ".yaml"
        )
        self.save_network_config(network_config_dir, overwrite)

        network_params_path = os.path.join(
            output_dir, output_filename_without_extension + ".pth"
        )
        self.save_network_params(network_params_path, overwrite)

    def enable_training(self):

        # Load optimizer if needed
        if not self.optimizer:

            assert (
                "optimizer" in self.network_config["training"]["config"]
            ), 'Required key "optimizer" in dictionary "config" is missing from network configuration.'
            assert (
                "type" in self.network_config["training"]["config"]["optimizer"]
            ), 'Required key "type" in dictionary "optimizer" is missing from network configuration.'

            network_parameters = filter(
                lambda p: p.requires_grad, self.model.parameters()
            )
            optimizer_type = self.network_config["training"]["config"]["optimizer"][
                "type"
            ]

            assert (
                optimizer_type in KNOWN_OPTIMIZERS
            ), 'Expected optimizer_type "{}" to be in the list of known optimizers, but it is not.'.format(
                optimizer_type
            )

            if optimizer_type == "adam":

                assert (
                    "learning_rate"
                    in self.network_config["training"]["config"]["optimizer"]
                ), 'Required key "learning_rate" in dictionary "optimizer" is missing to use the Adam optimizer.'

                self.optimizer = torch.optim.Adam(
                    network_parameters,
                    lr=self.network_config["training"]["config"]["optimizer"][
                        "learning_rate"
                    ],
                )

            elif optimizer_type == "sgd":

                assert (
                    "learning_rate"
                    in self.network_config["training"]["config"]["optimizer"]
                ), 'Required key "learning_rate" in dictionary "optimizer" is missing to use the SGD optimizer.'

                self.optimizer = torch.optim.SGD(
                    network_parameters,
                    lr=self.network_config["training"]["config"]["optimizer"][
                        "learning_rate"
                    ],
                )

            else:
                assert False, 'Optimizer "{}" is not defined.'.format(optimizer_type)

        # Enable training mode
        self.model.train()

    def enable_evaluation(self):

        # Enable evaluation mode
        self.model.eval()