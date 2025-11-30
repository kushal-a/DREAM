# Copyright (c) 2020 NVIDIA Corporation. All rights reserved.
# This work is licensed under the NVIDIA Source Code License - Non-commercial. Full
# text can be found in LICENSE.md

from enum import IntEnum

import albumentations as albu
import numpy as np
from PIL import Image as PILImage
import torch
from torch.utils.data import Dataset as TorchDataset
import torchvision.transforms as TVTransforms
import os

import dream

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Debug mode:
# 0: no debug mode
# 1: light debug
# 2: heavy debug
class ManipulatorNDDSDatasetDebugLevels(IntEnum):
    # No debug information
    NONE = 0
    # Minor debug information, passing of extra info but not saving to disk
    LIGHT = 1
    # Heavy debug information, including saving data to disk
    HEAVY = 2
    # Interactive debug mode, not intended to be used for actual training
    INTERACTIVE = 3


class ManipulatorNDDSDataset(TorchDataset):
    def __init__(
        self,
        ndds_dataset,
        network,
        config,
        augment_data=False,
        include_ground_truth=True,
        include_belief_maps=False,
        debug_mode=ManipulatorNDDSDatasetDebugLevels["NONE"],
    ):
        """
            Arguments:
                ndds_dataset = data, config
                    where `data` is a list of { 'name': '000001',
                                                'data_path': '.../000001.json',
                                                'image0_paths': { 'rgb': '.../000001.rgb.jpg' }}
                    and `config` is TODO
                keypoint_names = 
        """
        # Read in the camera intrinsics
        self.ndds_dataset_data = ndds_dataset[0]
        self.ndds_dataset_config = ndds_dataset[1]
        self.manipulator_name = config["manipulator"]["name"]
        self.keypoint_names = network.keypoint_names
        self.image_raw_resolution = config["training"]["config"]["image_raw_resolution"]
        (
            self.network_input_resolution,
            self.network_output_resolution
        ) = network.net_resolutions_from_image_raw_resolution(self.image_raw_resolution)
        self.augment_data = augment_data

        image_normalization = network.image_normalization
        image_preprocessing = network.image_preprocessing()
        self.position_normalization = network.position_normalization
        # If include_belief_maps is specified, include_ground_truth must also be
        # TBD: revisit better way of passing inputs, maybe to make one argument instead of two
        if include_belief_maps:
            assert (
                include_ground_truth
            ), 'If "include_belief_maps" is True, "include_ground_truth" must also be True.'
        self.include_ground_truth = include_ground_truth
        self.include_belief_maps = include_belief_maps

        self.dataset_name = network.network_config["data_path"].split('/')[-1]

        self.debug_mode = debug_mode

        assert (
            isinstance(image_normalization, dict) or not image_normalization
        ), 'Expected image_normalization to be either a dict specifying "mean" and "stdev", or None or False to specify no normalization.'

        # Image normalization
        # Basic PIL -> tensor without normalization, used for visualizing the net input image
        self.tensor_from_image_no_norm_tform = TVTransforms.Compose(
            [TVTransforms.ToTensor()]
        )

        if image_normalization:
            assert (
                "mean" in image_normalization and len(image_normalization["mean"]) == 3
            ), 'When image_normalization is a dict, expected key "mean" specifying a 3-tuple to exist, but it does not.'
            assert (
                "stdev" in image_normalization
                and len(image_normalization["stdev"]) == 3
            ), 'When image_normalization is a dict, expected key "stdev" specifying a 3-tuple to exist, but it does not.'

            self.tensor_from_image_tform = TVTransforms.Compose(
                [
                    TVTransforms.ToTensor(),
                    TVTransforms.Normalize(
                        image_normalization["mean"], image_normalization["stdev"]
                    ),
                ]
            )
        else:
            # Use the PIL -> tensor tform without normalization if image_normalization isn't specified
            self.tensor_from_image_tform = self.tensor_from_image_no_norm_tform

        assert (
            image_preprocessing in dream.image_proc.KNOWN_IMAGE_PREPROC_TYPES
        ), 'Image preprocessing type "{}" is not recognized.'.format(
            image_preprocessing
        )
        self.image_preprocessing = image_preprocessing
        # positions = torch.zeros((len(self),7,3))
        # for i in range(len(self)):
        #     positions[i] = self[i]["keypoint_positions"]
        # positions = positions.view(-1,3)
        # self.mean = torch.tensor(positions.mean(dim=0))
        # self.stdev = torch.tensor(positions.std(dim=0))
        # self.max = torch.tensor((positions - self.mean).abs().max(dim=0).values)
        # print("Dataset statistics:")
        # print("  Mean:\n{}".format(self.mean))
        # print("  Stdev:\n{}".format(self.stdev))
        # print("  Max (wrt mean):\n{}".format(self.max))
        # exit()

    def __len__(self):
        return len(self.ndds_dataset_data)
    
    def __getitem__(self, index):

        # Parse this datum
        datum = self.ndds_dataset_data[index]
        image_rgb_path = datum["image_paths"]["rgb"]

        # Extract keypoints from the json file
        data_path = datum["data_path"]

        optical_flow_path = self.convert_path(data_path)

        if self.include_ground_truth:
            keypoints = dream.utilities.load_keypoints(
                data_path, self.manipulator_name, self.keypoint_names
            )
        else:
            # Generate an empty 'keypoints' dict
            keypoints = dream.utilities.load_keypoints(
                data_path, self.manipulator_name, []
            )

        # Load image and transform to network input resolution -- pre augmentation
        image_rgb_raw = PILImage.open(image_rgb_path).convert("RGB")
        image_raw_resolution = image_rgb_raw.size

        # Do image preprocessing, including keypoint conversion
        image_rgb_before_aug = dream.image_proc.preprocess_image(
            image_rgb_raw, self.network_input_resolution, self.image_preprocessing
        )
        kp_projs_before_aug = dream.image_proc.convert_keypoints_to_netin_from_raw(
            keypoints["projections"],
            self.image_raw_resolution,
            self.network_input_resolution,
            self.image_preprocessing,
        )

        with np.load(optical_flow_path) as data:
            flow3 = np.array(data["optical_flow"], dtype=np.float16)

        flow3 -= flow3.mean()
        flow3 /= flow3.abs().max() + 1e-8

        flow_u8 = (flow3 * 255).astype(np.uint8)
        optical_img = PILImage.fromarray(flow_u8, mode="RGB")
        optical_img_before_aug = dream.image_proc.preprocess_image(
            optical_img, self.network_input_resolution, self.image_preprocessing
        )

        # Handle data augmentation
        if self.augment_data:
            augmentation = albu.Compose(
                [
                    albu.GaussNoise(),
                    albu.RandomBrightnessContrast(brightness_by_max=False),
                    albu.ShiftScaleRotate(rotate_limit=15),
                ],
                p=1.0,
                keypoint_params={"format": "xy", "remove_invisible": False},
            )
            data_to_aug = {
                "image": np.array(image_rgb_before_aug),
                "keypoints": kp_projs_before_aug,
            }
            augmented_data = augmentation(**data_to_aug)
            image_rgb_net_input = PILImage.fromarray(augmented_data["image"])
            kp_projs_net_input = augmented_data["keypoints"]
        else:
            image_rgb_net_input = image_rgb_before_aug
            optical_flow_net_input = optical_img_before_aug
            kp_projs_net_input = kp_projs_before_aug

        assert (
            image_rgb_net_input.size == self.network_input_resolution
        ), "Expected resolution for image_rgb_net_input to be equal to specified network input resolution, but they are different."

        # Now convert keypoints at network input to network output for use as the trained label
        kp_projs_net_output = dream.image_proc.convert_keypoints_to_netout_from_netin(
            kp_projs_net_input,
            self.network_input_resolution,
            self.network_output_resolution,
        )

        # Convert to tensor for output handling
        # This one goes through image normalization (used for inference)
        image_rgb_net_input_as_tensor = self.tensor_from_image_tform(
            image_rgb_net_input
        )
        optical_flow_net_input_as_tensor = torch.tensor(optical_flow_net_input)

        # This one is not (used for net input overlay visualizations - hence "viz")
        image_rgb_net_input_viz_as_tensor = self.tensor_from_image_no_norm_tform(
            image_rgb_net_input
        )

        # Convert keypoint data to tensors - use float32 size
        keypoint_positions_wrt_cam_as_tensor = torch.from_numpy(
            np.array(keypoints["positions_wrt_cam"])
        ).float()
        kp_projs_net_output_as_tensor = torch.from_numpy(
            np.array(kp_projs_net_output)
        ).float()

        keypoint_positions_wrt_cam_as_tensor -= torch.tensor(self.position_normalization["mean"])
        keypoint_positions_wrt_cam_as_tensor /= torch.tensor(self.position_normalization["max"])

        # Construct output sample
        sample = {
            "image_rgb_input": image_rgb_net_input_as_tensor,
            "keypoint_projections_output": kp_projs_net_output_as_tensor,
            "keypoint_positions": keypoint_positions_wrt_cam_as_tensor,
            "config": datum,
            "optical_flow": optical_flow_net_input_as_tensor[:,:,:2],
        }

        if self.debug_mode >= ManipulatorNDDSDatasetDebugLevels["LIGHT"]:
            kp_projections_as_tensor = torch.from_numpy(
                np.array(keypoints["projections"])
            ).float()
            sample["keypoint_projections_raw"] = kp_projections_as_tensor
            kp_projections_input_as_tensor = torch.from_numpy(
                kp_projs_net_input
            ).float()
            sample["keypoint_projections_input"] = kp_projections_input_as_tensor
            image_raw_resolution_as_tensor = torch.tensor(image_raw_resolution).float()
            sample["image_resolution_raw"] = image_raw_resolution_as_tensor
            sample["image_rgb_input_viz"] = image_rgb_net_input_viz_as_tensor

        # TODO: same as LIGHT debug, but also saves to disk
        if self.debug_mode >= ManipulatorNDDSDatasetDebugLevels["HEAVY"]:
            pass

        # Display to screen
        if self.debug_mode >= ManipulatorNDDSDatasetDebugLevels["INTERACTIVE"]:
            # Ensure that the points are consistent with the image transformations
            # The overlaid points on both image should be consistent, despite image transformations
            debug_image_raw = dream.image_proc.overlay_points_on_image(
                image_rgb_raw, keypoints["projections"], self.keypoint_names
            )
            debug_image_raw.show()

            debug_image = dream.image_proc.overlay_points_on_image(
                image_rgb_net_input, kp_projs_net_input, self.keypoint_names
            )
            debug_image.show()

            # Also show that the output resolution data are consistent
            image_rgb_net_output = image_rgb_net_input.resize(
                self.network_output_resolution, resample=PILImage.BILINEAR
            )
            debug_image_rgb_net_output = dream.image_proc.overlay_points_on_image(
                image_rgb_net_output, kp_projs_net_output, self.keypoint_names
            )
            debug_image_rgb_net_output.show()

            input("Press Enter to continue...")

        return sample
    
    @staticmethod
    def convert_path(self, path: str) -> str:
        # Split path into components
        parts = path.split(os.sep)
        
        # Replace the exact folder name "data"
        for i, p in enumerate(parts):
            if p == "data":
                parts[i] = "data_flow"
                break   # only the first exact match at that segment
        
        # Rebuild the path
        new_path = os.sep.join(parts)
        
        # Replace .rgb.jpg with .npz at the end
        if new_path.endswith(".rgb.jpg"):
            new_path = new_path[:-len(".rgb.jpg")] + ".npz"
        else:
            raise ValueError("Expected file to end with .rgb.jpg")
            
        return new_path


if __name__ == "__main__":
    from PIL import Image

    # beliefs = CreateBeliefMap((100,100),[(50,50),(-1,-1),(0,50),(50,0),(10,10)])
    # for i,b in enumerate(beliefs):
    #     print(b.shape)
    #     stack = np.stack([b,b,b],axis=0).transpose(2,1,0)
    #     im = Image.fromarray((stack*255).astype('uint8'))
    #     im.save('{}.png'.format(i))

    path = "/home/sbirchfield/data/FrankaSimpleHomeDR20k/"
    # path = '/home/sbirchfield/data/FrankaSimpleMPGammaDR105k/'

    keypoint_names = [
        "panda_link0",
        "panda_link2",
        "panda_link3",
        "panda_link4",
        "panda_link6",
        "panda_link7",
        "panda_hand",
    ]

    found_data = dream.utilities.find_ndds_data_in_dir(path)
    train_dataset = ManipulatorNDDSDataset(
        found_data,
        "panda",
        keypoint_names,
        (400, 400),
        (100, 100),
        include_belief_maps=True,
        augment_data=True,
    )
    trainingdata = torch.utils.data.DataLoader(
        train_dataset, batch_size=32, shuffle=False, num_workers=1, pin_memory=True
    )

    targets = iter(trainingdata).next()

    for i, b in enumerate(targets["belief_maps"][0]):
        print(b.shape)
        stack = np.stack([b, b, b], axis=0).transpose(2, 1, 0)
        im = Image.fromarray((stack * 255).astype("uint8"))
        im.save("{}.png".format(i))
