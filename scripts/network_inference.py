# Copyright (c) 2020 NVIDIA Corporation. All rights reserved.
# This work is licensed under the NVIDIA Source Code License - Non-commercial. Full
# text can be found in LICENSE.md

import argparse
import math
import os
from PIL import Image as PILImage

import numpy as np
from ruamel.yaml import YAML
import torch
import torchvision.transforms as TVTransforms

import dream

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def network_inference(args):

    if args.input_config_path:
        input_config_path = args.input_config_path
    else:
        input_config_path = os.path.splitext(args.input_params_path)[0] + ".yaml"

    keypoints_path = os.path.splitext(args.image_path)[0]
    keypoints_path = os.path.splitext(keypoints_path)[0]+ ".json"

    # Create parser
    print("# Opening config file:  {} ...".format(input_config_path))
    data_parser = YAML(typ="safe")
    with open(input_config_path, "r") as f:
        network_config = data_parser.load(f)

    # Overwrite GPU
    # If nothing is specified at the command line, None is the default, which uses all GPUs
    # TBD - think about a better way of doing this
    network_config["training"]["platform"]["gpu_ids"] = args.gpu_ids

    # Load network
    print("# Creating network...")
    dream_network = dream.create_network_from_config_data(network_config)

    print("Loading network with weights from:  {} ...".format(args.input_params_path))
    dream_network.model.load_state_dict(torch.load(args.input_params_path))
    dream_network.enable_evaluation()

    # Load in image
    print("# Loading image:  {} ...".format(args.image_path))
    image_rgb_OrigInput_asPilImage = PILImage.open(args.image_path).convert("RGB")

    # Use image preprocessing specified by config by default, unless user specifies otherwise
    if args.image_preproc_override:
        image_preprocessing = args.image_preproc_override
        print(
            "  Image preprocessing: '{}' --- as specified by user".format(
                image_preprocessing
            )
        )
    else:
        image_preprocessing = dream_network.image_preprocessing()
        print(
            "  Image preprocessing: '{}' --- default as specified by network config".format(
                image_preprocessing
            )
        )

    # Read in gt keypoints
    print(
        "# Loading ground truth keypoints from {} ...".format(keypoints_path)
    )
    
    # Grandparent directory of the image file
    input_data_path = os.path.dirname(os.path.abspath(args.image_path))
    found_data = dream.utilities.find_ndds_data_in_dir(input_data_path)
    enable_augment_data = False if not network_config['training']['config']['data_augmentation'] else True
    found_dataset = dream.datasets.ManipulatorNDDSDataset(
        found_data,
        dream_network,
        network_config,
        augment_data=enable_augment_data,
        include_ground_truth=True,
    )

    print("Detecting keypoints...")
    detection_result = dream_network.keypoints_from_image(
        image_rgb_OrigInput_asPilImage,
        image_preprocessing_override=image_preprocessing,
        debug=True,
        optical_flow = found_dataset.get_optical_flow_data(args.image_path)
    )

    positions = torch.tensor(detection_result["positions"]).unsqueeze(0)

    image_rgb_NetInput_asPilImage = detection_result["image_rgb_net_input"]


    img = found_dataset.tensor_from_image_no_norm_tform(
            image_rgb_NetInput_asPilImage
        ).unsqueeze(0)
    
    keypoints_gt = dream.utilities.load_keypoints(
        keypoints_path,
        dream_network.manipulator_name,
        dream_network.keypoint_names,
    )
    keypoints_gt = dream.image_proc.convert_keypoints_to_netin_from_raw(
        keypoints_gt["projections"],
        found_dataset.image_raw_resolution,
        found_dataset.network_input_resolution,
        found_dataset.image_preprocessing,
    )

    keypoints_gt = torch.tensor(keypoints_gt, dtype=torch.float32).unsqueeze(0)

    keypoints_overlay = dream.analysis.plot_pos_on_image(img,
                                            positions,
                                            keypoints_gt,
                                            found_dataset,
                                            dream_network,
                                            cols=1
                                        )
    keypoints_overlay.show(
        title="Keypoints (possibly with ground truth) on net input image"
    )

    print("Done.")


if __name__ == "__main__":

    print(
        "---------- Running 'network_inference.py' -------------------------------------------------"
    )

    # Parse input arguments
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "-i",
        "--input-params-path",
        required=True,
        help="Path to network parameters file (.pth).",
    )
    parser.add_argument(
        "-c",
        "--input-config-path",
        default=None,
        help="Path to network configuration file. If nothing is specified, the script will search for a config file by the same name as the network parameters file.",
    )
    parser.add_argument(
        "-m", "--image_path", required=True, help="Path to image used for inference."
    )
    parser.add_argument(
        "-g",
        "--gpu-ids",
        nargs="+",
        type=int,
        default=None,
        help="The GPU IDs on which to conduct network inference. Nothing specified means all GPUs will be utilized. Does not affect results, only how quickly the results are obtained.",
    )
    parser.add_argument(
        "-p",
        "--image-preproc-override",
        default=None,
        help="Overrides the image preprocessing specified by the network. (Debug argument.)",
    )
    args = parser.parse_args()

    # Run network inference
    network_inference(args)
