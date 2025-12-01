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
    state_dict = torch.load(args.input_params_path, weights_only=False)
    dream_network.model.load_state_dict(state_dict)
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

    print(f"Detecting keypoints {args.num_predictions} times...")
    all_positions = []

    detection_result = dream_network.keypoints_from_image(
        image_rgb_OrigInput_asPilImage,
        image_preprocessing_override=image_preprocessing,
        n=args.num_predictions
    )

    all_positions = torch.tensor(detection_result["positions"])

    # image_rgb_NetInput_asPilImage = detection_result["image_rgb_net_input"]


    joint_names = [
        'panda_link0',
        'panda_link2',
        'panda_link3',
        'panda_link4',
        'panda_link6',
        'panda_link7',
        'panda_hand'
    ]
    axis_names = ['x', 'y', 'z']

    N, J, C = all_positions.shape

    import matplotlib.pyplot as plt

    model_name = args.input_params_path.split("/")[-2]

    os.makedirs(f'plots/{model_name}', exist_ok=True)

    for j in range(J):
        fig, axes = plt.subplots(1, 3, figsize=(12, 3))
        for c in range(C):
            axes[c].hist(all_positions[:, j, c].cpu().numpy(), bins=30)
            axes[c].set_title(f"{joint_names[j]} - {axis_names[c]}")

        plt.tight_layout()

        fig.savefig(f'plots/{model_name}/{joint_names[j]}')

    stats = {
        "mean": all_positions.mean(dim=0),  # [7, 3]
        "std": all_positions.std(dim=0),  # [7, 3]
        "var": all_positions.var(dim=0),  # [7, 3]
        "min": all_positions.min(dim=0).values,
        "max": all_positions.max(dim=0).values,
    }

    import pandas as pd

    df = pd.DataFrame({
        "joint": joint_names,

        "mean_x": stats["mean"][:, 0].tolist(),
        "mean_y": stats["mean"][:, 1].tolist(),
        "mean_z": stats["mean"][:, 2].tolist(),

        "std_x": stats["std"][:, 0].tolist(),
        "std_y": stats["std"][:, 1].tolist(),
        "std_z": stats["std"][:, 2].tolist(),

        "min_x": stats["min"][:, 0].tolist(),
        "min_y": stats["min"][:, 1].tolist(),
        "min_z": stats["min"][:, 2].tolist(),

        "max_x": stats["max"][:, 0].tolist(),
        "max_y": stats["max"][:, 1].tolist(),
        "max_z": stats["max"][:, 2].tolist(),
    })

    df.to_csv(f"plots/{model_name}/keypoint_stats.csv", index=False)

    # Read in gt keypoints
    # print(
    #     "# Loading ground truth keypoints from {} ...".format(keypoints_path)
    # )
    #
    # # Grandparent directory of the image file
    # input_data_path = os.path.dirname(os.path.abspath(args.image_path))
    # found_data = dream.utilities.find_ndds_data_in_dir(input_data_path)
    # enable_augment_data = False if not network_config['training']['config']['data_augmentation'] else True
    # found_dataset = dream.datasets.ManipulatorNDDSDataset(
    #     found_data,
    #     dream_network,
    #     network_config,
    #     augment_data=enable_augment_data,
    #     include_ground_truth=True,
    # )
    #
    # img = found_dataset.tensor_from_image_no_norm_tform(
    #         image_rgb_NetInput_asPilImage
    #     ).unsqueeze(0)
    #
    # keypoints_gt = dream.utilities.load_keypoints(
    #     keypoints_path,
    #     dream_network.manipulator_name,
    #     dream_network.keypoint_names,
    # )
    # keypoints_gt = dream.image_proc.convert_keypoints_to_netin_from_raw(
    #     keypoints_gt["projections"],
    #     found_dataset.image_raw_resolution,
    #     found_dataset.network_input_resolution,
    #     found_dataset.image_preprocessing,
    # )
    #
    # keypoints_gt = torch.tensor(keypoints_gt, dtype=torch.float32).unsqueeze(0)
    #
    # keypoints_overlay = dream.analysis.plot_pos_on_image(img,
    #                                         positions,
    #                                         keypoints_gt,
    #                                         found_dataset,
    #                                         dream_network,
    #                                         cols=1
    #                                     )
    # keypoints_overlay.show(
    #     title="Keypoints (possibly with ground truth) on net input image"
    # )
    #
    # print("Done.")


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
        "-m", "--image-path", required=True, help="Path to image used for inference."
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
    parser.add_argument(
        '-n',
        '--num-predictions',
        type=int,
        default=4
    )
    args = parser.parse_args()

    # Run network inference
    network_inference(args)
