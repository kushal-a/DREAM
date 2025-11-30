# Copyright (c) 2020 NVIDIA Corporation. All rights reserved.
# This work is licensed under the NVIDIA Source Code License - Non-commercial. Full
# text can be found in LICENSE.md

import argparse
from collections import OrderedDict as odict
import pickle
import os
import random
import socket
import time

import numpy as np
from ruamel.yaml import YAML
import torch
from torch.utils.data import DataLoader as TorchDataLoader
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import dream
from pprint import pprint
import metrics


def train_network(args):
    config_file = os.path.abspath(args.config_file)

    # Parse all configurations and parameters
    yaml_parser = YAML(typ="safe")
    with open(config_file, "r") as f:
        network_config = yaml_parser.load(f)

    # Extract Sub Configurations
    dataset_config      = network_config["data_path"]
    manipulator_config  = network_config["manipulator"]
    architecture_config = network_config["architecture"]
    training_config     = network_config["training"]
    posediff_config_path= network_config["posediff_config"]
    logging_config      = network_config["logging"]


    # Handling Dataset Config Extraction
    dataset_path        = dataset_config["dataset_path"]
    dataset_type        = dataset_config["dataset_type"]
    if dataset_type == "real":
        input_data_path = os.path.join(dataset_path,
                                 dataset_type,
                                 "panda-" + dataset_config["real"]["cam"]
                                 )

    elif dataset_type == "synth":
        synth_type = dataset_config["synth"]["type"]
        input_data_path = os.path.join(dataset_path,
                                 dataset_type,
                                 "panda_synth_" + synth_type
                                 )

    else:
        raise Exception("Typo in config file dataset<dataset_type")
    found_data = dream.utilities.find_ndds_data_in_dir(input_data_path)

    # Handling Training Config Extraction
    training_config_config          = training_config["config"]
    epochs                          = training_config_config["epochs"]
    train_split                     = training_config_config["training_data_fraction"]
    batch_size                      = training_config_config["batch_size"]
    num_workers                     = training_config_config["worker_size"]
    optimizer                       = training_config_config["optimizer"]["type"]
    lr                              = training_config_config["optimizer"]["learning_rate"]
    training_image_preprocessing    = training_config_config["image_preprocessing"]
    image_raw_resolution            = training_config_config["image_raw_resolution"]
    training_net_input_resolution   = training_config_config["net_input_resolution"]

    image_raw_resolution = dream.utilities.load_image_resolution(
        found_data[1]["camera"]
    )
    val_split = 1.0 - train_split

    # Handling Logging Config Extraction
    output_dir      = logging_config["output_dir"]
    save_results    = logging_config["save_results"]
    verbose         = logging_config["verbose"]
    dream.utilities.makedirs(output_dir, exist_ok=args.force_overwrite)

    # Handling Required Overwrites for DREAM compatibility
    ## Handling redundencies
    network_config["data_path"] = input_data_path
    architecture_config["image_preprocessing"] = training_image_preprocessing
    training_config_config["validation_data_fraction"] = val_split
    training_config_config["image_raw_resolution"] = image_raw_resolution

    enable_augment_data = not args.not_augment_data
    data_augment_config = odict([("image_rgb", True)]) if enable_augment_data else False
    training_config_config["data_augmentation"] = data_augment_config

    ## Handling dummys
    try:
        user = os.getlogin()
    except:
        user = "not found"
    gpu_ids = args.gpu_ids if args.gpu_ids else []
    platform_config = odict(
        [
            ("user", user),
            ("hostname", socket.gethostname()),
            ("gpu_ids", gpu_ids)
        ]
    )

    ## Reconstructing config
    training_config["config"] = training_config_config
    training_config["platform"] = platform_config
    network_config["training"] = training_config
    network_config["architecture"] = architecture_config

    print("Network configuration:")
    pprint(network_config)

    training_start_time = time.time()

    # Load weights, logs and config from output directory if resuming training
    if args.resume_training:

        # Find the latest network we have
        dirlist = os.listdir(output_dir)
        epoch_weight_paths_unsorted = [
            x for x in dirlist if x.startswith("epoch") and x.endswith(".pth")
        ]
        epoch_numbers_unsorted = []
        for net_path in epoch_weight_paths_unsorted:
            epoch_number = int(net_path.split("_")[1].split(".")[0])
            epoch_numbers_unsorted.append(epoch_number)

        temp = sorted(
            zip(epoch_weight_paths_unsorted, epoch_numbers_unsorted),
            key=lambda pair: pair[1],
            reverse=True,
        )
        epoch_weight_paths = [x[0] for x in temp]
        epoch_numbers = [x[1] for x in temp]

        # Most recent network
        most_recent_epoch_weight_path = epoch_weight_paths[0]
        start_epoch = epoch_numbers[0]

        assert (
            start_epoch < epochs
        ), "Network is already trained for the number of requested epochs."

        # Find the best network to determine its validation loss
        best_valid_network_config_path = os.path.join(
            output_dir, "best_network.yaml"
        )
        assert os.path.exists(
            best_valid_network_config_path
        ), "Could not determine the best validation loss."

        valid_parser = YAML(typ="safe")
        with open(best_valid_network_config_path, "r") as f:
            best_valid_network_config = valid_parser.load(f)
        # best_valid_loss = best_valid_network_config["training"]["results"][
        #     "validation_loss"
        # ]["mean"]

        best_valid_add = best_valid_network_config["training"]["results"]["validation_add"]["mean"]

        # Load in the old training log
        if os.path.exists(os.path.join(output_dir, "training_log.pkl")):
            train_log_path = os.path.join(output_dir, "training_log.pkl")
            with open(train_log_path, "rb") as f:
                train_log = pickle.load(f)
            # Move this to make this consistent as if we're in the middle of training
            os.rename(
                train_log_path,
                os.path.join(
                    output_dir, "training_log_e{}.pkl".format(start_epoch)
                ),
            )

        elif os.path.exists(
            os.path.join(output_dir, "training_log_e{}.pkl".format(start_epoch))
        ):
            train_log_path = os.path.join(
                output_dir, "training_log_e{}.pkl".format(start_epoch)
            )
            with open(train_log_path, "rb") as f:
                train_log = pickle.load(f)
        else:
            assert False, "Could not determine training log file to resume."

        # Get the random seed that was used here - we need to to ensure test/valid splits are right
        random_seed = train_log["random_seed"]

        # Set the random seed here because it's different
        if not isinstance(train_log["start_time"], list):
            # Convert to a list
            train_log["start_time"] = [train_log["start_time"]]

        train_log["start_time"].append(training_start_time)

        # Also log the fact that we resumed
        if "epochs_resumed" in train_log:
            train_log["epochs_resumed"].append(start_epoch + 1)
        else:
            train_log["epochs_resumed"] = [start_epoch + 1]

        # Load corresponding config file to ensure we're consistent
        most_recent_config_path = most_recent_epoch_weight_path.replace("pth", "yaml")
        config_parser = YAML(typ="safe")

        with open(os.path.join(output_dir, most_recent_config_path), "r") as f:
            most_recent_network_config_file = config_parser.load(f)

        # Use this one instead!
        network_config = most_recent_network_config_file

        print("~~ RESUMING TRAINING FROM {} ~~".format(most_recent_epoch_weight_path))
        print("")
    else:
        # Determine the random seed
        random_seed = (
            args.random_seed if args.random_seed else random.randint(0, 999999)
        )

        train_log = {
            "epochs": [],
            "losses": [],
            "validation_add": [],
            "validation_auc": [],
            "batch_training_losses": [],
            "batch_training_sample_names": [],
            "batch_validation_sample_names": [],
            "start_time": training_start_time,
            "timestamps": [],
            "random_seed": random_seed,
        }
        best_valid_add = float("Inf")
        start_epoch = 0

    dream.utilities.set_random_seed(random_seed)

    dream_network = dream.create_network_from_config_data(network_config)
    dream_network.enable_training()

    # Create NDDS dataset and loader
    training_debug_mode = dream.datasets.ManipulatorNDDSDatasetDebugLevels["LIGHT"]

    found_dataset = dream.datasets.ManipulatorNDDSDataset(
        found_data,
        dream_network,
        network_config,
        augment_data=enable_augment_data,
        include_ground_truth=True,
        debug_mode=training_debug_mode,
    )

    # Split into train and validation subsets
    n_data = len(found_dataset)
    n_train_data = int(round(n_data * train_split))
    n_valid_data = n_data - n_train_data
    train_dataset, valid_dataset = torch.utils.data.random_split(
        found_dataset, [n_train_data, n_valid_data]
    )

    train_data_loader = TorchDataLoader(
        train_dataset, batch_size=batch_size, num_workers=num_workers
    )

    valid_data_loader = TorchDataLoader(
        valid_dataset, batch_size=batch_size, num_workers=num_workers
    )

    # Create a timestamped subdirectory for TensorBoard logs
    timestamp = time.strftime("%Y%m%d-%H%M%S", time.localtime())
    tb_log_dir = os.path.join(output_dir, "runs_" + timestamp)
    os.makedirs(tb_log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=tb_log_dir)

    # Train the network
    print("")
    print("TRAINING NETWORK ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print("")

    last_epoch_timestamp = 0.0

    for e in tqdm(range(start_epoch, epochs)):
        this_epoch = e + 1
        print("")
        print("Epoch {} ------------".format(this_epoch))

        # Training Phase ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        if verbose:
            print("")
            print("~~ Training Phase ~~")

        dream_network.enable_training()

        training_batch_losses = []
        training_batch_sample_names = []

        for batch_idx, sample in enumerate(tqdm(train_data_loader)):
            this_batch_sample_names = sample["config"]["name"]
            this_batch_size = sample["image_rgb_input"].shape[0]

            if verbose:
                print("Processing batch index {} for training...".format(batch_idx))
                print(
                    "Sample names in this training batch: {}".format(
                        this_batch_sample_names
                    )
                )
                print("This training batch size: {}".format(this_batch_size))

            # New unified training
            network_input_heads = sample["image_rgb_input"].cuda()
            training_labels = sample["keypoint_positions"].cuda()

            loss = dream_network.train(network_input_heads, training_labels)

            training_loss_this_batch = loss.item()
            training_batch_losses.append(training_loss_this_batch)
            if verbose:
                print(
                    "Training loss for this batch: {}".format(training_loss_this_batch)
                )
                print("")
            training_batch_sample_names.append(this_batch_sample_names)
            writer.add_scalar('Loss/train', training_loss_this_batch, e * len(train_data_loader) + batch_idx)

        mean_training_loss_per_batch = np.mean(training_batch_losses)
        std_training_loss_per_batch = np.std(training_batch_losses)

        # Evaluation Phase ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        if verbose:
            print("")
            print("~~ Validation Phase ~~")

        dream_network.enable_evaluation()

        all_pred_keypoints = []
        all_gt_keypoints = []

        with torch.no_grad():

            valid_batch_sample_names = []

            for valid_batch_idx, valid_sample in enumerate(tqdm(valid_data_loader)):

                this_valid_batch_sample_names = valid_sample["config"]["name"]
                this_valid_batch_size = valid_sample["image_rgb_input"].shape[0]

                if verbose:
                    print("Processing batch index {} for validation...".format(valid_batch_idx))
                    print(
                        "Sample names in this validation batch: {}".format(
                            this_valid_batch_sample_names
                        )
                    )
                    print("This validation batch size: {}".format(this_valid_batch_size))

                # New unified validation
                valid_network_input_heads = valid_sample["image_rgb_input"].cuda()
                valid_labels = valid_sample["keypoint_positions"].cuda()

                predicted_keypoints = dream_network.inference(valid_network_input_heads)
                predicted_keypoints *= found_dataset.clamps[found_dataset.dataset_name]["max"]
                predicted_keypoints += found_dataset.clamps[found_dataset.dataset_name]["mean"]

                all_pred_keypoints.append(predicted_keypoints)
                all_gt_keypoints.append(valid_labels)
                valid_batch_sample_names.append(this_valid_batch_sample_names)

                batch_valid_add = metrics.compute_add(predicted_keypoints, valid_labels)['add']
                batch_valid_auc = metrics.compute_auc(predicted_keypoints, valid_labels)['auc']

                writer.add_scalar('Batch/valid_ADD', batch_valid_add, e * len(train_data_loader) + batch_idx)
                writer.add_scalar('Batch/valid_AUC', batch_valid_auc, e * len(train_data_loader) + batch_idx)

        all_pred_keypoints = torch.cat(all_pred_keypoints, dim=0)
        all_gt_keypoints = torch.cat(all_gt_keypoints, dim=0)

        valid_add = metrics.compute_add(all_pred_keypoints, all_gt_keypoints)['add']
        valid_auc = metrics.compute_auc(all_pred_keypoints, all_gt_keypoints)['auc']

        writer.add_scalar('Epoch/train_noise_loss', mean_training_loss_per_batch, e)
        writer.add_scalar('Epoch/valid_ADD', valid_add, e)
        writer.add_scalar('Epoch/valid_AUC', valid_auc, e)

        writer.flush()
        writer.close()

        # Bookkeeping and print info
        dream_network.network_config["training"]["results"]["epochs_trained"] += 1
        dream_network.network_config["training"]["results"]["training_loss"] = odict(
            [
                ("mean", float(mean_training_loss_per_batch)),
                ("stdev", float(std_training_loss_per_batch)),
            ]
        )
        dream_network.network_config["training"]["results"]["validation_add"] = odict(
            [
                ("mean", float(valid_add)),
            ]
        )
        dream_network.network_config["training"]["results"]["validation_auc"] = odict(
            [
                ("mean", float(valid_auc)),
            ]
        )
        print(
            "Training Loss (batch-wise mean +- 1 stdev): {} +- {}".format(
                mean_training_loss_per_batch, std_training_loss_per_batch
            )
        )
        print("Validation ADD (mm):  {:.5f}".format(valid_add))
        print("Validation AUC (%):   {:.5f}".format(valid_auc))
        print("=" * 70)

        if valid_add < best_valid_add:
            print("Best network result so far (ADD: {:.5f} mm)".format(valid_add))
            best_valid_add = valid_add

            if save_results:
                dream_network.save_network(
                    output_dir, "best_network", overwrite=True
                )

        this_epoch_timestamp = time.time() - training_start_time
        print(
            "This epoch took {} seconds.".format(
                this_epoch_timestamp - last_epoch_timestamp
            )
        )
        last_epoch_timestamp = this_epoch_timestamp
        print("")

        # Append to history
        train_log["epochs"].append(this_epoch)
        train_log["losses"].append(mean_training_loss_per_batch)
        # train_log["validation_losses"].append(mean_valid_loss_per_batch)
        train_log["validation_add"].append(valid_add)
        train_log["validation_auc"].append(valid_auc)
        train_log["batch_training_losses"].append(training_batch_losses)
        # train_log["batch_validation_losses"].append(valid_batch_losses)
        train_log["batch_training_sample_names"].append(training_batch_sample_names)
        train_log["batch_validation_sample_names"].append(valid_batch_sample_names)
        train_log["timestamps"].append(this_epoch_timestamp)

        if save_results:
            # Write training log so far
            epoch_training_log_path = os.path.join(
                output_dir, "training_log_e{}.pkl".format(this_epoch)
            )
            with open(epoch_training_log_path, "wb") as f:
                pickle.dump(train_log, f)

            # Remove old training log
            last_epoch_training_log_path = os.path.join(
                output_dir, "training_log_e{}.pkl".format(e)
            )
            if os.path.exists(last_epoch_training_log_path):
                os.remove(last_epoch_training_log_path)

            # Save this epoch
            dream_network.save_network(
                output_dir, "epoch_{}".format(this_epoch), overwrite=True
            )

    # Save results
    if save_results:
        # Rename the final training log instead of re-writing it
        training_log_path = os.path.join(output_dir, "training_log.pkl")
        if os.path.exists(training_log_path):
            os.remove(training_log_path)
        os.rename(epoch_training_log_path, training_log_path)

    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print("")
    print("Done.")
    print("")
    print("Total training time: {} seconds.".format(time.time() - training_start_time))
    print("")


if __name__ == "__main__":

    # Parse input arguments
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "-c", "--config-file",
        default="./config/config.yaml", type=str,
        help="Path to config file."
    )
    parser.add_argument(
        "-f",
        "--force-overwrite",
        action="store_true",
        default=True,
        help="Forces overwriting of analysis results in the provided directory.",
    )
    parser.add_argument(
        "-not-a",
        "--not-augment-data",
        action="store_true",
        default=True,
        help="Disable data augmentation. Without this flag, data augmentation is enabled by default.",
    )
    parser.add_argument(
        "-g",
        "--gpu-ids",
        nargs="+",
        type=int,
        default=None,
        help="The GPU IDs on which to train the network. Nothing specified means all GPUs will be utilized.",
    )
    parser.add_argument(
        "-s", "--random-seed", type=int, help="Manually specify the random seed."
    )
    parser.add_argument(
        "-r",
        "--resume-training",
        action="store_true",
        default=False,
        help="Resumes training. The epoch argument provided now is the new training duration. All arguments must match the previously trained networks.",
    )

    args = parser.parse_args()

    # Train the network
    train_network(args)
