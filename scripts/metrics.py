import torch
import numpy as np

from typing import Union, Optional


def compute_mpjpe(
        predicted_keypoints: torch.Tensor,
        ground_truth_keypoints: torch.Tensor
):
    l2_distance = torch.sqrt(torch.sum((predicted_keypoints - ground_truth_keypoints) ** 2, dim=-1))

    mpjpe = torch.mean(l2_distance).item()
    mjpje_per_keypoint = torch.mean(l2_distance, dim=0).cpu().numpy()

    return {
        'mpjpe': mpjpe,
        'mjpje_per_keypoint': mjpje_per_keypoint
    }


def compute_pck(
        predicted_keypoints: torch.Tensor,
        ground_truth_keypoints: torch.Tensor,
        thresholds: Optional[list[float]] = None
):
    if thresholds is None:
        thresholds = [5.0, 10.0, 20.0]  # TODO: change this

    l2_distance = torch.sqrt(torch.sum((predicted_keypoints - ground_truth_keypoints) ** 2, dim=-1))

    scores = {}
    for threshold in thresholds:
        correct = (l2_distance < threshold).float()

        scores[f'pck_{threshold}'] = correct.mean().item()

    return scores


def compute_add(
        predicted_keypoints: torch.Tensor,
        ground_truth_keypoints: torch.Tensor
):
    l2_distance = torch.sqrt(torch.sum((predicted_keypoints - ground_truth_keypoints) ** 2, dim=-1))

    add = l2_distance.mean().item()

    return {'add': add}


def compute_auc(
        predicted_keypoints: torch.Tensor,
        ground_truth_keypoints: torch.Tensor,
        max_threshold: float = 100.0,
        num_thresholds: int = 100
):
    thresholds = torch.linspace(0.0, max_threshold, num_thresholds, device=predicted_keypoints.device)

    l2_distance = torch.sqrt(torch.sum((predicted_keypoints - ground_truth_keypoints) ** 2, dim=-1))

    per_sample_add = l2_distance.mean(dim=1)

    accuracies = []
    for threshold in thresholds:
        accuracy = (per_sample_add < threshold).float().mean().item()

        accuracies.append(accuracy)

    auc = np.trapezoid(accuracies, thresholds.cpu().numpy()) / max_threshold

    return {'auc': auc * 100}
