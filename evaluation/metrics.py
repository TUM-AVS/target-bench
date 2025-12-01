"""
Trajectory comparison metrics for evaluating world model performance.

Handles trajectories with different numbers of points by interpolation.
"""

import numpy as np
from scipy.interpolate import interp1d


def interpolate_trajectory(trajectory, num_points):
    """
    Interpolate a trajectory to have a specific number of points.
    
    Args:
        trajectory: numpy array of shape (N, 2) or (N, 3)
        num_points: desired number of points
        
    Returns:
        numpy array of shape (num_points, 2) or (num_points, 3)
    """
    if len(trajectory) == num_points:
        return trajectory
    
    # Create parameter t from 0 to 1
    t_original = np.linspace(0, 1, len(trajectory))
    t_new = np.linspace(0, 1, num_points)
    
    # Interpolate each dimension
    interpolated = np.zeros((num_points, trajectory.shape[1]))
    for dim in range(trajectory.shape[1]):
        interpolator = interp1d(t_original, trajectory[:, dim], kind='linear')
        interpolated[:, dim] = interpolator(t_new)
    
    return interpolated


def compute_ade(pred_trajectory, gt_trajectory, interpolate_to='gt'):
    """
    Compute Average Displacement Error (ADE).
    
    ADE is the average L2 distance between predicted and ground truth 
    trajectories across all timesteps.
    
    Args:
        pred_trajectory: predicted trajectory (N, 2)
        gt_trajectory: ground truth trajectory (M, 2)
        interpolate_to: 'gt' to interpolate pred to gt length,
                       'pred' to interpolate gt to pred length,
                       'max' to interpolate both to max length,
                       'min' to trim both to min length
                       
    Returns:
        float: ADE in meters
    """
    pred_trajectory = np.array(pred_trajectory)
    gt_trajectory = np.array(gt_trajectory)
    
    if len(pred_trajectory) == 0 or len(gt_trajectory) == 0:
        return float('inf')
    
    # Handle interpolation strategy
    if interpolate_to == 'gt':
        pred_interp = interpolate_trajectory(pred_trajectory, len(gt_trajectory))
        gt_interp = gt_trajectory
    elif interpolate_to == 'pred':
        pred_interp = pred_trajectory
        gt_interp = interpolate_trajectory(gt_trajectory, len(pred_trajectory))
    elif interpolate_to == 'max':
        max_len = max(len(pred_trajectory), len(gt_trajectory))
        pred_interp = interpolate_trajectory(pred_trajectory, max_len)
        gt_interp = interpolate_trajectory(gt_trajectory, max_len)
    elif interpolate_to == 'min':
        min_len = min(len(pred_trajectory), len(gt_trajectory))
        pred_interp = pred_trajectory[:min_len]
        gt_interp = gt_trajectory[:min_len]
    else:
        raise ValueError(f"Unknown interpolate_to mode: {interpolate_to}")
    
    # Compute average L2 distance
    displacements = np.linalg.norm(pred_interp - gt_interp, axis=1)
    ade = np.mean(displacements)
    
    return float(ade)


def compute_fde(pred_trajectory, gt_trajectory):
    """
    Compute Final Displacement Error (FDE).
    
    FDE is the L2 distance between the final points of predicted 
    and ground truth trajectories.
    
    Args:
        pred_trajectory: predicted trajectory (N, 2)
        gt_trajectory: ground truth trajectory (M, 2)
        
    Returns:
        float: FDE in meters
    """
    pred_trajectory = np.array(pred_trajectory)
    gt_trajectory = np.array(gt_trajectory)
    
    if len(pred_trajectory) == 0 or len(gt_trajectory) == 0:
        return float('inf')
    
    # Get final points
    pred_final = pred_trajectory[-1]
    gt_final = gt_trajectory[-1]
    
    # Compute L2 distance
    fde = np.linalg.norm(pred_final - gt_final)
    
    return float(fde)


def compute_miss_rate(pred_trajectory, gt_trajectory, threshold=2.0, interpolate_to='gt'):
    """
    Compute Miss Rate.
    
    Miss rate is the percentage of predicted points that are more than 
    a threshold distance away from corresponding ground truth points.
    
    Args:
        pred_trajectory: predicted trajectory (N, 2)
        gt_trajectory: ground truth trajectory (M, 2)
        threshold: distance threshold in meters (default 2.0m)
        interpolate_to: same options as compute_ade
        
    Returns:
        float: miss rate as a percentage (0-100)
    """
    pred_trajectory = np.array(pred_trajectory)
    gt_trajectory = np.array(gt_trajectory)
    
    if len(pred_trajectory) == 0 or len(gt_trajectory) == 0:
        return 100.0
    
    # Handle interpolation strategy
    if interpolate_to == 'gt':
        pred_interp = interpolate_trajectory(pred_trajectory, len(gt_trajectory))
        gt_interp = gt_trajectory
    elif interpolate_to == 'pred':
        pred_interp = pred_trajectory
        gt_interp = interpolate_trajectory(gt_trajectory, len(pred_trajectory))
    elif interpolate_to == 'max':
        max_len = max(len(pred_trajectory), len(gt_trajectory))
        pred_interp = interpolate_trajectory(pred_trajectory, max_len)
        gt_interp = interpolate_trajectory(gt_trajectory, max_len)
    elif interpolate_to == 'min':
        min_len = min(len(pred_trajectory), len(gt_trajectory))
        pred_interp = pred_trajectory[:min_len]
        gt_interp = gt_trajectory[:min_len]
    else:
        raise ValueError(f"Unknown interpolate_to mode: {interpolate_to}")
    
    # Compute distances
    displacements = np.linalg.norm(pred_interp - gt_interp, axis=1)
    
    # Count misses (points beyond threshold)
    num_misses = np.sum(displacements > threshold)
    miss_rate = (num_misses / len(displacements)) * 100.0
    
    return float(miss_rate)


def compute_soft_endpoint(pred_trajectory, gt_trajectory, sigma=0.6):
    """
    Compute Soft Endpoint (SE) metric.
    
    SE measures how close the predicted trajectory's final point is to the 
    ground-truth endpoint using a Gaussian-shaped distance penalty.
    
    SE = exp(-||pred_final - gt_final||^2 / (2 * sigma^2))
    
    Args:
        pred_trajectory: predicted trajectory (N, 2)
        gt_trajectory: ground truth trajectory (M, 2)
        sigma: tolerance parameter in meters (default 0.6m)
        
    Returns:
        float: SE score in [0, 1], where 1 is perfect alignment
    """
    pred_trajectory = np.array(pred_trajectory)
    gt_trajectory = np.array(gt_trajectory)
    
    if len(pred_trajectory) == 0 or len(gt_trajectory) == 0:
        return 0.0
    
    # Get final points
    pred_final = pred_trajectory[-1]
    gt_final = gt_trajectory[-1]
    
    # Compute Gaussian penalty
    distance_squared = np.sum((pred_final - gt_final) ** 2)
    se = np.exp(-distance_squared / (2 * sigma ** 2))
    
    return float(se)


def compute_approach_consistency(pred_trajectory, gt_trajectory, 
                                 num_ref_points=20,
                                 sigma_min=0.15, sigma_max=0.5, beta=0.25,
                                 gamma=5.0):
    """
    Compute Approach Consistency (AC) metric.
    
    AC evaluates whether the predicted trajectory maintains a similar 
    approaching tendency toward the target as the ground-truth path.
    
    Args:
        pred_trajectory: predicted trajectory (N, 2)
        gt_trajectory: ground truth trajectory (M, 2)
        num_ref_points: number of reference points to sample (default 20)
        sigma_min: minimum corridor radius in meters (default 0.15)
        sigma_max: maximum corridor radius in meters (default 0.5)
        beta: corridor shape parameter (default 0.25)
        gamma: penalty sharpness parameter (default 5.0)
        
    Returns:
        float: AC score in [0, 1]
    """
    pred_trajectory = np.array(pred_trajectory)
    gt_trajectory = np.array(gt_trajectory)
    
    if len(pred_trajectory) == 0 or len(gt_trajectory) == 0:
        return 0.0
    
    # Sample reference points uniformly along ground truth trajectory
    if len(gt_trajectory) < num_ref_points:
        ref_points = gt_trajectory
        ref_indices = np.arange(len(gt_trajectory))
    else:
        ref_indices = np.linspace(0, len(gt_trajectory) - 1, num_ref_points, dtype=int)
        ref_points = gt_trajectory[ref_indices]
    
    # Compute normalized progress for each reference point
    progress = ref_indices / max(len(gt_trajectory) - 1, 1)
    
    # Compute variable radius for each reference point (progress-dependent corridor)
    # Narrow at start and end, wider in the middle
    radii = sigma_min + (sigma_max - sigma_min) * np.exp(-((progress - 0.5) ** 2) / (2 * beta ** 2))
    
    # Check coverage for each predicted point
    num_covered = 0
    for pred_point in pred_trajectory:
        # Compute distances to all reference points
        distances = np.linalg.norm(ref_points - pred_point, axis=1)
        
        # Check if point falls within any corridor region
        if np.any(distances <= radii):
            num_covered += 1
    
    num_pred = len(pred_trajectory)
    
    # Compute AC score
    if num_covered == num_pred:
        ac = 1.0
    else:
        uncovered_ratio = (num_pred - num_covered) / num_pred
        ac = np.exp(-gamma * uncovered_ratio)
    
    return float(ac)


def compute_overall_score(ade, fde, miss_rate, se, ac, 
                         tau_ade=1.0, tau_fde=1.0,
                         weights=None):
    """
    Compute Overall Trajectory Evaluation Score as defined in metrics.md.
    
    The overall score is a weighted combination of normalized metrics:
    S = w_ADE * exp(-ADE/τ_ADE) + w_FDE * exp(-FDE/τ_FDE) + 
        w_MR * (1 - MR/100) + (w_SE + w_AC) * SE * AC
    
    Args:
        ade: Average Displacement Error in meters
        fde: Final Displacement Error in meters
        miss_rate: Miss Rate as percentage [0, 100]
        se: Soft Endpoint score [0, 1]
        ac: Approach Consistency score [0, 1]
        tau_ade: scale parameter for ADE (default 1.0)
        tau_fde: scale parameter for FDE (default 1.0)
        weights: dict with keys 'ade', 'fde', 'mr', 'se', 'ac'
                 (default from metrics.md)
    
    Returns:
        float: Overall score in [0, 1] (higher is better)
    """
    # Default weights from metrics.md
    if weights is None:
        weights = {
            'ade': 0.05,
            'fde': 0.10,
            'mr': 0.10,
            'se': 0.35,
            'ac': 0.30
        }
    
    # Normalize each metric to [0, 1] range
    s_ade = np.exp(-ade / tau_ade) if ade != float('inf') else 0.0
    s_fde = np.exp(-fde / tau_fde) if fde != float('inf') else 0.0
    s_mr = 1.0 - (miss_rate / 100.0)  # miss_rate is percentage
    s_se = se
    s_ac = ac
    
    # Compute weighted sum
    overall_score = (
        weights['ade'] * s_ade +
        weights['fde'] * s_fde +
        weights['mr'] * s_mr +
        (weights['se'] + weights['ac']) * s_se * s_ac
    )
    
    return float(overall_score)


def compute_all_metrics(pred_trajectory, gt_trajectory, 
                       miss_threshold=2.0, 
                       interpolate_mode='gt',
                       se_sigma=0.6,
                       ac_params=None,
                       overall_score_params=None):
    """
    Compute all trajectory metrics at once.
    
    Metrics computed:
    - ADE: Average Displacement Error
    - FDE: Final Displacement Error
    - MR: Miss Rate
    - SE: Soft Endpoint
    - AC: Approach Consistency
    - Overall Score: Weighted combination of all metrics
    
    Args:
        pred_trajectory: predicted trajectory (N, 2)
        gt_trajectory: ground truth trajectory (M, 2)
        miss_threshold: threshold for miss rate in meters
        interpolate_mode: interpolation strategy ('gt', 'pred', 'max', 'min')
        se_sigma: sigma parameter for Soft Endpoint (default 0.6)
        ac_params: dict of parameters for Approach Consistency (optional)
        overall_score_params: dict with 'tau_ade', 'tau_fde', 'weights' (optional)
        
    Returns:
        dict: dictionary containing all metrics
    """
    if ac_params is None:
        ac_params = {}
    if overall_score_params is None:
        overall_score_params = {}
    
    # Compute individual metrics
    ade = compute_ade(pred_trajectory, gt_trajectory, interpolate_mode)
    fde = compute_fde(pred_trajectory, gt_trajectory)
    miss_rate = compute_miss_rate(pred_trajectory, gt_trajectory, 
                                   miss_threshold, interpolate_mode)
    se = compute_soft_endpoint(pred_trajectory, gt_trajectory, se_sigma)
    ac = compute_approach_consistency(pred_trajectory, gt_trajectory, **ac_params)
    
    # Compute overall score
    overall = compute_overall_score(ade, fde, miss_rate, se, ac, **overall_score_params)
    
    metrics = {
        'ade': ade,
        'fde': fde,
        'miss_rate': miss_rate,
        'se': se,
        'ac': ac,
        'overall_score': overall,
        'miss_threshold': miss_threshold,
        'interpolate_mode': interpolate_mode,
        'pred_points': len(pred_trajectory),
        'gt_points': len(gt_trajectory)
    }
    
    return metrics

