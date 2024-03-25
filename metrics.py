"""Anomaly metrics."""
from bisect import bisect

from kornia.contrib import connected_components
from numpy import ndarray
from scipy.ndimage import label
from sklearn import metrics
from torch import Tensor
from torchmetrics.functional.classification import binary_roc  # 0.10.3
import numpy as np
import torch
import torchmetrics.functional as MF

def pixel_params_verify(gts, preds):
    if isinstance(gts, list):
        gts = np.stack(gts)
    if isinstance(preds, list):
        preds = np.stack(preds)
    if preds.ndim == 4:
        preds = preds[:, 0]
    if gts.ndim == 4:
        gts = gts[:, 0]
    gts[gts > 0.5] = 1
    gts[gts < 1] = 0
    gts = gts.astype(int)
    assert isinstance(preds, ndarray), "type(preds) must be ndarray"
    assert isinstance(gts, ndarray), "type(gts) must be ndarray"
    assert preds.ndim == 3, "preds.ndim must be 3 (num_test_data, h, w)"
    assert gts.ndim == 3, "gts.ndim must be 3 (num_test_data, h, w)"
    assert preds.shape == gts.shape, "preds.shape and gts.shape must be same"
    assert set(gts.flatten()) == {0, 1}, "set(gts.flatten()) must be {0, 1}"
    return gts, preds

def compute_image_auc(
    gts, preds
):
    """
    Computes retrieval statistics (AUROC, FPR, TPR).

    Args:
        gts: [np.array or list] [N] Binary labels - 1
                                    if image is an anomaly, 0 if not.
        preds: [np.array or list] [N] Assignment weights
                                    per image. Higher indicates higher
                                    probability of being an anomaly.
    """
    return metrics.roc_auc_score(gts, preds)

def compute_image_auc_torch(gts, preds):
    return MF.auroc(preds, gts).item()

def compute_pixel_auc(gts, preds):
    """
    Computes pixel-wise statistics (AUROC, FPR, TPR) for anomaly segmentations
    and ground truth segmentation masks.

    Args:
        gts: [list of np.arrays or np.array] [NxHxW] Contains
                            predefined ground truth segmentation masks
        preds: [list of np.arrays or np.array] [NxHxW] Contains
                                generated segmentation masks.
    """
    gts, preds = pixel_params_verify(gts, preds)
    return metrics.roc_auc_score(gts.ravel(), preds.ravel())

def compute_pixel_auc_torch(gts, preds):
    return MF.auroc(preds.reshape(-1), gts.reshape(-1)).item()

def compute_ap(gts: ndarray, preds: ndarray):
    gts, preds = pixel_params_verify(gts, preds)
    return metrics.average_precision_score(gts.ravel(), preds.ravel())

def compute_ap_torch(gts, preds):
    return MF.average_precision(preds.reshape(-1), gts.reshape(-1)).item()
    
# original mvtec pro metrics
#-------------------------------------------#
def trapezoid(x, y, x_max=None):
    """
    This function calculates the definit integral of a curve given by
    x- and corresponding y-values. In contrast to, e.g., 'numpy.trapz()',
    this function allows to define an upper bound to the integration range by
    setting a value x_max.

    Points that do not have a finite x or y value will be ignored with a
    warning.

    Args:
        x: Samples from the domain of the function to integrate
          Need to be sorted in ascending order. May contain the same value
          multiple times. In that case, the order of the corresponding
          y values will affect the integration with the trapezoidal rule.
        y: Values of the function corresponding to x values.
        x_max: Upper limit of the integration. The y value at max_x will be
          determined by interpolating between its neighbors. Must not lie
          outside of the range of x.

    Returns:
        Area under the curve.
    """

    x = np.asarray(x)
    y = np.asarray(y)
    finite_mask = np.logical_and(np.isfinite(x), np.isfinite(y))
    x = x[finite_mask]
    y = y[finite_mask]

    # Introduce a correction term if max_x is not an element of x.
    correction = 0.
    if x_max is not None:
        if x_max not in x:
            # Get the insertion index that would keep x sorted after
            # np.insert(x, ins, x_max).
            ins = bisect(x, x_max)
            # x_max must be between the minimum and the maximum, so the
            # insertion_point cannot be zero or len(x).
            assert 0 < ins < len(x)

            # Calculate the correction term which is the integral between
            # the last x[ins-1] and x_max. Since we do not know the exact value
            # of y at x_max, we interpolate between y[ins] and y[ins-1].
            y_interp = y[ins - 1] + ((y[ins] - y[ins - 1]) *
                                     (x_max - x[ins - 1]) /
                                     (x[ins] - x[ins - 1]))
            correction = 0.5 * (y_interp + y[ins - 1]) * (x_max - x[ins - 1])

        # Cut off at x_max.
        mask = x <= x_max
        x = x[mask]
        y = y[mask]

    # Return area under the curve using the trapezoidal rule.
    return np.sum(0.5 * (y[1:] + y[:-1]) * (x[1:] - x[:-1])) + correction


def _compute_pro_original_mvtec(anomaly_maps, ground_truth_maps):
    """Compute the PRO curve for a set of anomaly maps with corresponding ground
    truth maps.

    Args:
        anomaly_maps: List of anomaly maps (2D numpy arrays) that contain a
          real-valued anomaly score at each pixel.

        ground_truth_maps: List of ground truth maps (2D numpy arrays) that
          contain binary-valued ground truth labels for each pixel.
          0 indicates that a pixel is anomaly-free.
          1 indicates that a pixel contains an anomaly.

    Returns:
        fprs: numpy array of false positive rates.
        pros: numpy array of corresponding PRO values.
    """


    # Structuring element for computing connected components.
    structure = np.ones((3, 3), dtype=int)

    num_ok_pixels = 0
    num_gt_regions = 0

    shape = (len(anomaly_maps),
             anomaly_maps[0].shape[0],
             anomaly_maps[0].shape[1])
    fp_changes = np.zeros(shape, dtype=np.uint32)
    assert shape[0] * shape[1] * shape[2] < np.iinfo(fp_changes.dtype).max, \
        'Potential overflow when using np.cumsum(), consider using np.uint64.'

    pro_changes = np.zeros(shape, dtype=np.float64)

    for gt_ind, gt_map in enumerate(ground_truth_maps):

        # Compute the connected components in the ground truth map.
        labeled, n_components = label(gt_map, structure)
        num_gt_regions += n_components

        # Compute the mask that gives us all ok pixels.
        ok_mask = labeled == 0
        num_ok_pixels_in_map = np.sum(ok_mask)
        num_ok_pixels += num_ok_pixels_in_map

        # Compute by how much the FPR changes when each anomaly score is
        # added to the set of positives.
        # fp_change needs to be normalized later when we know the final value
        # of num_ok_pixels -> right now it is only the change in the number of
        # false positives
        fp_change = np.zeros_like(gt_map, dtype=fp_changes.dtype)
        fp_change[ok_mask] = 1

        # Compute by how much the PRO changes when each anomaly score is
        # added to the set of positives.
        # pro_change needs to be normalized later when we know the final value
        # of num_gt_regions.
        pro_change = np.zeros_like(gt_map, dtype=np.float64)
        for k in range(n_components):
            region_mask = labeled == (k + 1)
            region_size = np.sum(region_mask)
            pro_change[region_mask] = 1. / region_size

        fp_changes[gt_ind, :, :] = fp_change
        pro_changes[gt_ind, :, :] = pro_change

    # Flatten the numpy arrays before sorting.
    anomaly_scores_flat = np.array(anomaly_maps).ravel()
    fp_changes_flat = fp_changes.ravel()
    pro_changes_flat = pro_changes.ravel()

    # Sort all anomaly scores.
    sort_idxs = np.argsort(anomaly_scores_flat).astype(np.uint32)[::-1]

    # Info: np.take(a, ind, out=a) followed by b=a instead of
    # b=a[ind] showed to be more memory efficient.
    np.take(anomaly_scores_flat, sort_idxs, out=anomaly_scores_flat)
    anomaly_scores_sorted = anomaly_scores_flat
    np.take(fp_changes_flat, sort_idxs, out=fp_changes_flat)
    fp_changes_sorted = fp_changes_flat
    np.take(pro_changes_flat, sort_idxs, out=pro_changes_flat)
    pro_changes_sorted = pro_changes_flat

    del sort_idxs

    # Get the (FPR, PRO) curve values.
    np.cumsum(fp_changes_sorted, out=fp_changes_sorted)
    fp_changes_sorted = fp_changes_sorted.astype(np.float32, copy=False)
    np.divide(fp_changes_sorted, num_ok_pixels, out=fp_changes_sorted)
    fprs = fp_changes_sorted

    np.cumsum(pro_changes_sorted, out=pro_changes_sorted)
    np.divide(pro_changes_sorted, num_gt_regions, out=pro_changes_sorted)
    pros = pro_changes_sorted

    # Merge (FPR, PRO) points that occur together at the same threshold.
    # For those points, only the final (FPR, PRO) point should be kept.
    # That is because that point is the one that takes all changes
    # to the FPR and the PRO at the respective threshold into account.
    # -> keep_mask is True if the subsequent score is different from the
    # score at the respective position.
    # anomaly_scores_sorted = [7, 4, 4, 4, 3, 1, 1]
    # ->          keep_mask = [T, F, F, T, T, F]
    keep_mask = np.append(np.diff(anomaly_scores_sorted) != 0, np.True_)
    del anomaly_scores_sorted

    fprs = fprs[keep_mask]
    pros = pros[keep_mask]
    del keep_mask

    # To mitigate the adding up of numerical errors during the np.cumsum calls,
    # make sure that the curve ends at (1, 1) and does not contain values > 1.
    np.clip(fprs, a_min=None, a_max=1., out=fprs)
    np.clip(pros, a_min=None, a_max=1., out=pros)

    # Make the fprs and pros start at 0 and end at 1.
    zero = np.array([0.])
    one = np.array([1.])

    return np.concatenate((zero, fprs, one)), np.concatenate((zero, pros, one))

def compute_pro(gts, preds):
    '''
    >>> original url : https://www.mvtec.com/company/research/datasets/mvtec-ad/
    >>> code url : https://www.mydrive.ch/shares/60736/698155e0e6d0467c4ff6203b16a31dc9/download/439517473-1665667812/mvtec_ad_evaluation.tar.xz
    '''
    gts, preds = pixel_params_verify(gts, preds)
    integration_limit = 0.3

    # original code
    all_fprs, all_pros = _compute_pro_original_mvtec(anomaly_maps=preds, ground_truth_maps=gts)
    au_pro = trapezoid(all_fprs, all_pros, x_max=integration_limit)
    au_pro /= integration_limit
    return au_pro

# https://github.com/openvinotoolkit/anomalib/blob/main/src/anomalib/utils/metrics/aupro.py
def interp1d(old_x: Tensor, old_y: Tensor, new_x: Tensor) -> Tensor:
    """Function to interpolate a 1D signal linearly to new sampling points.

    Args:
        old_x (Tensor): original 1-D x values (same size as y)
        old_y (Tensor): original 1-D y values (same size as x)
        new_x (Tensor): x-values where y should be interpolated at

    Returns:
        Tensor: y-values at corresponding new_x values.
    """

    # Compute slope
    eps = torch.finfo(old_y.dtype).eps
    slope = (old_y[1:] - old_y[:-1]) / (eps + (old_x[1:] - old_x[:-1]))

    # Prepare idx for linear interpolation
    idx = torch.searchsorted(old_x, new_x)

    # searchsorted looks for the index where the values must be inserted
    # to preserve order, but we actually want the preceeding index.
    idx -= 1
    # we clamp the index, because the number of intervals = old_x.size(0) -1,
    # and the left neighbour should hence be at most number of intervals -1, i.e. old_x.size(0) - 2
    idx = torch.clamp(idx, 0, old_x.size(0) - 2)

    # perform actual linear interpolation
    y_new = old_y[idx] + slope[idx] * (new_x - old_x[idx])

    return y_new

def compute_pro_torch(gts, preds, integration_limit=0.3):
    if gts.min() < 0 or gts.max() > 1:
        raise ValueError(
            f"kornia.contrib.connected_components expects input to lie in the interval [0, 1], but found "
            f"interval was [{gts.min()}, {gts.max()}]."
        )
    components = connected_components(gts.unsqueeze(1).type(torch.float), num_iterations=1000)  # kornia expects N1HW format, kornia expects FloatTensor
    labels = components.unique()
    for new_label, old_label in enumerate(labels):
        components[components == old_label] = new_label
    cca = components.int().flatten()
    thresholds = None
    fpr: Tensor = binary_roc(preds=preds.flatten(), target=gts.flatten(), thresholds=thresholds)[0]  # only need fpr
    output_size = torch.where(fpr <= integration_limit)[0].size(0)
    tpr = torch.zeros(output_size, device=preds.device, dtype=torch.float)
    fpr = torch.zeros(output_size, device=preds.device, dtype=torch.float)
    new_idx = torch.arange(0, output_size, device=preds.device, dtype=torch.float)
    labels = cca.unique()[1:]  # 0 is background
    background = cca == 0
    _fpr: Tensor
    _tpr: Tensor
    for label in labels:
        interp: bool = False
        new_idx[-1] = output_size - 1
        mask = cca == label
        _fpr, _tpr = binary_roc(preds=preds.flatten()[background | mask], target=mask[background | mask], thresholds=thresholds)[:-1]
        if _fpr[_fpr <= integration_limit].max() == 0:
            _fpr_limit = _fpr[_fpr > integration_limit].min()
        else:
            _fpr_limit = integration_limit
        _fpr_idx = torch.where(_fpr <= _fpr_limit)[0]
        if not torch.allclose(_fpr[_fpr_idx].max(), torch.tensor(integration_limit).to(preds.device)):
            _tmp_idx = torch.searchsorted(_fpr, integration_limit)
            _fpr_idx = torch.cat([_fpr_idx, _tmp_idx.unsqueeze_(0)])
            _slope = 1 - ((_fpr[_tmp_idx] - integration_limit) / (_fpr[_tmp_idx] - _fpr[_tmp_idx - 1]))
            interp = True
        _fpr = _fpr[_fpr_idx]
        _tpr = _tpr[_fpr_idx]
        _fpr_idx = _fpr_idx.float()
        _fpr_idx /= _fpr_idx.max()
        _fpr_idx *= new_idx.max()
        if interp:
            new_idx[-1] = _fpr_idx[-2] + ((_fpr_idx[-1] - _fpr_idx[-2]) * _slope)
        _tpr = interp1d(_fpr_idx, _tpr, new_idx)
        _fpr = interp1d(_fpr_idx, _fpr, new_idx)
        tpr += _tpr
        fpr += _fpr
    # Actually perform the averaging
    tpr /= labels.size(0)
    fpr /= labels.size(0)
    aupro = MF.auc(fpr, tpr, reorder=True)
    aupro = aupro / fpr[-1]  # normalize the area
    return aupro.item()