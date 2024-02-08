# python imports
import os
import numpy as np
from scipy.ndimage.morphology import distance_transform_edt

# third-party imports
from ext.lab2im import utils, edit_volumes


def fast_dice(x, y, labels):
    """Fast implementation of Dice scores.
    :param x: input label map
    :param y: input label map of the same size as x
    :param labels: numpy array of labels to evaluate on
    :return: numpy array with Dice scores in the same order as labels.
    """

    assert x.shape == y.shape, 'both inputs should have same size, had {} and {}'.format(x.shape, y.shape)

    if len(labels) > 1:
        # sort labels
        labels_sorted = np.sort(labels)

        # build bins for histograms
        label_edges = np.sort(np.concatenate([labels_sorted - 0.1, labels_sorted + 0.1]))
        label_edges = np.insert(label_edges, [0, len(label_edges)], [labels_sorted[0] - 0.1, labels_sorted[-1] + 0.1])

        # compute Dice and re-arrange scores in initial order
        hst = np.histogram2d(x.flatten(), y.flatten(), bins=label_edges)[0]
        idx = np.arange(start=1, stop=2 * len(labels_sorted), step=2)
        dice_score = 2 * np.diag(hst)[idx] / (np.sum(hst, 0)[idx] + np.sum(hst, 1)[idx] + 1e-5)
        dice_score = dice_score[np.searchsorted(labels_sorted, labels)]

    else:
        dice_score = dice(x == labels[0], y == labels[0])

    return dice_score


def dice(x, y):
    """Implementation of dice scores for 0/1 numpy array"""
    return 2 * np.sum(x * y) / (np.sum(x) + np.sum(y))


def surface_distances(x, y, hausdorff_percentile=None):
    """Computes the maximum boundary distance (Hausdorff distance), and the average boundary distance of two masks.
    :param x: numpy array (boolean or 0/1)
    :param y: numpy array (boolean or 0/1)
    :param hausdorff_percentile: (optional) percentile (from 0 to 100) for which to compute the Hausdorff distance.
    Set this to 100 to compute the real Hausdorff distance (default). Can also be a list, where HD will be computed for
    the provided values.
    :return: max_dist, mean_dist(, coordinate_max_distance)
    max_dist: scalar with HD computed for the given percentile (or list if hausdorff_percentile was given as a list).
    mean_dist: scalar with average surface distance
    coordinate_max_distance: only returned return_coordinate_max_distance is True."""

    assert x.shape == y.shape, 'both inputs should have same size, had {} and {}'.format(x.shape, y.shape)
    n_dims = len(x.shape)

    hausdorff_percentile = 100 if hausdorff_percentile is None else hausdorff_percentile
    hausdorff_percentile = utils.reformat_to_list(hausdorff_percentile)

    # crop x and y around ROI
    _, crop_x, _ = edit_volumes.crop_volume_around_region(x)
    _, crop_y, _ = edit_volumes.crop_volume_around_region(y)

    # set distances to maximum volume shape if they are not defined
    if (crop_x is None) | (crop_y is None):
        return max(x.shape)

    crop = np.concatenate([np.minimum(crop_x, crop_y)[:n_dims], np.maximum(crop_x, crop_y)[n_dims:]])
    x = edit_volumes.crop_volume_with_idx(x, crop)
    y = edit_volumes.crop_volume_with_idx(y, crop)

    # detect edge
    x_dist_int = distance_transform_edt(x * 1)
    x_edge = (x_dist_int == 1) * 1
    y_dist_int = distance_transform_edt(y * 1)
    y_edge = (y_dist_int == 1) * 1

    # calculate distance from edge
    x_dist = distance_transform_edt(np.logical_not(x_edge))
    y_dist = distance_transform_edt(np.logical_not(y_edge))

    # find distances from the 2 surfaces
    x_dists_to_y = y_dist[x_edge == 1]
    y_dists_to_x = x_dist[y_edge == 1]

    # compute final metrics
    max_dist = list()
    for hd_percentile in hausdorff_percentile:
        if hd_percentile == 100:
            max_dist.append(np.max(np.concatenate([x_dists_to_y, y_dists_to_x])))
        else:
            max_dist.append(np.percentile(np.concatenate([x_dists_to_y, y_dists_to_x]), hd_percentile))

    # convert max dist back to scalar if dist only computed for 1 percentile
    if len(max_dist) == 1:
        max_dist = max_dist[0]
    return max_dist


def evaluation(gt_dir,
               seg_dir,
               path_dice=None,
               path_hausdorff=None,
               labels_to_regions_indices=None,
               percentile_hausdorff=95,
               recompute=True,
               verbose=True):

    # check whether to recompute
    compute_dice = not os.path.isfile(path_dice) if (path_dice is not None) else True
    compute_hausdorff = not os.path.isfile(path_hausdorff) if (path_hausdorff is not None) else False

    if compute_dice | compute_hausdorff | recompute:

        # get list label maps to compare
        path_gts = utils.list_images_in_folder(gt_dir)
        path_segs = utils.list_images_in_folder(seg_dir)
        if len(path_gts) != len(path_segs):
            raise ValueError('gt and segmentation folders must have the same amount of label maps.')

        # figure out the number of regions
        _, _, n_dims, n_regions, _, _ = utils.get_volume_info(path_gts[0])
        if labels_to_regions_indices is not None:
            labels_to_regions_indices = utils.load_array_if_path(labels_to_regions_indices)
            n_regions = labels_to_regions_indices.shape[1]

        # initialise result matrices
        dice_coefs = np.nan * np.ones((n_regions, len(path_segs)))
        max_dists = np.nan * np.ones((n_regions, len(path_segs)))

        # loop over segmentations
        loop_info = utils.LoopInfo(len(path_segs), 10, 'evaluating', print_time=True)
        for subject_idx in range(len(path_gts)):
            if verbose:
                loop_info.update(subject_idx)

            # load gt labels and segmentation
            gt = utils.load_volume(path_gts[subject_idx], dtype='int32')
            seg = utils.load_volume(path_segs[subject_idx], dtype='int32')

            # convert them to multi-hot if necessary
            if len(gt.shape) == n_dims:
                if labels_to_regions_indices is not None:  # transform to multi-hot
                    gt = edit_volumes.labels_to_regions(gt, labels_to_regions_indices)
                else:
                    gt = utils.add_axis(gt, axis=-1)
            if len(seg.shape) == n_dims:
                if labels_to_regions_indices is not None:  # very unlikely to end-up here as segms are region-based
                    seg = edit_volumes.labels_to_regions(seg, labels_to_regions_indices)
                else:
                    seg = utils.add_axis(seg, axis=-1)

            # compute metrics
            for idx_region in range(n_regions):
                tmp_gt = gt[..., idx_region]
                tmp_seg = seg[..., idx_region]
                if np.max(tmp_gt) > 0:
                    dice_coefs[idx_region, subject_idx] = dice(tmp_gt, tmp_seg)
                    max_dists[idx_region, subject_idx] = surface_distances(tmp_gt, tmp_seg, percentile_hausdorff)

        # write results
        if path_dice is not None:
            utils.mkdir(os.path.dirname(path_dice))
            np.save(path_dice, dice_coefs)
        if path_hausdorff is not None:
            utils.mkdir(os.path.dirname(path_hausdorff))
            np.save(path_hausdorff, max_dists)
