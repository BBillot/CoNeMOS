# python imports
import os
import csv
import time
import numpy as np
import keras.layers as KL
from keras.models import Model

# project imports
from CoNeMOS import unet, evaluate

# third-party imports
from ext.lab2im import utils, layers, edit_volumes


def predict(path_images,
            path_segmentations,
            path_model,
            path_posteriors=None,
            path_volumes=None,
            condition_type=None,
            n_conditioned_layers=0,
            segm_regions=None,
            labels_to_regions_indices=None,
            min_pad=None,
            cropping=None,
            norm_perc=0.005,
            flip=False,
            sigma_smoothing=0,
            n_levels=4,
            unet_feat_count=16,
            feat_multiplier=2,
            activation='relu',
            final_pred_activation='sigmoid',
            n_conv_per_level=2,
            conv_size=3,
            norm_type='batch',
            multi_head=False,
            gt_folder=None,
            compute_distances=True,
            recompute=True,
            verbose=True):

    # prepare input/output filepaths
    path_images, path_segmentations, path_posteriors, path_volumes, compute, unique_vol_file = \
        prepare_output_files(path_images, path_segmentations, path_posteriors, path_volumes, recompute)

    # get label lists
    segm_regions = np.load(segm_regions) if segm_regions is not None else np.ones(1, dtype='int32')
    n_regions = len(segm_regions)
    assert np.array_equal(segm_regions, np.arange(1, n_regions + 1)), 'labels should be increasing without skips'
    size_condition_vector = n_regions if condition_type is not None else 0
    n_output_unet_channels = 1 if ((n_regions < 2) | (condition_type is not None)) else n_regions

    # prepare volumes if necessary
    if unique_vol_file & (path_volumes[0] is not None):
        write_csv(path_volumes[0], None, True, segm_regions, skip_first=False)

    # build network
    _, _, n_dims, _, _, _ = utils.get_volume_info(path_images[0])
    model_input_shape = [None] * n_dims + [1] if condition_type != 'channel' else [None] * n_dims + [1 + n_regions]
    net = build_model(path_model=path_model,
                      input_shape=model_input_shape,
                      n_output_unet_channels=n_output_unet_channels,
                      n_levels=n_levels,
                      unet_feat_count=unet_feat_count,
                      feat_multiplier=feat_multiplier,
                      activation=activation,
                      final_pred_activation=final_pred_activation,
                      n_conv_per_level=n_conv_per_level,
                      conv_size=conv_size,
                      norm_type=norm_type,
                      multi_head=multi_head,
                      condition_type=condition_type,
                      size_condition_vector=size_condition_vector,
                      n_conditioned_layers=n_conditioned_layers,
                      sigma_smoothing=sigma_smoothing,
                      flip=flip)

    # set cropping/padding
    if (cropping is not None) & (min_pad is not None):
        cropping = utils.reformat_to_list(cropping, length=n_dims, dtype='int')
        min_pad = utils.reformat_to_list(min_pad, length=n_dims, dtype='int')
        min_pad = np.minimum(cropping, min_pad)

    # perform segmentation
    list_times = []
    loop_info = utils.LoopInfo(len(path_images), 1, 'predicting', True)
    for i in range(len(path_images)):
        if verbose:
            loop_info.update(i)

        # compute segmentation only if needed
        if compute[i]:

            # preprocessing
            image, aff, h, im_res, shape, pad_idx, crop_idx = preprocess(path_images[i], n_levels, cropping, min_pad,
                                                                         norm_perc)

            # prediction
            if condition_type is not None:
                post_patch = list()
                for region in range(n_regions):
                    cond_input = (np.arange(n_regions) == np.array(region + 1)[..., None] - 1).astype('float32')
                    if 'film' in condition_type:
                        cond_input = utils.add_axis(cond_input, axis=0)
                        start = time.time()
                        post_patch.append(net.predict([image, cond_input]))
                        list_times.append(time.time() - start)
                    else:
                        cond_input = np.tile(utils.add_axis(cond_input, axis=[0] * (n_dims + 1)), image.shape)
                        start = time.time()
                        post_patch.append(net.predict(np.concatenate([image, cond_input], axis=-1)))
                        list_times.append(time.time() - start)
                post_patch = np.concatenate(post_patch, axis=-1)
            else:
                start = time.time()
                post_patch = net.predict(image)
                list_times.append(time.time() - start)

            # postprocessing
            seg, posteriors, volumes = postprocess(post_patch, shape, pad_idx, crop_idx, im_res)

            # write results to disk
            utils.save_volume(seg, aff, h, path_segmentations[i], dtype='int32')
            if path_posteriors[i] is not None:
                # if n_regions > 1:
                #     posteriors = utils.add_axis(posteriors, axis=[0, -1])
                utils.save_volume(posteriors, aff, h, path_posteriors[i], dtype='float32')

            # write volumes
            if path_volumes[i] is not None:
                row = [os.path.basename(path_images[i]).replace('.nii.gz', '')] + [str(vol) for vol in volumes]
                write_csv(path_volumes[i], row, unique_vol_file, segm_regions, skip_first=False)

    # print time stats
    if list_times:
        average_time = np.mean(list_times)
        if n_output_unet_channels != n_regions:
            average_time *= n_regions
        print('average time: %ss' % np.around(average_time, 3))

    # evaluate
    if gt_folder is not None:
        eval_folder = os.path.dirname(path_segmentations[0])
        if compute_distances:
            path_hausdorff = os.path.join(eval_folder, 'hausdorff_95.npy')
        else:
            path_hausdorff = None
        evaluate.evaluation(gt_folder,
                            eval_folder,
                            path_dice=os.path.join(eval_folder, 'dice.npy'),
                            path_hausdorff=path_hausdorff,
                            labels_to_regions_indices=labels_to_regions_indices,
                            recompute=recompute,
                            verbose=verbose)


def prepare_output_files(path_images, out_seg, out_posteriors, out_volumes, recompute):

    # check inputs
    assert path_images is not None, 'please specify an input file/folder (--i)'
    assert out_seg is not None, 'please specify an output file/folder (--o)'

    # convert path to absolute paths
    path_images = os.path.abspath(path_images)
    basename = os.path.basename(path_images)
    out_seg = os.path.abspath(out_seg)
    out_posteriors = os.path.abspath(out_posteriors) if (out_posteriors is not None) else out_posteriors
    out_volumes = os.path.abspath(out_volumes) if (out_volumes is not None) else out_volumes

    # path_images is a text file
    if basename[-4:] == '.txt':

        # input images
        if not os.path.isfile(path_images):
            raise Exception('provided text file containing paths of input images does not exist' % path_images)
        with open(path_images, 'r') as f:
            path_images = [line.replace('\n', '') for line in f.readlines() if line != '\n']

        # define helper to deal with outputs
        def text_helper(path, name):
            if path is not None:
                assert path[-4:] == '.txt', 'if path_images given as text file, so must be %s' % name
                with open(path, 'r') as ff:
                    path = [line.replace('\n', '') for line in ff.readlines() if line != '\n']
                recompute_files = [not os.path.isfile(p) for p in path]
            else:
                path = [None] * len(path_images)
                recompute_files = [False] * len(path_images)
            unique_file = False
            return path, recompute_files, unique_file

        # use helper on all outputs
        out_seg, recompute_seg, _ = text_helper(out_seg, 'path_segmentations')
        out_posteriors, recompute_post, _ = text_helper(out_posteriors, 'path_posteriors')
        out_volumes, recompute_volume, unique_volume_file = text_helper(out_volumes, 'path_volume')

    # path_images is a folder
    elif ('.nii.gz' not in basename) & ('.nii' not in basename) & ('.mgz' not in basename) & ('.npz' not in basename):

        # input images
        if os.path.isfile(path_images):
            raise Exception('Extension not supported for %s, only use: nii.gz, .nii, .mgz, or .npz' % path_images)
        path_images = utils.list_images_in_folder(path_images)

        # define helper to deal with outputs
        def helper_dir(path, name, file_type, suffix):
            unique_file = False
            if path is not None:
                assert path[-4:] != '.txt', '%s can only be given as text file when path_images is.' % name
                if file_type == 'csv':
                    if path[-4:] != '.csv':
                        print('%s provided without csv extension. Adding csv extension.' % name)
                        path += '.csv'
                    path = [path] * len(path_images)
                    recompute_files = [True] * len(path_images)
                    unique_file = True
                else:
                    if (path[-7:] == '.nii.gz') | (path[-4:] == '.nii') | (path[-4:] == '.mgz') | (path[-4:] == '.npz'):
                        raise Exception('Output FOLDER had a FILE extension' % path)
                    new_suffix = '_' + suffix if suffix else ''
                    path = [os.path.join(path, os.path.basename(p)) for p in path_images]
                    path = [p.replace('.nii', '%s.nii' % new_suffix) for p in path]
                    path = [p.replace('.mgz', '%s.mgz' % new_suffix) for p in path]
                    path = [p.replace('.npz', '%s.npz' % new_suffix) for p in path]
                    recompute_files = [not os.path.isfile(p) for p in path]
                utils.mkdir(os.path.dirname(path[0]))
            else:
                path = [None] * len(path_images)
                recompute_files = [False] * len(path_images)
            return path, recompute_files, unique_file

        # use helper on all outputs
        out_seg, recompute_seg, _ = helper_dir(out_seg, 'path_segmentations', '', '')
        out_posteriors, recompute_post, _ = helper_dir(out_posteriors, 'path_posteriors', '', '')
        out_volumes, recompute_volume, unique_volume_file = helper_dir(out_volumes, 'path_volumes', 'csv', '')

    # path_images is an image
    else:

        # input image
        assert os.path.isfile(path_images), 'file does not exist: %s \n' \
                                            'please make sure the path and the extension are correct' % path_images
        path_images = [path_images]

        # define helper to deal with outputs
        def helper_im(path, name, file_type, suffix):
            unique_file = False
            if path is not None:
                assert path[-4:] != '.txt', '%s can only be given as text file when path_images is.' % name
                if file_type == 'csv':
                    if path[-4:] != '.csv':
                        print('%s provided without csv extension. Adding csv extension.' % name)
                        path += '.csv'
                    recompute_files = [True]
                    unique_file = True
                else:
                    if ('.nii.gz' not in path) & ('.nii' not in path) & ('.mgz' not in path) & ('.npz' not in path):
                        new_suffix = '_' + suffix if suffix else ''
                        file_name = os.path.basename(path_images[0]).replace('.nii', '%s.nii' % new_suffix)
                        file_name = file_name.replace('.mgz', '%s.mgz' % new_suffix)
                        file_name = file_name.replace('.npz', '%s.npz' % new_suffix)
                        path = os.path.join(path, file_name)
                    recompute_files = [not os.path.isfile(path)]
                utils.mkdir(os.path.dirname(path))
            else:
                recompute_files = [False]
            path = [path]
            return path, recompute_files, unique_file

        # use helper on all outputs
        out_seg, recompute_seg, _ = helper_im(out_seg, 'path_segmentations', '', 'seg')
        out_posteriors, recompute_post, _ = helper_im(out_posteriors, 'path_posteriors', '', 'posteriors')
        out_volumes, recompute_volume, unique_volume_file = helper_im(out_volumes, 'path_volumes', 'csv', '')

    recompute_list = [recompute | re_seg | re_post | re_vol for (re_seg, re_post, re_vol)
                      in zip(recompute_seg, recompute_post, recompute_volume)]

    return path_images, out_seg, out_posteriors, out_volumes, recompute_list, unique_volume_file


def preprocess(path_image, n_levels, crop=None, min_pad=None, norm_perc=0.005):

    # read image and corresponding info
    im, shape, aff, n_dims, n_channels, h, im_res = utils.get_volume_info(path_image, True)

    # crop image if necessary
    if crop is not None:
        crop = utils.reformat_to_list(crop, length=n_dims, dtype='int')
        crop_shape = [utils.find_closest_number_divisible_by_m(s, 2 ** n_levels, 'higher') for s in crop]
        im, crop_idx = edit_volumes.crop_volume(im, cropping_shape=crop_shape, return_crop_idx=True)
    else:
        crop_idx = None

    # normalise image
    im = edit_volumes.rescale_volume(im, 0., 1., min_percentile=norm_perc, max_percentile=100 - norm_perc)

    # pad image
    input_shape = im.shape[:n_dims]
    pad_shape = [utils.find_closest_number_divisible_by_m(s, 2 ** n_levels, 'higher') for s in input_shape]
    if min_pad is not None:
        min_pad = utils.reformat_to_list(min_pad, length=n_dims, dtype='int')
        min_pad = [utils.find_closest_number_divisible_by_m(s, 2 ** n_levels, 'higher') for s in min_pad]
        pad_shape = np.maximum(pad_shape, min_pad)
    im, pad_idx = edit_volumes.pad_volume(im, padding_shape=pad_shape, return_pad_idx=True)

    # add batch and channel axes
    im = utils.add_axis(im) if n_channels > 1 else utils.add_axis(im, axis=[0, -1])

    return im, aff, h, im_res, shape, pad_idx, crop_idx


def build_model(path_model,
                input_shape,
                n_output_unet_channels,
                n_levels,
                unet_feat_count,
                feat_multiplier,
                activation,
                final_pred_activation,
                n_conv_per_level,
                conv_size,
                norm_type,
                multi_head,
                condition_type,
                n_conditioned_layers,
                size_condition_vector,
                sigma_smoothing,
                flip):

    assert os.path.isfile(path_model), "The provided model path does not exist."

    # build UNet
    net = unet.unet(input_shape=input_shape,
                    n_output_channels=n_output_unet_channels,
                    n_levels=n_levels,
                    n_features_init=unet_feat_count,
                    feat_mult=feat_multiplier,
                    activation=activation,
                    final_pred_activation=final_pred_activation,
                    n_conv_per_level=n_conv_per_level,
                    conv_size=conv_size,
                    norm_type=norm_type,
                    condition_type=condition_type,
                    n_conditioned_layers=n_conditioned_layers,
                    multi_head=multi_head,
                    size_condition_vector=size_condition_vector)
    net.load_weights(path_model, by_name=True)

    # segment flipped image and average the two segmentations
    if flip:
        input_image = net.inputs[0]
        seg = net.output
        image_flipped = layers.RandomFlip(axis=0, prob=1)(input_image)
        last_tensor = net(image_flipped)
        last_tensor = layers.RandomFlip(axis=0, prob=1)(last_tensor)
        last_tensor = KL.Lambda(lambda x: 0.5 * (x[0] + x[1]), name='average_flips')([seg, last_tensor])
        net = Model(inputs=net.inputs, outputs=last_tensor)

    # smooth posteriors if specified
    if sigma_smoothing > 0:
        last_tensor = net.output
        last_tensor = layers.GaussianBlur(sigma=sigma_smoothing)(last_tensor)
        net = Model(inputs=net.inputs, outputs=last_tensor)

    return net


def postprocess(post_patch, shape, pad_idx, crop_idx, im_res, threshold=0.5):

    # get posteriors
    post_patch = np.squeeze(post_patch)

    # crop posteriors and get segmentation
    post_patch = edit_volumes.crop_volume_with_idx(post_patch, pad_idx, n_dims=3, return_copy=False)

    # paste patches back to matrix of original image size
    if crop_idx is not None:
        if len(post_patch.shape) == len(shape):
            posteriors = np.zeros(shape=shape)
        else:
            posteriors = np.zeros(shape=[*shape, post_patch.shape[-1]])
        posteriors[crop_idx[0]:crop_idx[3], crop_idx[1]:crop_idx[4], crop_idx[2]:crop_idx[5], ...] = post_patch
    else:
        posteriors = post_patch

    # get seg
    seg = posteriors > threshold

    # compute volumes
    volumes = np.sum(seg, axis=tuple(range(0, len(shape))))
    volumes = np.around(volumes * np.prod(im_res), 3)

    return seg, posteriors, volumes


def postprocess_distance_maps(dist_patch, shape, pad_idx, crop_idx, labels_segmentation, aff, topology_classes=None):

    # get posteriors
    dist_patch = np.squeeze(dist_patch)

    # normalise posteriors and get hard segmentation
    seg_patch = labels_segmentation[dist_patch.argmin(-1).astype('int32')].astype('int32')

    # get the biggest component for each topological class separately
    if topology_classes is not None:
        for topology_class in np.unique(topology_classes)[1:]:
            tmp_topology_indices = np.where(topology_classes == topology_class)[0]
            tmp_mask = np.zeros_like(seg_patch)
            for tmp_label in labels_segmentation[tmp_topology_indices]:
                tmp_mask *= (seg_patch == tmp_label) * 1
            biggest_component = edit_volumes.get_largest_connected_component(tmp_mask)
            tmp_mask = tmp_mask & np.logical_not(biggest_component)
            seg_patch *= np.logical_not(tmp_mask)

    # crop back to original shape
    dist_patch = edit_volumes.crop_volume_with_idx(dist_patch, pad_idx, n_dims=3, return_copy=False)

    # paste patches back to matrix of original image size
    if crop_idx is not None:
        # we need to go through this because of the posteriors of the background, otherwise pad_volume would work
        seg = np.zeros(shape=shape, dtype='int32')
        dist_maps = np.zeros(shape=[*shape, labels_segmentation.shape[0]])
        dist_maps[..., 0] = 10 * np.ones(shape)  # place background around patch
        seg[crop_idx[0]:crop_idx[3], crop_idx[1]:crop_idx[4], crop_idx[2]:crop_idx[5]] = seg_patch
        dist_maps[crop_idx[0]:crop_idx[3], crop_idx[1]:crop_idx[4], crop_idx[2]:crop_idx[5], :] = dist_patch
    else:
        seg = seg_patch
        dist_maps = dist_patch

    # align prediction back to first orientation
    seg = edit_volumes.align_volume_to_ref(seg, aff=np.eye(4), aff_ref=aff, n_dims=3, return_copy=False)
    dist_maps = edit_volumes.align_volume_to_ref(dist_maps, np.eye(4), aff_ref=aff, n_dims=3, return_copy=False)

    return seg, dist_maps


def get_flip_indices(labels_segmentation, n_neutral_labels):

    # get position labels
    n_sided_labels = int((len(labels_segmentation) - n_neutral_labels) / 2)
    neutral_labels = labels_segmentation[:n_neutral_labels]
    left = labels_segmentation[n_neutral_labels:n_neutral_labels + n_sided_labels]

    # get correspondence between labels
    lr_corresp = np.stack([labels_segmentation[n_neutral_labels:n_neutral_labels + n_sided_labels],
                           labels_segmentation[n_neutral_labels + n_sided_labels:]])
    lr_corresp_unique, lr_corresp_indices = np.unique(lr_corresp[0, :], return_index=True)
    lr_corresp_unique = np.stack([lr_corresp_unique, lr_corresp[1, lr_corresp_indices]])
    lr_corresp_unique = lr_corresp_unique[:, 1:] if not np.all(lr_corresp_unique[:, 0]) else lr_corresp_unique

    # get unique labels
    labels_segmentation, unique_idx = np.unique(labels_segmentation, return_index=True)

    # get indices of corresponding labels
    lr_indices = np.zeros_like(lr_corresp_unique)
    for i in range(lr_corresp_unique.shape[0]):
        for j, lab in enumerate(lr_corresp_unique[i]):
            lr_indices[i, j] = np.where(labels_segmentation == lab)[0]

    # build 1d vector to swap LR corresponding labels taking into account neutral labels
    flip_indices = np.zeros_like(labels_segmentation)
    for i in range(len(flip_indices)):
        if labels_segmentation[i] in neutral_labels:
            flip_indices[i] = i
        elif labels_segmentation[i] in left:
            flip_indices[i] = lr_indices[1, np.where(lr_corresp_unique[0, :] == labels_segmentation[i])]
        else:
            flip_indices[i] = lr_indices[0, np.where(lr_corresp_unique[1, :] == labels_segmentation[i])]

    return labels_segmentation, flip_indices, unique_idx


def write_csv(path_csv, data, unique_file, labels=None, skip_first=True, last_first=False):

    # initialisation
    utils.mkdir(os.path.dirname(path_csv))
    if skip_first:
        labels = labels[1:]
    header = [str(lab) for lab in labels]
    if last_first:
        header = [header[-1]] + header[:-1]
    if (not unique_file) & (data is None):
        raise ValueError('data can only be None when initialising a unique volume file')

    # modify data
    if unique_file:
        if data is None:
            type_open = 'w'
            data = ['subject'] + header
        else:
            type_open = 'a'
        data = [data]
    else:
        type_open = 'w'
        header = [''] + header
        data = [header, data]

    # write csv
    with open(path_csv, type_open) as csvFile:
        writer = csv.writer(csvFile)
        writer.writerows(data)
