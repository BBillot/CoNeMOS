# python imports
import os
import copy
import numpy as np
from keras import models

# third-party imports
from ext.lab2im import utils

from CoNeMOS import data_loader, augmentation, unet, metrics


def check_training(image_dir,
                   labels_dir,
                   result_dir,
                   n_examples,
                   names_output_tensors,
                   filenames_output_tensors,
                   condition_type=None,
                   n_conditioned_layers=0,
                   segm_regions=None,
                   labels_to_regions_indices=None,
                   label_descriptor_dir=None,
                   subjects_prob=None,
                   data_perc=100,
                   mask_loss=False,
                   batchsize=1,
                   output_shape=None,
                   flip_axis=None,
                   scaling_bounds=.2,
                   rotation_bounds=180,
                   shearing_bounds=.012,
                   translation_bounds=10,
                   nonlin_std=4.,
                   nonlin_scale=.05,
                   randomise_res=False,
                   downsample=False,
                   max_res_iso=6.,
                   max_res_aniso=6.,
                   blur_factor=1.05,
                   bias_field_std=.8,
                   bias_scale=.03,
                   noise_hr=0.03,
                   noise_lr=0.02,
                   norm_perc=0.005,
                   gamma=0.4,
                   n_levels=4,
                   unet_feat_count=16,
                   feat_multiplier=2,
                   activation='elu',
                   final_pred_activation='sigmoid',
                   n_conv_per_level=2,
                   conv_size=3,
                   norm_type='batch',
                   multi_head=False,
                   boundary_weights=100,
                   reduce_type='mean',
                   checkpoint=None):

    if condition_type not in ['channel', 'film', 'film_image', 'film_last', 'film_image_last']:
        raise ValueError('condition_type should be one of: channel, film, film_image, film_last, film_image_last')
    if mask_loss and condition_type is not None:
        raise ValueError('cannot use loss masking with conditioning')
    if condition_type is not None and multi_head:
        raise ValueError('cannot use multi-task learning (simultaneous segmetnation of all labels) with conditioning')

    # prepare data files
    path_images, path_labels, path_descriptors, subjects_prob = data_loader.get_paths(image_dir,
                                                                                      labels_dir,
                                                                                      label_descriptor_dir,
                                                                                      subjects_prob,
                                                                                      data_perc)

    # get label lists
    segm_regions = np.load(segm_regions) if segm_regions is not None else np.ones(1, dtype='int32')
    n_regions = len(segm_regions)
    assert np.array_equal(segm_regions, np.arange(1, n_regions + 1)), 'labels should be increasing without skips'
    size_condition_vector = n_regions if condition_type is not None else 0
    n_output_unet_channels = 1 if ((n_regions < 2) | (condition_type is not None)) else n_regions
    if labels_to_regions_indices is not None:
        labels_to_regions_indices = np.load(labels_to_regions_indices)
        assert labels_to_regions_indices.shape[1] == n_regions, 'labels to region mapping should have n_regions columns'
    else:
        labels_to_regions_indices = None

    # create augmentation model
    im_shape, _, _, _, _, atlas_res = utils.get_volume_info(path_images[0], aff_ref=np.eye(4))
    augmentation_model = augmentation.build_augmentation_model(im_shape,
                                                               atlas_res,
                                                               condition_type,
                                                               size_condition_vector,
                                                               output_shape=output_shape,
                                                               output_div_by_n=2 ** n_levels,
                                                               flip_axis=flip_axis,
                                                               scaling_bounds=scaling_bounds,
                                                               rotation_bounds=rotation_bounds,
                                                               shearing_bounds=shearing_bounds,
                                                               translation_bounds=translation_bounds,
                                                               nonlin_std=nonlin_std,
                                                               nonlin_scale=nonlin_scale,
                                                               randomise_res=randomise_res,
                                                               downsample=downsample,
                                                               max_res_iso=max_res_iso,
                                                               max_res_aniso=max_res_aniso,
                                                               blur_factor=blur_factor,
                                                               bias_field_std=bias_field_std,
                                                               bias_scale=bias_scale,
                                                               noise_hr=noise_hr,
                                                               noise_lr=noise_lr,
                                                               norm_perc=norm_perc,
                                                               gamma=gamma)
    unet_input_shape = augmentation_model.output[0].get_shape().as_list()[1:]

    # prepare the segmentation model
    unet_model = unet.unet(input_shape=unet_input_shape,
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
                           size_condition_vector=size_condition_vector,
                           multi_head=multi_head,
                           input_model=augmentation_model)

    # fine-tuning with dice metric
    final_model = metrics.metrics_model(unet_model, n_output_unet_channels, 'dice', boundary_weights, reduce_type,
                                        labels_to_regions_indices=labels_to_regions_indices, mask_loss=mask_loss)
    final_model.load_weights(checkpoint, by_name=True)

    # input generator
    generator = data_loader.build_model_inputs(path_images, path_labels, path_descriptors, subjects_prob, batchsize)

    # redefine model to include all the layers to check
    list_output_tensors = []
    for name in names_output_tensors:
        list_output_tensors.append(final_model.get_layer(name).output)
    model_to_check = models.Model(inputs=final_model.inputs, outputs=list_output_tensors)

    # predict
    n = len(str(n_examples))
    for i in range(1, n_examples + 1):

        outputs = model_to_check.predict(next(generator))

        for output, name in zip(outputs, filenames_output_tensors):
            for b in range(batchsize):
                tmp_name = copy.deepcopy(name)
                if isinstance(output, np.ndarray):
                    tmp_output = np.squeeze(output[b, ...])
                else:
                    tmp_output = output
                if '_argmax' in tmp_name:
                    tmp_output = tmp_output.argmax(-1)
                    tmp_name = tmp_name.replace('_argmax', '')
                if '_convert' in tmp_name:
                    tmp_output = segm_regions[tmp_output]
                    tmp_name = tmp_name.replace('_convert', '')
                if '_save' in tmp_name:
                    path = os.path.join(result_dir, tmp_name.replace('_save', '') + '_%.{}d'.format(n) % i + '.nii.gz')
                    if batchsize > 1:
                        path = path.replace('.nii.gz', '_%s.nii.gz' % (b + 1))
                    if '_int32' in name:
                        path = path.replace('_int32', '')
                        utils.save_volume(tmp_output, np.eye(4), None, path, dtype='int32')
                    else:
                        utils.save_volume(tmp_output, np.eye(4), None, path)
                else:
                    print('{0} : {1}'.format(tmp_name, tmp_output))
