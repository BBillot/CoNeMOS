# python imports
import os
import numpy as np
from keras import models
import keras.callbacks as KC
from keras.optimizers import Adam

# third-party imports
from ext.lab2im import utils

from CoNeMOS import data_loader, augmentation, unet, metrics


def training(image_dir,
             labels_dir,
             model_dir,
             condition_type=None,
             n_conditioned_layers=0,
             segm_regions=None,
             labels_to_regions_indices=None,
             label_descriptor_dir=None,
             subjects_prob=None,
             data_perc=100,
             mask_loss=False,
             batchsize=1,
             cropping_shape=None,
             flip_axis=None,
             scaling_bounds=.2,
             rotation_bounds=180,
             shearing_bounds=.012,
             translation_bounds=10,
             nonlin_std=4.,
             nonlin_scale=.05,
             randomise_res=False,
             max_res_iso=6.,
             max_res_aniso=6.,
             bias_field_std=1.,
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
             lr=1e-4,
             steps_per_epoch=1000,
             n_epochs=500,
             wl2_epochs=5,
             boundary_weights=100,
             reduce_type='mean',
             checkpoint=None):
    """
    :param image_dir:
    :param labels_dir:
    :param model_dir:
    :param segm_regions: sorted numpy array with all the segmentation regions to segment.
    Defaults to None, where we assume binary label maps (but we still need it when using conditioning, to know the size
    of the conditioning vector). Should not include background for partially annotated datasets.
    :param label_descriptor_dir: path (or list of paths) to folders containing label descriptors (numpy arrays that say
    which structure is segmented in each training label map). Defaults to None
    :param condition_type: whether to use FiLM conditioning (condition_type='film'), input channel conditioning
    ('channel'), or no contitioning (None). Additionally we xcan also condition on the image (add _image) and condition
    the very last likelihood layer (add _last)
    :param n_conditioned_layers: number of layers to condition, starting from the end of the network. This only works if
    film is in condition_type. Leave to zero to condition all layers.
    :param labels_to_regions_indices: to use to convert label-based groud truth segmentations into hierarchical region-
    based segmentations. 2D matrix specifying how to split labels into regions. We assume that each region is the sum of
    one or several labels. This should be of size n_region * n_label. Needs to be triangular inferior, so order the
    labels in segmentation_labels consequently.
    :param subjects_prob: numpy array as long as the number of training subjects with relative probability of being
    sampled during training
    :param data_perc: percentage of the available training data to use. default is 100.
    :param mask_loss:
    :param batchsize:
    :param output_shape:
    :param flip_axis: None=flipping in any axis, set to False to disable
    :param scaling_bounds:
    :param rotation_bounds:
    :param shearing_bounds:
    :param translation_bounds:
    :param nonlin_std:
    :param nonlin_scale:
    :param randomise_res:
    :param downsample:
    :param max_res_iso:
    :param max_res_aniso:
    :param blur_factor:
    :param bias_field_std:
    :param bias_scale:
    :param noise_hr:
    :param noise_lr:
    :param norm_perc:
    :param gamma:
    :param n_levels:
    :param unet_feat_count:
    :param feat_multiplier:
    :param activation: can be 'linear' (i.e. identity), 'softmax', 'sigmoid', 'relu', etc.
    :param final_pred_activation: can be 'linear' (i.e. identity), 'softmax', 'sigmoid', 'relu', etc.
    :param n_conv_per_level:
    :param conv_size:
    :param norm_type:
    :param multi_head: can be 'decoder' (each head is an entire decoder), 'layer' (we use an additional conv layer
    for each label to segment). Default is False, where nu multi_head is used.
    :param lr:
    :param steps_per_epoch:
    :param n_epochs:
    :param wl2_epochs:
    :param boundary_weights:
    :param reduce_type: function to compute the final loss scores across label values. Can be 'mean' or 'sum'.
    :param checkpoint:
    """

    if condition_type not in [None, 'channel', 'film', 'film_image', 'film_last', 'film_image_last']:
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
                                                               output_shape=cropping_shape,
                                                               output_div_by_n=2 ** n_levels,
                                                               flip_axis=flip_axis,
                                                               scaling_bounds=scaling_bounds,
                                                               rotation_bounds=rotation_bounds,
                                                               shearing_bounds=shearing_bounds,
                                                               translation_bounds=translation_bounds,
                                                               nonlin_std=nonlin_std,
                                                               nonlin_scale=nonlin_scale,
                                                               randomise_res=randomise_res,
                                                               max_res_iso=max_res_iso,
                                                               max_res_aniso=max_res_aniso,
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

    # input generator
    generator = data_loader.build_model_inputs(path_images, path_labels, path_descriptors, subjects_prob, batchsize)
    input_generator = utils.build_training_generator(generator, batchsize)

    # pre-training with weighted L2, input is fit to the softmax rather than the probabilities
    if (wl2_epochs > 0) & (metrics != 'distance_maps'):
        wl2_model = models.Model(unet_model.inputs, [unet_model.get_layer('unet_likelihood').output])
        wl2_model = metrics.metrics_model(wl2_model, n_output_unet_channels, 'wl2', boundary_weights,
                                          labels_to_regions_indices=labels_to_regions_indices, mask_loss=mask_loss)
        train_model(wl2_model, input_generator, lr, wl2_epochs, steps_per_epoch, model_dir, 'wl2', checkpoint)
        checkpoint = os.path.join(model_dir, 'models', 'wl2_%03d.h5' % wl2_epochs)

    # fine-tuning with dice metric
    final_model = metrics.metrics_model(unet_model, n_output_unet_channels, 'dice', boundary_weights, reduce_type,
                                        labels_to_regions_indices=labels_to_regions_indices, mask_loss=mask_loss)
    train_model(final_model, input_generator, lr, n_epochs, steps_per_epoch, model_dir, 'dice', checkpoint)


def train_model(model,
                generator,
                learning_rate,
                n_epochs,
                n_steps,
                model_dir,
                metric_type,
                path_checkpoint=None):

    # prepare model and log folders
    utils.mkdir(model_dir)
    models_dir = os.path.join(model_dir, 'models')
    log_dir = os.path.join(model_dir, 'logs')
    utils.mkdir(models_dir)
    utils.mkdir(log_dir)

    # model saving callback
    save_file_name = os.path.join(models_dir, '%s_{epoch:03d}.h5' % metric_type)
    callbacks = [KC.ModelCheckpoint(save_file_name, verbose=1)]

    # TensorBoard callback
    if metric_type == 'dice':
        callbacks.append(KC.TensorBoard(log_dir=log_dir, histogram_freq=0, write_graph=True, write_images=False))

    compile_model = True
    init_epoch = 0
    if path_checkpoint is not None:
        if metric_type in os.path.basename(path_checkpoint):
            init_epoch = int(os.path.basename(path_checkpoint).split(metric_type)[1][1:-3])
        model.load_weights(path_checkpoint, by_name=True)

    # compile
    if compile_model:
        model.compile(optimizer=Adam(learning_rate=learning_rate), loss=metrics.IdentityLoss().loss)

    # fit
    model.fit(generator, epochs=n_epochs, steps_per_epoch=n_steps, callbacks=callbacks, initial_epoch=init_epoch)
