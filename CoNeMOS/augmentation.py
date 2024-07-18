import numpy as np
import tensorflow as tf
from keras import models
import keras.layers as KL

# lab2im
from ext.lab2im import layers, utils
from ext.lab2im import edit_tensors as l2i_et


def build_augmentation_model(im_shape,
                             atlas_res,
                             condition_type,
                             size_condition_vector=0,
                             output_shape=None,
                             output_div_by_n=None,
                             flip_axis=None,  # None=flipping in any axis, set to False to disable
                             scaling_bounds=0.15,
                             rotation_bounds=15,
                             shearing_bounds=0.012,
                             translation_bounds=False,
                             nonlin_std=3.,
                             nonlin_scale=.0625,
                             randomise_res=False,
                             max_res_iso=4.,
                             max_res_aniso=8.,
                             bias_field_std=.5,
                             bias_scale=.025,
                             noise_hr=0.08,
                             noise_lr=0.08,
                             norm_perc=0.02,
                             gamma=0.4):

    # reformat resolutions and get shapes
    im_shape = utils.reformat_to_list(im_shape)
    n_dims, _ = utils.get_dims(im_shape)
    target_res = atlas_res

    # define model inputs
    image_input = KL.Input(shape=im_shape + [1], name='image_input', dtype='float32')
    labels_input = KL.Input(shape=im_shape + [1], name='labels_input', dtype='int32')
    model_inputs = [image_input, labels_input]
    if condition_type == 'channel':
        cond_input = KL.Input(shape=[size_condition_vector], name='cond_input', dtype='float32')
        model_inputs.append(cond_input)
    else:
        cond_input = None

    # deform labels
    labels, image = layers.RandomSpatialDeformation(scaling_bounds=scaling_bounds,
                                                    rotation_bounds=rotation_bounds,
                                                    shearing_bounds=shearing_bounds,
                                                    translation_bounds=translation_bounds,
                                                    nonlin_std=nonlin_std,
                                                    nonlin_scale=nonlin_scale,
                                                    prob=0.95,
                                                    inter_method=['nearest', 'linear'])([labels_input, image_input])

    # cropping
    output_shape = get_shapes(im_shape, output_shape, output_div_by_n)
    if output_shape != im_shape:
        labels, image = layers.RandomCrop(output_shape)([labels, image])

    # flipping
    if flip_axis is not False:
        labels, image = layers.RandomFlip(axis=flip_axis)([labels, image])

    # apply bias field
    if bias_field_std > 0:
        image = layers.BiasFieldCorruption(bias_field_std, bias_scale, False, prob=0.95)(image)

    # first normalisation
    image = layers.IntensityAugmentation(normalise=True, norm_perc=norm_perc)(image)

    # if necessary, loop over channels to 1) blur, 2) downsample to simulated LR, and 3) upsample to target
    if randomise_res:

        # sample resolution
        max_res_iso = np.array(utils.reformat_to_list(max_res_iso, length=n_dims, dtype='float'))
        max_res_aniso = np.array(utils.reformat_to_list(max_res_aniso, length=n_dims, dtype='float'))
        max_res = np.maximum(max_res_iso, max_res_aniso)
        res, blur_res = layers.SampleResolution(atlas_res, max_res_iso, max_res_aniso, prob_iso=.4, prob_min=.2)(image)

        # blur
        sigma = l2i_et.blurring_sigma_for_downsampling(atlas_res, res, thickness=blur_res)
        image = layers.DynamicGaussianBlur(0.75 * max_res / np.array(atlas_res), 1.03)([image, sigma])

        # downsample
        image = layers.MimicAcquisition(atlas_res, target_res, output_shape, noise_std=noise_lr)([image, res])

    # intensity augmentation
    image = layers.IntensityAugmentation(noise_std=noise_hr, normalise=True, gamma_std=gamma, prob_noise=0.9)(image)

    # conditional label
    if condition_type == 'channel':
        cond = KL.Lambda(lambda x: l2i_et.expand_dims(x, axis=[1] * n_dims), name='expand_dims_cond')(cond_input)
        cond = KL.Lambda(lambda x: tf.tile(x, [1, *output_shape, 1]), name='tile_cond')(cond)
        image = KL.Lambda(lambda x: tf.concat([x[0], tf.cast(x[1], x[0].dtype)], -1), name='cat_im_cond')([image, cond])

    # dummy layer enables to keep the labels when plugging this model to other models
    labels = KL.Lambda(lambda x: tf.cast(x, dtype='int32'), name='labels_out')(labels)
    image = KL.Lambda(lambda x: x[0], name='image_out')([image, labels])

    # build model
    brain_model = models.Model(inputs=model_inputs, outputs=[image, labels])
    return brain_model


def get_shapes(im_shape, output_shape, output_div_by_n):

    # reformat output shape to be smaller or equal to im_shape and divisible by output_div_by_n
    if output_shape is not None:
        n_dims = len(im_shape)
        output_shape = utils.reformat_to_list(output_shape, length=n_dims, dtype='int')
        output_shape = [min(im_shape[i], output_shape[i]) for i in range(n_dims)]
        if output_div_by_n is not None:
            output_shape = [utils.find_closest_number_divisible_by_m(s, output_div_by_n) for s in output_shape]

    # make sure output_shape=im_shape is divisible by output_div_by_n
    else:
        output_shape = im_shape
        if output_div_by_n is not None:
            output_shape = [utils.find_closest_number_divisible_by_m(s, output_div_by_n) for s in output_shape]

    return output_shape
