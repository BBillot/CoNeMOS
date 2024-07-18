import numpy as np
from keras import Model
import keras.layers as KL

from ext.lab2im import utils
from CoNeMOS.film import FiLM


def unet(input_shape,
         n_output_channels,  # 1 for conditioning, n_regions otherwise
         n_levels=4,
         n_features_init=16,
         feat_mult=2,
         activation='elu',
         final_pred_activation='sigmoid',
         n_conv_per_level=2,
         conv_size=3,
         norm_type='batch',
         condition_type=None,
         size_condition_vector=0,  # 1-hot encoding of the labels, different from n_output_channels
         n_conditioned_layers=0,  # only use if condition_type == 'film' or 'film_image'. 0 means all layers are filmed
         multi_head=False,
         input_model=None):

    # prepare layers
    n_dims = len(input_shape) - 1
    conv_layer = getattr(KL, 'Conv%dD' % n_dims)
    conv_args = {'kernel_size': conv_size, 'activation': activation, 'padding': 'same', 'data_format': 'channels_last'}
    if condition_type in ['film', 'film_image', 'film_last', 'film_image_last']:
        conv_args['use_bias'] = False

    # prepare conditioning
    n_conv_layers = n_conv_per_level * (2 * n_levels - 1)
    if condition_type in ['film_last', 'film_image_last']:  # if film in name the last layer is automatically filmed
        n_conv_layers += 1
    if condition_type not in ['film', 'film_image', 'film_last', 'film_image_last'] and n_conditioned_layers != 0:
        raise ValueError('only specify n_conditioned_layers when condition_type is film or film_image, '
                         'had {} and {}'.format(n_conditioned_layers, condition_type))
    if condition_type in ['film', 'film_image', 'film_last', 'film_image_last'] and n_conditioned_layers == 0:
        n_conditioned_layers = n_conv_layers

    # get input layer
    if input_model is None:
        input_tensor = KL.Input(shape=input_shape, name='unet_input')
        x = input_tensor
    else:
        input_tensor = input_model.inputs
        x = input_model.outputs
        if isinstance(x, list):
            x = x[0]

    # get MLP for conditioning
    if condition_type in ['film', 'film_image', 'film_last', 'film_image_last']:
        condition = KL.Input(shape=[size_condition_vector], name='cond_input', dtype='float32')
        input_tensor = input_tensor + [condition] if isinstance(input_tensor, list) else [input_tensor, condition]
        for i in range(4):
            condition = KL.Dense(64, name='dense_%s' % (i + 1))(condition)
            condition = KL.LeakyReLU(0.2)(condition)
    else:
        condition = None

    # down-arm
    x = encoder(x, n_dims, n_levels, n_features_init, feat_mult, n_conv_per_level, norm_type, conv_args,
                condition, n_conditioned_layers, n_conv_layers)

    # build intermediate model
    encoder_model = Model(inputs=input_tensor, outputs=x)

    # condition on image
    x_pooled = KL.GlobalAveragePooling3D()(x)
    if condition_type in ['film_image', 'film_image_last']:
        condition = KL.Concatenate(axis=-1)([condition, x_pooled])

    # up-arm
    if multi_head == 'decoder':
        list_binary_segm = list()
        for labels in range(n_output_channels):
            tmp_x = decoder(x, n_dims, n_levels, n_features_init, feat_mult, n_conv_per_level, norm_type, conv_args,
                            encoder_model, None, 0, n_conv_layers, decoder_idx=labels + 1)
            list_binary_segm.append(conv_layer(1, 1, activation=None, name='unet_head%s_conv1x1' % (labels + 1))(tmp_x))
        x = KL.Concatenate(axis=-1, name='unet_likelihood')(list_binary_segm)

    elif multi_head == 'layer':
        x = decoder(x, n_dims, n_levels, n_features_init, feat_mult, n_conv_per_level, norm_type, conv_args,
                    encoder_model, None, 0, n_conv_layers)
        list_binary_segm = list()
        for labels in range(n_output_channels):
            tmp_x = conv_block(x, n_dims, 1, 1, conv_args, norm_type, 'net_head%s' % (labels + 1), False, None)
            list_binary_segm.append(conv_layer(1, 1, activation=None, name='unet_head%s_conv1x1' % (labels + 1))(tmp_x))
        x = KL.Concatenate(axis=-1, name='unet_likelihood')(list_binary_segm)

    else:  # should go here if we use conditioning
        x = decoder(x, n_dims, n_levels, n_features_init, feat_mult, n_conv_per_level, norm_type, conv_args,
                    encoder_model, condition, n_conditioned_layers, n_conv_layers)
        if condition_type in ['film_last', 'film_image_last']:  # if film in name the last layer is automatically filmed
            x = conv_layer(n_output_channels, 1, activation=None, use_bias=False, name='unet_likelihood')(x)
            x = FiLM(n_dims=n_dims, name='unet_likelihood_film')([x, condition])
        else:
            x = conv_layer(n_output_channels, 1, activation=None, name='unet_likelihood')(x)
    x = KL.Activation(final_pred_activation, name='unet_%s' % final_pred_activation)(x)

    return Model(inputs=input_tensor, outputs=x, name='unet')


def encoder(x, n_dims, n_levels, n_features_init, feat_mult, n_conv_per_level, norm_type, conv_args,
            condition, n_conditioned_layers, n_conv_layers):

    maxpool = getattr(KL, 'MaxPooling%dD' % n_dims)
    idx_start_conditioning = n_conv_layers - n_conditioned_layers

    # down-arm
    for level in range(n_levels):

        # conditioning
        if n_conditioned_layers > 0:
            use_condition = [(level * n_conv_per_level + i) >= idx_start_conditioning for i in range(n_conv_per_level)]
        else:
            use_condition = [False] * n_conv_per_level

        # convolution block
        base_name = 'unet_down%d' % level
        n_feat = np.round(n_features_init * feat_mult ** level).astype(int)
        x = conv_block(x, n_dims, n_conv_per_level, n_feat, conv_args, norm_type, base_name, use_condition, condition)

        # max pool if we're not at the last level
        if level < (n_levels - 1):
            x = maxpool(padding='same', name='unet_down%d_maxpool' % level)(x)

    return x


def decoder(x, n_dims, n_levels, n_features_init, feat_mult, n_conv_per_level, norm_type, conv_args, encoder_model,
            condition, n_conditioned_layers, n_conv_layers, decoder_idx=None):

    upsample = getattr(KL, 'UpSampling%dD' % n_dims)
    idx_start_conditioning = (n_conv_layers - n_conditioned_layers) - n_levels * n_conv_per_level

    for level in range(n_levels - 1):

        # upsample matching the max pooling layers size
        base_name = 'unet_up%d' % (n_levels + level)
        if decoder_idx is not None:
            base_name = base_name.replace('unet', 'unet_head%s' % decoder_idx)
        x = upsample(size=(2,) * n_dims, name=base_name + '_upsample')(x)

        # concatenate with layer from down-arm
        cat_tensor = encoder_model.get_layer('unet_down%d_conv%d' % (n_levels - 2 - level, n_conv_per_level - 1)).output
        x = KL.concatenate([cat_tensor, x], axis=n_dims + 1, name=base_name + '_concat')

        # conditioning
        if n_conditioned_layers > 0:
            use_condition = [(level * n_conv_per_level + i) >= idx_start_conditioning for i in range(n_conv_per_level)]
        else:
            use_condition = [False] * n_conv_per_level

        # convolution block
        n_feat = np.round(n_features_init * feat_mult ** (n_levels - 2 - level)).astype(int)
        x = conv_block(x, n_dims, n_conv_per_level, n_feat, conv_args, norm_type, base_name, use_condition, condition)

    return x


def conv_block(x, n_dims, n_conv_per_level, n_feat, conv_args, norm_type, base_name, use_conditioning, condition):

    conv_layer = getattr(KL, 'Conv%dD' % n_dims)
    use_conditioning = utils.reformat_to_list(use_conditioning, length=n_conv_per_level)

    for conv in range(n_conv_per_level):

        # convolution + non-linearity
        x = conv_layer(n_feat, **conv_args, name=base_name + '_conv%d' % (conv + 1))(x)

        # normalisation
        if norm_type == 'batch':
            x = KL.BatchNormalization(axis=-1, name=base_name + '_bn%s' % (conv + 1))(x)
        elif norm_type == 'instance':
            x = KL.GroupNormalization(n_feat, axis=-1, name=base_name + '_in%s' % (conv + 1))(x)
        elif norm_type is not None:
            ValueError("norm_type should be either 'batch', 'instance' or None")

        # feature-wise transformation (FiLM)
        if use_conditioning[conv]:
            x = FiLM(n_dims=n_dims, name=base_name + '_film%s' % (conv + 1))([x, condition])

    return x
