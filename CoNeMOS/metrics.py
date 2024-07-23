import tensorflow as tf
import keras.layers as KL
from keras.models import Model

from ext.lab2im import layers


def metrics_model(input_model,
                  n_channels,
                  loss_type='dice',
                  boundary_weights=100,
                  labels_to_regions_indices=None,
                  mask_loss=False):

    # get prediction
    labels_pred = input_model.outputs[0]

    # get GT and convert it to one hot
    labels_gt = input_model.get_layer('labels_out').output
    if n_channels > 1:
        if labels_to_regions_indices is not None:
            labels_gt = layers.LabelsToRegions(labels_to_regions_indices, name='gt_regions')(labels_gt)
        else:
            labels_gt = KL.CategoryEncoding(num_tokens=n_channels, output_mode='one_hot', name='gt_regions')(labels_gt)

    # add input to mask the loss to available regions only
    if mask_loss:
        mask_input = KL.Input(shape=[n_channels], dtype='float32', name='mask_input')
        model_inputs = input_model.inputs + [mask_input]
        loss_layer_inputs = [labels_gt, labels_pred, mask_input]
    else:
        model_inputs = input_model.inputs
        loss_layer_inputs = [labels_gt, labels_pred]

    # compute loss
    if loss_type == 'dice':
        loss = layers.DiceLoss(boundary_weights=boundary_weights,
                               make_probabilistic=False, name='loss')(loss_layer_inputs)
    elif loss_type == 'wl2':
        loss = layers.WeightedL2Loss(name='loss')(loss_layer_inputs)
    else:
        raise Exception('metrics should be "dice" "wl2", got {}'.format(loss_type))

    # create the model and return
    model = Model(inputs=model_inputs, outputs=loss)
    return model


class IdentityLoss(object):
    """Very simple loss, as the computation of the loss as been directly implemented in the model."""
    def __init__(self, keepdims=True):
        self.keepdims = keepdims

    def loss(self, y_true, y_predicted):
        """Because the metrics is already calculated in the model, we simply return y_predicted.
           We still need to put y_true in the inputs, as it's expected by keras."""
        loss = y_predicted

        tf.debugging.check_numerics(loss, 'Loss not finite')
        return loss
