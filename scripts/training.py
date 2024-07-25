from CoNeMOS.training import training
import numpy as np

image_dir = '/path/to/image/dir'
labels_dir = '/path/to/individual/labels/dir'
model_dir = '/path/to/folder/where/all/training/data/will/be/saved'

# general params
segm_regions = np.arange(1, 6)
batchsize = 2
cropping_shape = 96
subjects_prob = '/path/to/numpy/vector/with/prob/to/samples/training/images.npy'

# conditioning params
condition_type = 'film'
label_descriptor_dir = '/data/vision/polina/scratch/bbillot/hypernet_data/training/labelled_regions/all_indiv'
n_conditioned_layers = 0  # this means all layers are conditioned
mask_loss = False

# spatial augm
flip_axis = None
scaling_bounds = .2
rotation_bounds = 180
shearing_bounds = .012
translation_bounds = 10
nonlin_std = 4.
nonlin_scale = .05

# resolution augm
randomise_res = False
max_res_iso = 6.
max_res_aniso = 6.

# intensity augm
bias_field_std = 1.
bias_scale = .03
noise_hr = 0.01
noise_lr = 0.01
norm_perc = 0.005
gamma = 0.4

# architecture params
n_levels = 4
unet_feat_count = 16
feat_multiplier = 2
activation = 'relu'
final_pred_activation = 'sigmoid'
n_conv_per_level = 2
conv_size = 3
norm_type = None
multi_head = False

# learning
lr = 1e-4
steps_per_epoch = 1000
n_epochs = 50
wl2_epochs = 2
boundary_weights = 0
checkpoint = None

training(image_dir=image_dir,
         labels_dir=labels_dir,
         model_dir=model_dir,
         segm_regions=segm_regions,
         batchsize=batchsize,
         cropping_shape=cropping_shape,
         subjects_prob=subjects_prob,
         condition_type=condition_type,
         label_descriptor_dir=label_descriptor_dir,
         n_conditioned_layers=n_conditioned_layers,
         mask_loss=mask_loss,
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
         gamma=gamma,
         n_levels=n_levels,
         unet_feat_count=unet_feat_count,
         feat_multiplier=feat_multiplier,
         activation=activation,
         final_pred_activation=final_pred_activation,
         n_conv_per_level=n_conv_per_level,
         conv_size=conv_size,
         norm_type=norm_type,
         multi_head=multi_head,
         lr=lr,
         steps_per_epoch=steps_per_epoch,
         n_epochs=n_epochs,
         wl2_epochs=wl2_epochs,
         boundary_weights=boundary_weights,
         checkpoint=checkpoint)
