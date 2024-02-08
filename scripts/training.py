from hypernet.training import training
import numpy as np

image_dir = '/path/to/image/dir'
labels_dir = '/path/to/individual/labels/dir'
model_dir = '/path/to/folder/where/all/training/data/will/be/saved'

# general params
condition_type = 'film_last'
n_conditioned_layers = 15
segm_regions = np.arange(1, 6)
label_descriptor_dir = '/data/vision/polina/scratch/bbillot/hypernet_data/training/labelled_regions/all_indiv'
subjects_prob = '/path/to/numpy/vector/with/prob/to/samples/training/images.npy'
batchsize = 1

# spatial augm
flip_axis = None
scaling_bounds = .2
rotation_bounds = 180
shearing_bounds = .012
translation_bounds = 10
nonlin_std = 4.
nonlin_scale = .05

# intensity augm
bias_field_std = .8
bias_scale = .03
noise_hr = 0.03
noise_lr = 0.02
norm_perc = 0.005
gamma = 0.4

# architecture params
n_levels = 4
unet_feat_count = 16
feat_multiplier = 1
activation = 'relu'
final_pred_activation = 'sigmoid'
n_conv_per_level = 2
conv_size = 3
norm_type = None
multi_head = False

# learning
lr = 1e-5
steps_per_epoch = 500
n_epochs = 98
wl2_epochs = 2
boundary_weights = 0
checkpoint = None

training(image_dir=image_dir,
         labels_dir=labels_dir,
         model_dir=model_dir,
         condition_type=condition_type,
         n_conditioned_layers=n_conditioned_layers,
         segm_regions=segm_regions,
         label_descriptor_dir=label_descriptor_dir,
         subjects_prob=subjects_prob,
         batchsize=batchsize,
         flip_axis=flip_axis,
         scaling_bounds=scaling_bounds,
         rotation_bounds=rotation_bounds,
         shearing_bounds=shearing_bounds,
         translation_bounds=translation_bounds,
         nonlin_std=nonlin_std,
         nonlin_scale=nonlin_scale,
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
