import sys
from argparse import ArgumentParser
from CoNeMOS.training import training
from ext.lab2im.utils import infer

parser = ArgumentParser()

# ------------------------------------------------- General parameters -------------------------------------------------

# PATHS
parser.add_argument("--image_dir", nargs='+', type=str, dest="image_dir", default=None)
parser.add_argument("--labels_dir", nargs='+', type=str, dest="labels_dir", default=None)
parser.add_argument("--model_dir", type=str, dest="model_dir", default=None)

# GENERAL
parser.add_argument("--segm_regions", type=str, dest="segm_regions", default=None)
parser.add_argument("--labels_to_regions_indices", type=str, dest="labels_to_regions_indices", default=None)
parser.add_argument("--batchsize", type=int, dest="batchsize", default=2)
parser.add_argument("--cropping_shape", type=int, dest="cropping_shape", default=None)
parser.add_argument("--data_perc", type=float, dest="data_perc", default=100)
parser.add_argument("--subjects_prob", type=str, dest="subjects_prob", default=None)

# ---------------------------------------------- Conditioning parameters -----------------------------------------------
parser.add_argument("--condition_type", type=str, dest="condition_type", default=None)
parser.add_argument("--label_descriptor_dir", type=str, dest="label_descriptor_dir", default=None)
parser.add_argument("--n_conditioned_layers", type=int, dest="n_conditioned_layers", default=0)
parser.add_argument("--loss_masking", action='store_true', dest="mask_loss")

# ---------------------------------------------- Augmentation parameters -----------------------------------------------

# SPATIAL
parser.add_argument("--flipping", dest="flip_axis", type=infer, default=None)  # None means flip in any dim
parser.add_argument("--scaling", dest="scaling_bounds", type=infer, default=0.2)
parser.add_argument("--rotation", dest="rotation_bounds", type=infer, default=180)
parser.add_argument("--shearing", dest="shearing_bounds", type=infer, default=.012)
parser.add_argument("--translation", dest="translation_bounds", type=infer, default=10)
parser.add_argument("--nonlin_std", type=float, dest="nonlin_std", default=4.)
parser.add_argument("--nonlin_scale", type=float, dest="nonlin_scale", default=.05)

# RESOLUTION
parser.add_argument("--randomise_res", action='store_true', dest="randomise_res")
parser.add_argument("--max_res_iso", type=float, dest="max_res_iso", default=6.)
parser.add_argument("--max_res_aniso", type=float, dest="max_res_aniso", default=6.)

# INTENSITY
parser.add_argument("--bias_std", type=float, dest="bias_field_std", default=1.)
parser.add_argument("--bias_scale", type=float, dest="bias_scale", default=.03)
parser.add_argument("--noise_hr", type=float, dest="noise_hr", default=.01)
parser.add_argument("--noise_lr", type=float, dest="noise_lr", default=.01)
parser.add_argument("--norm_perc", type=float, dest="norm_perc", default=.005)
parser.add_argument("--gamma", type=float, dest="gamma", default=.4)

# -------------------------------------------- UNet architecture parameters --------------------------------------------
parser.add_argument("--n_levels", type=int, dest="n_levels", default=4)
parser.add_argument("--unet_feat", type=int, dest="unet_feat_count", default=16)
parser.add_argument("--feat_mult", type=int, dest="feat_multiplier", default=2)
parser.add_argument("--activation", type=str, dest="activation", default='relu')
parser.add_argument("--final_pred_activation", type=str, dest="final_pred_activation", default='sigmoid')
parser.add_argument("--n_conv_per_level", type=int, dest="n_conv_per_level", default=2)
parser.add_argument("--conv_size", type=int, dest="conv_size", default=3)
parser.add_argument("--norm_type", type=str, dest="norm_type", default=None)
parser.add_argument("--multi_head", dest="multi_head", type=infer, default=False)

# ------------------------------------------------- Training parameters ------------------------------------------------
parser.add_argument("--lr", type=float, dest="lr", default=1e-4)
parser.add_argument("--steps_per_epoch", type=int, dest="steps_per_epoch", default=1000)
parser.add_argument("--n_epochs", type=int, dest="n_epochs", default=500)
parser.add_argument("--wl2_epochs", type=int, dest="wl2_epochs", default=5)
parser.add_argument("--boundary_weights", type=float, dest="boundary_weights", default=0)
parser.add_argument("--checkpoint", type=str, dest="checkpoint", default=None)


# PRINT ALL ARGUMENTS
print('\nScript name:',  sys.argv[0])
print('\nScript arguments:')
args = parser.parse_args()
for arg in vars(args):
    print(arg, getattr(args, arg))
print('')
training(**vars(args))
