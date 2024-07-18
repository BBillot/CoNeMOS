import os
import copy
import numpy as np
from keras import models

# third-party imports
from ext.lab2im import utils


from CoNeMOS import data_loader, augmentation


def check_augmentation(image_dir,
                       labels_dir,
                       result_dir,
                       n_examples,
                       names_output_tensors,
                       filenames_output_tensors,
                       condition_type=None,
                       segm_regions=None,
                       label_descriptor_dir=None,
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
                       n_levels=4):

    # prepare data files
    path_images, path_labels, path_descriptors = data_loader.get_paths(image_dir, labels_dir, label_descriptor_dir)

    # get label lists
    segm_regions = np.load(segm_regions) if segm_regions is not None else np.ones(1, dtype='int32')
    n_regions = len(segm_regions)
    assert np.array_equal(segm_regions, np.arange(1, n_regions + 1)), 'labels should be increasing without skips'
    size_condition_vector = n_regions if condition_type is not None else 0

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

    # input generator
    input_generator = data_loader.build_model_inputs(path_images, path_labels, path_descriptors, batchsize)

    # redefine model to include all the layers to check
    list_output_tensors = []
    for name in names_output_tensors:
        list_output_tensors.append(augmentation_model.get_layer(name).output)
    model_to_check = models.Model(inputs=augmentation_model.inputs, outputs=list_output_tensors)

    # predict
    n = len(str(n_examples))
    for i in range(1, n_examples + 1):

        outputs = model_to_check.predict(next(input_generator))

        for output, name in zip(outputs, filenames_output_tensors):
            for b in range(batchsize):
                tmp_name = copy.deepcopy(name)
                tmp_output = np.squeeze(output[b, ...])
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
