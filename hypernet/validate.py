# python imports
import os
import re
import logging
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.python.summary.summary_iterator import summary_iterator

# project imports
from hypernet.predict import predict

# third-party imports
from ext.lab2im import utils


def validate_training(image_dir,
                      gt_dir,
                      models_dir,
                      validation_main_dir,
                      condition_type=None,
                      n_conditioned_layers=0,
                      segm_regions=None,
                      labels_to_regions_indices=None,
                      step_eval=5,
                      min_pad=None,
                      cropping=None,
                      norm_perc=0.005,
                      flip=False,
                      sigma_smoothing=0.,
                      n_levels=4,
                      unet_feat_count=16,
                      feat_multiplier=2,
                      activation='relu',
                      final_pred_activation='sigmoid',
                      n_conv_per_level=2,
                      conv_size=3,
                      norm_type='batch',
                      multi_head=False,
                      recompute=False):

    # create result folder
    utils.mkdir(validation_main_dir)

    # loop over models
    list_models = utils.list_files(models_dir, expr=['dice', '.h5'], cond_type='and')
    list_models = [p for p in list_models if int(os.path.basename(p)[-6:-3]) % step_eval == 0]
    loop_info = utils.LoopInfo(len(list_models), 1, 'validating', True)
    for model_idx, path_model in enumerate(list_models):

        # build names and create folders
        model_val_dir = os.path.join(validation_main_dir, os.path.basename(path_model).replace('.h5', ''))
        dice_path = os.path.join(model_val_dir, 'dice.npy')
        utils.mkdir(model_val_dir)

        if (not os.path.isfile(dice_path)) | recompute:
            loop_info.update(model_idx)
            predict(path_images=image_dir,
                    path_segmentations=model_val_dir,
                    path_model=path_model,
                    condition_type=condition_type,
                    n_conditioned_layers=n_conditioned_layers,
                    segm_regions=segm_regions,
                    labels_to_regions_indices=labels_to_regions_indices,
                    min_pad=min_pad,
                    cropping=cropping,
                    norm_perc=norm_perc,
                    flip=flip,
                    sigma_smoothing=sigma_smoothing,
                    n_levels=n_levels,
                    unet_feat_count=unet_feat_count,
                    feat_multiplier=feat_multiplier,
                    activation=activation,
                    final_pred_activation=final_pred_activation,
                    n_conv_per_level=n_conv_per_level,
                    conv_size=conv_size,
                    norm_type=norm_type,
                    multi_head=multi_head,
                    gt_folder=gt_dir,
                    compute_distances=False,
                    recompute=recompute,
                    verbose=False)


def plot_validation_curves(list_validation_dirs, architecture_names=None, eval_indices=None,
                           skip_first_dice_row=True, size_max_circle=100, figsize=(11, 6), y_lim=None, fontsize=18,
                           list_linestyles=None, list_colours=None, plot_legend=False, draw_line=None):
    """This function plots the validation curves of several networks, based on the results of validate_training().
    It takes as input a list of validation folders (one for each network), each containing subfolders with dice scores
    for the corresponding validated epoch."""

    n_curves = len(list_validation_dirs)

    if eval_indices is not None:
        if isinstance(eval_indices, (np.ndarray, str)):
            if isinstance(eval_indices, str):
                eval_indices = np.load(eval_indices)
            eval_indices = np.squeeze(utils.reformat_to_n_channels_array(eval_indices, n_dims=len(eval_indices)))
            eval_indices = [eval_indices] * len(list_validation_dirs)
        elif isinstance(eval_indices, list):
            for (i, e) in enumerate(eval_indices):
                if isinstance(e, np.ndarray):
                    eval_indices[i] = np.squeeze(utils.reformat_to_n_channels_array(e, n_dims=len(e)))
                else:
                    raise TypeError('if provided as a list, eval_indices should only contain numpy arrays')
        else:
            raise TypeError('eval_indices can be a numpy array, a path to a numpy array, or a list of numpy arrays.')
    else:
        eval_indices = [None] * len(list_validation_dirs)

    # reformat model names
    if architecture_names is None:
        architecture_names = [os.path.basename(os.path.dirname(d)) for d in list_validation_dirs]
    else:
        architecture_names = utils.reformat_to_list(architecture_names, len(list_validation_dirs))

    # prepare legend labels
    if plot_legend is False:
        list_legend_labels = ['_nolegend_'] * n_curves
    elif plot_legend is True:
        list_legend_labels = architecture_names
    else:  # integer
        list_legend_labels = architecture_names
        list_legend_labels = ['_nolegend_' if i >= plot_legend else list_legend_labels[i] for i in range(n_curves)]

    # prepare linestyles
    if list_linestyles is not None:
        list_linestyles = utils.reformat_to_list(list_linestyles)
    else:
        list_linestyles = [None] * n_curves

    # prepare curve colours
    if list_colours is not None:
        list_colours = utils.reformat_to_list(list_colours)
    else:
        list_colours = [None] * n_curves

    # loop over architectures
    plt.figure(figsize=figsize)
    for idx, (net_val_dir, net_name, linestyle, colour, legend_label, eval_idx) in enumerate(zip(list_validation_dirs,
                                                                                                 architecture_names,
                                                                                                 list_linestyles,
                                                                                                 list_colours,
                                                                                                 list_legend_labels,
                                                                                                 eval_indices)):

        list_epochs_dir = utils.list_subfolders(net_val_dir, whole_path=False)

        # loop over epochs
        list_net_scores = list()
        list_epochs = list()
        for epoch_dir in list_epochs_dir:

            # build names and create folders
            path_epoch_scores = os.path.join(net_val_dir, epoch_dir, 'dice.npy')
            if os.path.isfile(path_epoch_scores):
                if eval_idx is not None:
                    list_net_scores.append(np.mean(np.abs(np.load(path_epoch_scores)[eval_idx, :])))
                else:
                    if skip_first_dice_row:
                        list_net_scores.append(np.mean(np.abs(np.load(path_epoch_scores)[1:, :])))
                    else:
                        list_net_scores.append(np.mean(np.abs(np.load(path_epoch_scores))))
                list_epochs.append(int(re.sub('[^0-9]', '', epoch_dir)))

        # plot validation scores for current architecture
        if list_net_scores:  # check that archi has been validated for at least 1 epoch
            list_net_scores = np.array(list_net_scores)
            list_epochs = np.array(list_epochs)
            list_epochs, idx = np.unique(list_epochs, return_index=True)
            list_net_scores = list_net_scores[idx]
            max_score = np.max(list_net_scores)
            epoch_max_score = list_epochs[np.argmax(list_net_scores)]
            print('\n'+net_name)
            print('epoch max score: %d' % epoch_max_score)
            print('max score: %0.3f' % max_score)
            plt.plot(list_epochs, list_net_scores, label=legend_label, linestyle=linestyle, color=colour)
            plt.scatter(epoch_max_score, max_score, s=size_max_circle, color=colour)

    # finalise plot
    plt.grid()
    if draw_line is not None:
        draw_line = utils.reformat_to_list(draw_line)
        list_linestyles = ['dotted', 'dashed', 'solid', 'dashdot'][:len(draw_line)]
        for line, linestyle in zip(draw_line, list_linestyles):
            plt.axhline(line, color='black', linestyle=linestyle)
    plt.tick_params(axis='both', labelsize=fontsize)
    plt.ylabel('Scores', fontsize=fontsize)
    plt.xlabel('Epochs', fontsize=fontsize)
    if y_lim is not None:
        plt.ylim(y_lim[0], y_lim[1] + 0.01)  # set right/left limits of plot
    plt.title('Validation curves', fontsize=fontsize)
    if plot_legend:
        plt.legend(fontsize=fontsize)
    plt.tight_layout(pad=1)
    plt.show()


def draw_learning_curve(path_tensorboard_files, architecture_names, figsize=(11, 6), fontsize=18,
                        y_lim=None, remove_legend=False):
    """This function draws the learning curve of several trainings on the same graph."""

    # reformat inputs
    path_tensorboard_files = utils.reformat_to_list(path_tensorboard_files)
    architecture_names = utils.reformat_to_list(architecture_names)
    assert len(path_tensorboard_files) == len(architecture_names), 'names and tensorboard lists should have same length'

    # loop over architectures
    plt.figure(figsize=figsize)
    for path_tensorboard_file, name in zip(path_tensorboard_files, architecture_names):

        path_tensorboard_file = utils.reformat_to_list(path_tensorboard_file)

        # extract loss at the end of all epochs
        list_losses = list()
        list_epochs = list()
        logging.getLogger('tensorflow').disabled = True
        for path in path_tensorboard_file:
            for e in summary_iterator(path):
                for v in e.summary.value:
                    if v.tag == 'loss' or v.tag == 'accuracy' or v.tag == 'epoch_loss':
                        list_losses.append(v.simple_value)
                        list_epochs.append(e.step)
        plt.plot(np.array(list_epochs), 1-np.array(list_losses), label=name, linewidth=2)

    # finalise plot
    plt.grid()
    if not remove_legend:
        plt.legend(fontsize=fontsize)
    plt.xlabel('Epochs', fontsize=fontsize)
    plt.ylabel('Scores', fontsize=fontsize)
    if y_lim is not None:
        plt.ylim(y_lim[0], y_lim[1] + 0.01)  # set right/left limits of plot
    plt.tick_params(axis='both', labelsize=fontsize)
    plt.title('Learning curves', fontsize=fontsize)
    plt.tight_layout(pad=1)
    plt.show()


def plot_validation_curves_fetal(list_validation_dirs, net_names, region_names, fontsize=10, y_lim=None, ncol=3):

    colours = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple',
               'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan',
               'k', 'b', 'g', 'r', 'c', 'm', 'gold'][:len(list_validation_dirs)]

    n_regions = len(region_names)

    fig = plt.figure(figsize=(13, 10))
    axs = fig.subplots(n_regions + 1, 1, sharex=True, sharey=True)

    for subplot_idx in range(n_regions):
        print('\n%s' % region_names[subplot_idx])

        axs[subplot_idx].grid()

        list_best_scores = list()
        list_epoch_best_scores = list()
        list_archi_best_scores = list()

        for net_idx in range(len(list_validation_dirs)):

            # initialise scores
            list_epochs = list()
            list_mean_scores = list()

            main_val_dir = list_validation_dirs[net_idx]
            if isinstance(main_val_dir, list):
                main_val_dir = main_val_dir[subplot_idx]

            if os.path.isdir(main_val_dir):
                # get scores
                for epoch_dir in utils.list_subfolders(main_val_dir, whole_path=False):
                    path_epoch_scores = os.path.join(main_val_dir, epoch_dir, 'dice.npy')
                    if os.path.isfile(path_epoch_scores):
                        list_epochs.append(int(re.sub('[^0-9]', '', epoch_dir)))
                        dice = np.load(path_epoch_scores)
                        if dice.shape[0] == 1:
                            list_mean_scores.append(np.nanmean(dice))
                        else:
                            list_mean_scores.append(np.nanmean(dice[subplot_idx]))

            # plot validation scores for current architecture
            if list_mean_scores:

                # re-order scores
                list_epochs, idx = np.unique(np.array(list_epochs), return_index=True)
                list_mean_scores = np.array(list_mean_scores)[idx]

                # print best scores
                max_score = np.max(list_mean_scores)
                epoch_max_score = list_epochs[np.argmax(list_mean_scores)]
                list_best_scores.append(max_score)
                list_epoch_best_scores.append(epoch_max_score)
                list_archi_best_scores.append(net_names[net_idx])

                # plot
                axs[subplot_idx].plot(list_epochs, list_mean_scores, label=net_names[net_idx], color=colours[net_idx])
                axs[subplot_idx].scatter(epoch_max_score, max_score, s=50, color=colours[net_idx])

        # print results
        indices = np.argsort(list_best_scores)[::-1]
        list_best_scores = np.array(list_best_scores)[indices]
        list_epoch_best_scores = np.array(list_epoch_best_scores)[indices]
        list_archi_best_scores = np.array(list_archi_best_scores)[indices]
        for i in range(len(list_best_scores)):
            print(f'{i + 1:<2}',
                  f'{list_archi_best_scores[i]:<35}',
                  'dice: %.4f' % list_best_scores[i],
                  '   epoch: %03d' % list_epoch_best_scores[i])

        axs[subplot_idx].tick_params(axis='both', labelsize=fontsize-1)
        axs[subplot_idx].set_ylabel('Dice %s' % region_names[subplot_idx], fontsize=fontsize)

    print('\naverage scores')
    axs[-1].grid()

    list_best_scores = list()
    list_epoch_best_scores = list()
    list_archi_best_scores = list()

    # last average plot
    for net_idx in range(len(list_validation_dirs)):

        main_val_dir = list_validation_dirs[net_idx]

        list_epochs = list()
        list_mean_scores = list()

        # get scores
        if isinstance(main_val_dir, list):
            epoch_dirs_per_val_dir = [utils.list_subfolders(p, whole_path=False) for p in main_val_dir]
            n_epoch_min = np.min([len(p) for p in epoch_dirs_per_val_dir])
            for epoch_idx in range(n_epoch_min):
                list_scores_epoch = list()
                for val_dir in main_val_dir:
                    path_epoch_scores = os.path.join(val_dir, epoch_dirs_per_val_dir[0][epoch_idx], 'dice.npy')
                    list_scores_epoch.append(np.nanmean(np.load(path_epoch_scores)))
                list_mean_scores.append(np.nanmean(list_scores_epoch))
            list_epochs = [int(re.sub('[^0-9]', '', subdir)) for subdir in epoch_dirs_per_val_dir[0][:n_epoch_min]]
        else:
            if os.path.isdir(main_val_dir):
                for epoch_dir in utils.list_subfolders(main_val_dir, whole_path=False):
                    path_epoch_scores = os.path.join(main_val_dir, epoch_dir, 'dice.npy')
                    if os.path.isfile(path_epoch_scores):
                        dice = np.load(path_epoch_scores)
                        list_mean_scores.append(np.nanmean(dice))
                        list_epochs.append(int(re.sub('[^0-9]', '', epoch_dir)))

        # plot validation scores for current architecture
        if list_mean_scores:
            # re-order scores
            list_epochs, idx = np.unique(np.array(list_epochs), return_index=True)
            list_mean_scores = np.array(list_mean_scores)[idx]

            # print best scores
            max_score = np.max(list_mean_scores)
            epoch_max_score = list_epochs[np.argmax(list_mean_scores)]
            list_best_scores.append(max_score)
            list_epoch_best_scores.append(epoch_max_score)
            list_archi_best_scores.append(net_names[net_idx])

            # plot
            axs[-1].plot(list_epochs, list_mean_scores, label=net_names[net_idx], color=colours[net_idx])
            axs[-1].scatter(epoch_max_score, max_score, s=50, color=colours[net_idx])

    # print results
    indices = np.argsort(list_best_scores)[::-1]
    list_best_scores = np.array(list_best_scores)[indices]
    list_epoch_best_scores = np.array(list_epoch_best_scores)[indices]
    list_archi_best_scores = np.array(list_archi_best_scores)[indices]
    for i in range(len(list_best_scores)):
        print(f'{i + 1:<2}',
              f'{list_archi_best_scores[i]:<35}',
              'dice: %.4f' % list_best_scores[i],
              '   epoch: %03d' % list_epoch_best_scores[i])

    axs[-1].tick_params(axis='both', labelsize=fontsize - 1)
    axs[-1].set_ylabel('Dice Average', fontsize=fontsize)
    axs[-1].set_ylim(y_lim)

    # finalise plot
    axs[0].legend(fontsize=fontsize - 2, ncol=ncol)
    # axs[-1].set_xlabel('Epochs', fontsize=fontsize)
    fig.tight_layout()
    plt.show()
