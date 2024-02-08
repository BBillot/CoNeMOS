# python imports
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# third-party imports
from ext.lab2im import utils
# from statannot.statannot_best_box_only import add_stat_annotation


def draw_boxplots(list_score_paths,
                  list_datasets,
                  list_methods,
                  names=None,
                  methods_names=None,
                  eval_indices=None,
                  skip_first_row=False,
                  palette=None,
                  order=None,
                  figsize=None,
                  remove_legend=True,
                  fontsize=23,
                  fontsize_legend=22,
                  fontsize_xticks=None,
                  legend_loc=None,
                  ncol_legend=1,
                  y_label=None,
                  threshold=None,
                  boxplot_width=0.6,
                  boxplot_linewidth=1.5,
                  outlier_size=2,
                  av_plot_ylim=None,
                  av_plot_step_yticks=0.1,
                  av_yticks=None,
                  use_tick_formatter=True,
                  av_plot_title='Average Dice across structures',
                  av_plot_rotation_x_ticks=0,
                  draw_subplots=False,
                  show_average_on_subplots=False,
                  subplot_ylim=None,
                  subplot_step_yticks=0.1,
                  list_subplot_titles=None,
                  path_av_figure=None,
                  grey_background=False,
                  add_stat_annots=False,
                  bonferroni_correction=True,
                  text_annotations=None,
                  line_offset_to_box=0.03,
                  marker_spacing=0.2,
                  dot_size=70):
    """
    This function draws boxplots for scores obtained by different methods on a set of different datasets.
    The first plot displays boxplots for average scores of all methods on all datasets.
    The other boxplots display structure-wise scores for all methods for a given dataset.
    This function collects scores from numpy arrays, one per method and per dataset.
    Numpy arrays are assumed to be organised such that rows=structures, and columns=subjects.
    Scores of different structures (e.g. contraletral structures) can be plotted together in the same box by
    giving them the same name.
    We can select a subset of structures to include in the subplots by specifying rows indices.
    :param list_score_paths: list of paths of all numpy arrays, each having scores for a single method on a single
    dataset. Each numpy array is assumed to be organised such that rows=structures, and columns=subjects.
    All arrays must have the same row ordering, i.e. rows must correspond to the same structures.
    :param list_datasets: list of all datasets.
    :param list_methods: list of all methods.
    :param names: list of structure names. All structures with the same name will be plotted together (useful
    for contralateral structures). Must have the same length and ordering as eval_indices, if provided.
    Can be a sequence, a numpy 1d array, or the path to a numpy array. (only used if draw_subplots is True).
    :param eval_indices: indices of the structure to include in the boxplots. These indices correspond the structure
    rows in the provided numpy arrays. Must have the same length and ordering as names.
    Can be a sequence, a numpy 1d array, or the path to a numpy array.
    :param palette: (optional) dictionary where keys are methods names and items are rgb colours in tuples of length 3.
    :param order: (optional) Order in which to display methods boxplots.
    :param figsize: (optional) size of each plot. Default is (11, 6).
    :param remove_legend: (optional) Remove legend from average plot. The legend does not appear in any other plot.
    :param legend_loc: (optional) where to put the legend, if remove_legend is False. See matplotlib.pyplot/legend.
    :param fontsize: (optional) Fontsize of the ticks and axes names
    :param fontsize_legend: (optional) Fontsize of the plots titles.
    :param y_label: (optional) Name of y-axis in all plots.
    :param boxplot_width: (optional) width of all boxes.
    :param boxplot_linewidth: (optional) linewidth of all boxes.
    :param outlier_size: (optional) size of the outliers markers.
    :param av_plot_ylim: (optional) range of y-axis in the average plot. Default is [0.5, 0.92]
    :param av_plot_step_yticks: (optional) spacing between y-axis ticks for the average plot.
    :param av_plot_title: (optional) title of the average plot
    :param av_plot_rotation_x_ticks: (optional) angle by which to rotate x_labels in average plot. Default is 0.
    :param draw_subplots: (optional) whether to draw subplots i.e. plots for each dataset with structure-wise boxplots
    :param show_average_on_subplots: (optional) whether to display boxes for average scores on each single-dataset plot.
    :param subplot_ylim: (optional) range of y-axis in single-dataset plots. Default is [0.4, 1.]
    :param subplot_step_yticks: (optional) spacing between y-axis ticks in the single dataset plots.
    :param list_subplot_titles: (optional) list of all single-dataset plots.
    """

    # get sequence default values
    if figsize is None:
        figsize = (11, 6)
    if av_plot_ylim is None:
        av_plot_ylim = [0.5, 0.92]
    if subplot_ylim is None:
        subplot_ylim = [0.4, 1.]

    # reformat indices and method names
    eval_indices = utils.reformat_to_list(eval_indices, load_as_numpy=True)
    methods_names = utils.reformat_to_list(methods_names) if methods_names is not None else [None] * len(list_methods)

    # get data for boxplot of average results
    df = pd.DataFrame(columns=['dice scores', 'datatype', 'method'])
    set_datatype = set()
    for path_dice, datatype, method, method_name in zip(list_score_paths, list_datasets, list_methods, methods_names):

        # check if dice exists
        pass_iter = True
        dice = None
        if isinstance(path_dice, str):
            pass_iter = False
            if os.path.isfile(path_dice):
                dice = path_dice
            else:
                dice = np.array([-1])
        elif isinstance(path_dice, np.ndarray):
            dice = path_dice
            pass_iter = False
        if pass_iter:
            continue

        # use method name
        if method_name is None:
            method_name = method

        # load dice
        dice = utils.load_array_if_path(dice)
        if len(dice.shape) > 1:
            if eval_indices is not None:
                dice = dice[eval_indices, :]
            elif skip_first_row:
                dice = dice[1:, :]
            dice = np.mean(dice, axis=0)
        if threshold is not None:
            dice = dice[dice > threshold]
        df = df.append(pd.DataFrame({'dice scores': dice,
                                     'datatype': [datatype] * dice.shape[0],
                                     'method': [method_name] * dice.shape[0]}), ignore_index=True, sort=False)
        set_datatype.add(datatype)

    # draw boxplot of average results
    plt.figure(figsize=figsize)
    if grey_background:
        sns.set_style('darkgrid')
        sns.set_context(rc={"grid.linewidth": 1.5})
    else:
        sns.set_style('whitegrid')
    ax = sns.boxplot(data=df, x='datatype', y='dice scores', hue='method', linewidth=boxplot_linewidth,
                     width=boxplot_width, palette=palette, order=order, fliersize=outlier_size)

    # grey background
    if grey_background:
        ax.set_axisbelow(True)
        [ax.spines[side].set_visible(False) for side in ax.spines]
        ax.set_facecolor('#EBEBEB')
        ax.grid(which='major', color='white', linewidth=1.5)

    # y axis
    if y_label is not None:
        ax.set_ylabel(y_label, fontsize=fontsize)  # set fontsize of y-axis title
    else:
        ax.set_ylabel('', fontsize=2)
    ax.set_ylim(av_plot_ylim[0], av_plot_ylim[1] + 0.01)  # set right/left limits of plot

    # x axis
    ax.set_xlabel('', fontsize=fontsize)  # give no name to x-axis
    ax.set_xlim(-boxplot_width / 2 - .05, len(set_datatype) - (1 - boxplot_width / 2 - .05))  # set right/left limits

    # if add_stat_annots:
    #     add_stat_annotation(ax,
    #                         data=df,
    #                         x='datatype',
    #                         y='dice scores',
    #                         hue='method',
    #                         order=order,
    #                         box_width=boxplot_width,
    #                         box_linewidth=boxplot_linewidth,
    #                         outlier_size=outlier_size,
    #                         test='t-test_welch',
    #                         bonferroni_correction=bonferroni_correction,
    #                         text=text_annotations,
    #                         dot_size=dot_size,
    #                         line_offset_to_box=line_offset_to_box,
    #                         marker_spacing=marker_spacing)

    # general tick parameters and plot x ticks
    if fontsize_xticks is not None:
        ax.tick_params(axis='x', labelsize=fontsize_xticks)  # adjust tick size
        ax.tick_params(axis='y', labelsize=fontsize)  # adjust tick size
    else:
        ax.tick_params(axis='both', labelsize=fontsize)  # adjust tick size
    plt.xticks(np.arange(0, len(set_datatype), step=1), rotation=av_plot_rotation_x_ticks)

    # plot y ticks
    if av_yticks is not None:
        plt.yticks(av_yticks)
        tick_nb_elements = len(max(av_yticks.astype(np.str), key=len))
    else:
        plt.yticks(np.arange(av_plot_ylim[0], av_plot_ylim[1] + .01, step=av_plot_step_yticks))
        tick_nb_elements = len(str(av_plot_step_yticks))

    def format_func(value, tick_number):
        value = float(np.around(value, 6))
        if value == 0:
            return '0'
        elif value < 1:
            return ('%.{}f'.format(tick_nb_elements) % value)[1:tick_nb_elements]  # str(value)[1:tick_nb_elements]
        else:
            return str(value)
    if use_tick_formatter:
        ax.yaxis.set_major_formatter(plt.FuncFormatter(format_func))

    # plot legend
    if remove_legend:
        ax.get_legend().remove()
    else:
        plt.legend(
            fontsize=fontsize_legend,
            loc=legend_loc if legend_loc is not None else 'best',
            handletextpad=0.2,  # space between colour boxes and text
            handlelength=1.,     # length of colour boxes
            borderpad=0.25,      # space between text and edge of the legend box
            labelspacing=0.18,  # vertical spacing between entries
            ncol=ncol_legend,
            columnspacing=0.7,
            facecolor='white' if grey_background else 'inherit',
            framealpha=1 if grey_background else 0.9,  # transparency
        )

    # title
    if av_plot_title is not None:
        plt.title(av_plot_title, fontsize=fontsize)
    plt.tight_layout()

    # save figure
    if path_av_figure is not None:
        utils.mkdir(os.path.dirname(path_av_figure))
        plt.savefig(path_av_figure)
    plt.show()

    # draw boxplot for every modality of every dataset
    if draw_subplots:

        # reformat structure names
        if names is not None:
            names = utils.reformat_to_list(names, load_as_numpy=True)
            unique_names = list()
            for name in names:
                if name not in unique_names:
                    unique_names.append(name)
            if show_average_on_subplots:
                unique_names = unique_names + ['av']
        else:
            unique_names = None
        if list_subplot_titles is None:
            list_subplot_titles = order

        for (dataset_to_plot, subplot_title) in zip(order, list_subplot_titles):
            df = pd.DataFrame(columns=['dice scores', 'struct', 'method'])

            # gather data
            for datatype, path_dice, method in zip(list_datasets, list_score_paths, list_methods):
                if datatype == dataset_to_plot:
                    dice = np.load(path_dice)
                    if eval_indices is not None:
                        dice = dice[eval_indices, :]
                    for idx, name in enumerate(names):
                        df = df.append(pd.DataFrame({'dice scores': dice[idx, :],
                                                     'struct': [name] * dice.shape[1],
                                                     'method': [method] * dice.shape[1]}),
                                       ignore_index=True, sort=False)
                    if show_average_on_subplots:
                        dice = np.mean(dice, axis=0)
                        df = df.append(pd.DataFrame({'dice scores': dice,
                                                     'struct': ['av'] * dice.shape[0],
                                                     'method': [method] * dice.shape[0]}),
                                       ignore_index=True, sort=False)

            # draw datatype-specific plot
            plt.figure(figsize=figsize)
            sns.set_style('whitegrid')
            ax = sns.boxplot(data=df, x='struct', y='dice scores', hue='method', linewidth=boxplot_linewidth,
                             width=boxplot_width, palette=palette, order=unique_names, fliersize=outlier_size)
            if y_label is not None:
                ax.set_ylabel(y_label, fontsize=fontsize)  # set fontsize of y-axis title
            else:
                ax.set_ylabel('', fontsize=2)
            ax.set_xlabel('', fontsize=2)  # give no name to x-axis
            ax.set_xlim(-boxplot_width / 2 - .02, len(unique_names) - (1 - boxplot_width / 2 - .02))
            ax.set_ylim(subplot_ylim[0], subplot_ylim[1] + .01)  # set right/left limits of plot
            ax.tick_params(axis='both', labelsize=fontsize)  # adjust tick size
            plt.yticks(np.arange(subplot_ylim[0], subplot_ylim[1] + .01, step=subplot_step_yticks))
            plt.xticks(np.arange(0, len(unique_names), step=1))
            if remove_legend:
                ax.get_legend().remove()
            else:
                handles, labels = ax.get_legend_handles_labels()
                ax.legend(handles=handles, labels=labels, fontsize=fontsize_legend)
            plt.title(subplot_title, fontsize=fontsize)
            plt.tight_layout()
            plt.show()
