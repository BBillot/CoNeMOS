import numpy as np
from ext.lab2im import utils
import numpy.random as npr


def build_model_inputs(path_images,
                       path_labels,
                       path_descriptors=None,
                       subjects_prob=None,
                       batchsize=1,
                       dtype_images='float32',
                       dtype_labels='int32',
                       dtype_descriptor='float32'):

    # Generate!
    while True:

        # randomly pick as many images as batchsize
        indices = npr.choice(np.arange(len(path_labels)), size=batchsize, p=subjects_prob)

        # initialise input lists
        list_batch_images = list()
        list_batch_labels = list()
        list_batch_label_descriptors = list()

        for idx in indices:

            # get batch image
            image = utils.load_volume(path_images[idx], dtype=dtype_images)
            list_batch_images.append(utils.add_axis(image, axis=[0, -1]))

            # get batch label map
            labels = utils.load_volume(path_labels[idx], dtype=dtype_labels)
            list_batch_labels.append(utils.add_axis(labels, axis=[0, -1]))

            # build list of training pairs
            if path_descriptors is not None:
                label_descriptor = np.load(path_descriptors[idx]).astype(dtype_descriptor)
                list_batch_label_descriptors.append(utils.add_axis(label_descriptor, axis=0))

        # build list of training pairs
        list_model_inputs = [list_batch_images, list_batch_labels]
        if path_descriptors is not None:
            list_model_inputs.append(list_batch_label_descriptors)

        # concatenate individual input types if batchsize > 1
        if batchsize > 1:
            list_model_inputs = [np.concatenate(item, 0) for item in list_model_inputs]
        else:
            list_model_inputs = [item[0] for item in list_model_inputs]

        yield list_model_inputs


def get_paths(image_dir, labels_dir, label_descriptor_dir=None, subjects_prob=None, data_perc=100):
    """List all the images in image_dir with the corresponding label maps in labels_dir.
    Images and label maps must have the same ordering in each folder.
    image_dir and labels dir can also be list of folders.
    Optionally, we can also add label descriptors in the case of label maps with several labels (the label descriptors
    simply tell which labels are present in a label map with one-hot encoding). label_descriptor_dir can also be a list
    of folders.
    """

    # reformat
    if not isinstance(image_dir, (list, tuple)):
        image_dir = [image_dir]
    if not isinstance(labels_dir, (list, tuple)):
        labels_dir = [labels_dir]
    if label_descriptor_dir is not None:
        if not isinstance(label_descriptor_dir, (list, tuple)):
            label_descriptor_dir = [label_descriptor_dir]

    # paths images/labels
    path_images = list()
    path_labels = list()
    for (im_dir, lab_dir) in zip(image_dir, labels_dir):
        path_images += utils.list_images_in_folder(im_dir)
        path_labels += utils.list_images_in_folder(lab_dir)

    n_images = len(path_images)
    assert len(path_labels) == n_images, "There should be %s label maps, got %s" % (n_images, len(path_labels))

    # path label descriptors
    if label_descriptor_dir is not None:
        path_label_descriptors = list()
        for lab_descriptor_dir in label_descriptor_dir:
            path_label_descriptors += utils.list_files(lab_descriptor_dir, expr='.npy')
    else:
        path_label_descriptors = None

    # make sure subjects_prob sums to 1
    if subjects_prob is not None:
        subjects_prob = utils.load_array_if_path(subjects_prob)
        subjects_prob /= np.sum(subjects_prob)

    # decrease size of datasets for ablation
    if data_perc != 100:
        stop_idx = min(n_images, int(n_images * data_perc / 100))
        path_images = path_images[:stop_idx]
        path_labels = path_labels[:stop_idx]
        if path_label_descriptors is not None:
            path_label_descriptors = path_label_descriptors[:stop_idx]
        if subjects_prob is not None:
            subjects_prob = subjects_prob[:stop_idx]
            values, counts = np.unique(subjects_prob, return_counts=True)
            for value, counts in zip(values, counts):
                subjects_prob[subjects_prob == value] = 1 / (len(values) * counts)
            subjects_prob /= np.sum(subjects_prob)

    return path_images, path_labels, path_label_descriptors, subjects_prob
