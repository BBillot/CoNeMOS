# internal python import
import warnings

# third party
import os
import numpy as np
import neurite as ne
import tensorflow as tf
import voxelmorph as vxm
import keras.layers as KL
from datetime import datetime


current_time = str(datetime.now().strftime("%m_%d-%I_%M_%S_%p"))
path = os.path.join("/autofs/space/sand_001/users/kv567/train/", current_time)
os.mkdir(path)


def draw_crop_mask(x, crop_min=0, crop_max=0.5, axis=None, prob=1, rand=None, bilateral=False):
    """
    Draw a mask for a multiplicative cropping of the field of view of an ND tensor along an axis.
    Arguments:
        x: Input tensor or NumPy array defining the shape and data type of the mask.
        crop_min: Minimum proportion of voxels to remove.
        crop_max: Maximum proportion of voxels to remove.
        axis: Axis along which to crop, where None means any axis. With more than one
            axis specified, a single axis will be chosen at each execution.
        prob: Cropping probability. A value of 1 means always, 0 never.
        rand: Random-number generator. May be seeded for reproducible randomization.
        bilateral: Randomly distribute cropping proportion between top and bottom end.
    Returns:
        ND tensor of the input data type, with singleton dimensions where needed.
    Author:
        mu40
    If you find this function useful, please consider citing:
        M Hoffmann, B Billot, DN Greve, JE Iglesias, B Fischl, AV Dalca
        SynthMorph: learning contrast-invariant registration without acquired images
        IEEE Transactions on Medical Imaging (TMI), 41 (3), 543-558, 2022
        https://doi.org/10.1109/TMI.2021.3116879
    """
    if rand is None:
        rand = tf.random.Generator.from_non_deterministic_state()

    # Convert to tensor.
    x = tf.concat(x, axis=0)
    num_dim = len(x.shape)

    # Validate inputs.
    if axis is None:
        axis = range(num_dim)
    if np.isscalar(axis):
        axis = [axis]
    assert all(-1 < x < num_dim for x in axis), f'out-of-range axis in {axis}'
    assert 0 <= crop_min <= crop_max <= 1, f'invalid proportions {crop_min}, {crop_max}'

    # Decide how much to crop, making sure maxval is >0 to avoid errors.
    prop_cut = crop_max
    if crop_min < crop_max:
        prop_cut = rand.uniform(shape=[], minval=crop_min, maxval=crop_max)

    # Decide whether to crop.
    assert 0 <= prob <= 1, f'{prob} not a probability'
    if prob < 1:
        rand_bit = tf.less(rand.uniform(shape=[]), prob)
        prop_cut *= tf.cast(rand_bit, prop_cut.dtype)

    # Split cropping proportion for top and bottom.
    rand_prop = rand.uniform(shape=[])
    if not bilateral:
        rand_prop = tf.cast(tf.less(rand_prop, 0.5), prop_cut.dtype)
    prop_low = prop_cut * rand_prop
    prop_cen = 1 - prop_cut

    # Draw axis and determine FOV width.
    ind = rand.uniform(shape=[], maxval=len(axis), dtype=tf.int32)
    axis = tf.gather(axis, ind)
    width = tf.gather(tf.shape(x), indices=axis)

    # Assemble mask and reshape for multiplication.
    prop = tf.range(1, delta=1 / tf.cast(width, prop_cut.dtype))
    mask = tf.logical_and(tf.greater_equal(prop, prop_low), tf.less(prop, prop_low + prop_cen))
    mask = tf.cast(mask, x.dtype)
    shape = tf.roll((width, *np.ones(num_dim - 1)), shift=axis, axis=0)
    return tf.reshape(mask, shape)


def subsample_axis(x, stride_min=1, stride_max=1, axes=None, prob=1, upsample=True, rand=None):
    """
    Symmetrically subsample a tensor by a factor f (stride) along a single axis
    using nearest-neighbor interpolation and optionally upsample again, to reduce
    its resolution. Both f and the subsampling axis can be randomly drawn.
    Arguments:
        x: Input tensor or NumPy array of any type.
        stride_min: Minimum subsampling factor.
        stride_max: Maximum subsampling factor.
        axes: Tensor axes to draw the subsampling axis from. None means all axes.
        prob: Subsampling probability. A value of 1 means always, 0 never.
        upsample: Upsample the tensor to restore its original shape.
        rand: Random-number generator. May be seeded for reproducible randomization.
    Returns:
        Tensor with randomly thick slices along a random axis.
    Author:
        mu40
    If you find this function useful, please consider citing:
        M Hoffmann, B Billot, DN Greve, JE Iglesias, B Fischl, AV Dalca
        SynthMorph: learning contrast-invariant registration without acquired images
        IEEE Transactions on Medical Imaging (TMI), 41 (3), 543-558, 2022
        https://doi.org/10.1109/TMI.2021.3116879
    """
    # Validate inputs.
    if not tf.is_tensor(x):
        x = tf.constant(x)
    if rand is None:
        rand = tf.random.Generator.from_non_deterministic_state()

    # Validate axes.
    num_dim = len(x.shape)
    if axes is None:
        axes = range(num_dim)
    if np.isscalar(axes):
        axes = [axes]
    assert all(i in range(num_dim) for i in axes), 'invalid axis passed'

    # Draw axis and thickness.
    assert (0 < stride_min) and (stride_min <= stride_max), 'invalid strides'
    ind = rand.uniform(shape=[], minval=0, maxval=len(axes), dtype=tf.int32)
    ax = tf.gather(axes, ind)
    width = tf.gather(tf.shape(x), indices=ax)
    thick = rand.uniform(shape=[], minval=stride_min, maxval=stride_max)

    # Decide whether to downsample.
    assert 0 <= prob <= 1, f'{prob} not a probability'
    if prob < 1:
        rand_bit = tf.less(rand.uniform(shape=[]), prob)
        rand_not = tf.logical_not(rand_bit)
        thick = thick * tf.cast(rand_bit, thick.dtype) + tf.cast(rand_not, thick.dtype)

    # Resample.
    num_slice = tf.cast(width, thick.dtype) / thick + 0.5
    num_slice = tf.cast(num_slice, width.dtype)
    ind = tf.linspace(start=0, stop=width - 1, num=num_slice)
    ind = tf.cast(ind + 0.5, width.dtype)
    x = tf.gather(x, ind, axis=ax)
    if upsample:
        ind = tf.linspace(start=0, stop=tf.shape(x)[ax] - 1, num=width)
        ind = tf.cast(ind + 0.5, width.dtype)
        x = tf.gather(x, ind, axis=ax)

    return x


def draw_flip_matrix(grid_shape, shift_center=True, last_row=True, dtype=tf.float32, seed=None):
    """
    Draw a matrix that randomly flips the axes of N-dimensional space.
    Arguments:
        grid_shape: The spatial extent of the image in voxels, excluding batches and features.
        shift_center: Whether zero is at the center of the grid. Should be identical to the value
            used for vxm.utils.affine_to_shift or vxm.layers.SpatialTransformer.
        last_row: Whether to append the last row of the matrix.
        dtype: Output data type. Should be floating-point.
        seed: Integer for reproducible randomization.
    Returns:
        N(N+1) or (N+1)(N+1) flipping matrix of type tf.float32, depending on last_row.
    Author:
        mu40
    If you find this function useful, please consider citing:
        M Hoffmann, B Billot, DN Greve, JE Iglesias, B Fischl, AV Dalca
        SynthMorph: learning contrast-invariant registration without acquired images
        IEEE Transactions on Medical Imaging (TMI), 41 (3), 543-558, 2022
        https://doi.org/10.1109/TMI.2021.3116879
    """
    dtype = tf.dtypes.as_dtype(dtype)
    num_dim = len(grid_shape)
    grid_shape = tf.constant(grid_shape, dtype=dtype)

    # Decide which axes to flip.
    rand_bit = tf.greater(tf.random.normal(shape=(num_dim,), seed=seed), 0)
    rand_bit = tf.cast(rand_bit, dtype)
    diag = tf.pow(tf.cast(-1, dtype), rand_bit)
    diag = tf.linalg.diag(diag)

    # Account for center shift if needed.
    shift = tf.multiply(grid_shape - 1, rand_bit)
    shift = tf.reshape(shift, shape=(-1, 1))
    if shift_center:
        shift = tf.zeros(shape=(num_dim, 1), dtype=dtype)

    # Compose output.
    out = tf.concat((diag, shift), axis=1)
    if last_row:
        row = dtype.as_numpy_dtype((*[0] * num_dim, 1))
        row = np.reshape(row, newshape=(1, -1))
        out = tf.concat((out, row), axis=0)

    return out


def draw_swap_matrix(num_dim, last_row=True, dtype=tf.float32, seed=None):
    """
    Draw matrix that randomly swaps the axes of N-dimensional space.
    Arguments:
        num_dim: The number of spatial dimensions, excluding batches and features.
        last_row: Whether to append the last row of the matrix.
        dtype: Output data type. Should be floating-point.
        seed: Integer for reproducible randomization.
    Returns:
        N(N+1) or (N+1)(N+1) swapping matrix of type tf.float32, depending on last_row.
    Author:
        mu40
    If you find this function useful, please consider citing:
        M Hoffmann, B Billot, DN Greve, JE Iglesias, B Fischl, AV Dalca
        SynthMorph: learning contrast-invariant registration without acquired images
        IEEE Transactions on Medical Imaging (TMI), 41 (3), 543-558, 2022
        https://doi.org/10.1109/TMI.2021.3116879
    """
    dtype = tf.dtypes.as_dtype(dtype)

    mat = tf.eye(num_dim, num_dim + 1, dtype=dtype)
    mat = tf.random.shuffle(mat, seed=seed)

    row = dtype.as_numpy_dtype((*[0] * num_dim, 1))
    row = np.reshape(row, newshape=(1, -1))

    return tf.concat((mat, row), axis=0) if last_row else mat


class GaussianNoise(tf.keras.layers.Layer):
    """
    Sample noise from a normal distribution, sampling its standard deviation (SD)
    from a uniform distribution.
    Author:
        mu40
    If you find this function useful, please consider citing:
        M Hoffmann, B Billot, DN Greve, JE Iglesias, B Fischl, AV Dalca
        SynthMorph: learning contrast-invariant registration without acquired images
        IEEE Transactions on Medical Imaging (TMI), 41 (3), 543-558, 2022
        https://doi.org/10.1109/TMI.2021.3116879
    """

    def __init__(self, noise_min=0.01, noise_max=0.10, noise_only=False, absolute=False, axes=(0, -1), seed=None,
                 **kwargs):
        """
        Parameters:
            noise_min: Minimum noise SD. Shape must be broadcastable with the output shape.
            noise_max: Maximum noise SD. Shape must be broadcastable with the output shape.
            noise_only: Return the noise tensor only, instead of adding it to the input.
            absolute: Instead of interpreting the SD bounds relative to the absolute maximum
                value of the input tensor, treat them as absolute values.
            axes: Input axes along which noise will be sampled with a separate SD.
            seed: Optional seed for initializing the random number generation.
        """
        self.noise_min = noise_min
        self.noise_max = noise_max
        self.noise_only = noise_only
        self.absolute = absolute
        self.axes = axes
        self.seed = seed
        self.rand = None
        super().__init__(**kwargs)

    def get_config(self):
        config = super().get_config().copy()
        config.update({'noise_min': self.noise_min,
                       'noise_max': self.noise_max,
                       'noise_only': self.noise_only,
                       'absolute': self.absolute,
                       'axes': self.axes,
                       'seed': self.seed})
        return config

    def build(self, in_shape):
        num_dim = len(in_shape)
        self.axes = np.ravel(self.axes)
        self.axes = [ax + num_dim if ax < 0 else ax for ax in self.axes]
        assert all(0 <= ax < num_dim for ax in self.axes), 'invalid axes'

        self.rand = tf.random.Generator.from_non_deterministic_state()
        if self.seed is not None:
            self.rand.reset_from_seed(self.seed)

    def call(self, x, **kwargs):
        if self.noise_max == 0:
            return x

        # Types.
        assert x.dtype.is_floating or x.dtype.is_complex, 'non-FP output type'
        real_type = x.dtype.real_dtype

        # Shapes.
        shape_out = tf.shape(x)
        shape_sd = []
        for i, _ in enumerate(x.shape):
            shape_sd.append(shape_out[i] if i in self.axes else 1)

        # Standard deviation.
        sd = self.rand.uniform(shape_sd, minval=self.noise_min, maxval=self.noise_max, dtype=real_type)
        if not self.absolute:
            sd *= tf.reduce_max(tf.abs(x))

        # Direct sampling of complex numbers not supported.
        if x.dtype.is_complex:
            noise = tf.complex(self.rand.normal(shape_out, stddev=sd, dtype=real_type),
                               self.rand.normal(shape_out, stddev=sd, dtype=real_type))

        else:
            noise = self.rand.normal(shape_out, stddev=sd, dtype=real_type)

        return noise if self.noise_only else x + noise


def image_to_image(in_shape=None,
                   out_shape=None,
                   input_model=None,
                   max_shift=0,
                   max_rotate=0,
                   max_scale=0,
                   max_shear=0,
                   normal_shift=False,  # Sample normally, treating max as SD.
                   normal_rotate=False,  # Sample normally, treating max as SD.
                   normal_scale=False,  # Sample normally, truncating beyond 2 SDs.
                   normal_shear=False,  # Sample normally, treating max as SD.
                   axes_flip=False,
                   axes_swap=False,
                   warp_res=16,
                   warp_min=0,
                   warp_max=1,
                   warp_zero_mean=False,
                   crop_min=0,
                   crop_max=0.2,
                   crop_prob=0,
                   crop_axes=None,
                   lut_blur_min=32,
                   lut_blur_max=64,
                   noise_min=0.01,
                   noise_max=0.10,
                   blur_min=0,
                   blur_max=1,
                   bias_res=40,
                   bias_min=0,
                   bias_max=0.3,
                   bias_func=tf.exp,
                   slice_stride_min=3,
                   slice_stride_max=6,
                   slice_prob=0,
                   slice_axes=None,
                   clip_max=255,
                   normalize=True,
                   gamma=0.25,
                   half_res=False,
                   seeds=None,
                   return_vel=False,
                   return_def=False,
                   return_aff=False,
                   name=0,
                   **kwargs):
    """Build model for image augmentation.
    This is work in progress and should not be shared externally. If you would
    like to use the code for an abstract or paper, please contact the author.
    Author:
        mu40
    Arguments:
        in_shape: List of the spatial dimensions of the input images.
        out_shape: List of the spatial dimensions of the outputs. Inputs will
            be symmetrically cropped or zero-padded to fit. If None, uses the
            input shape.
        input_model: Model whose outputs will be used as data inputs, and whose
            inputs will be used as inputs to the returned model.
        max_shift: Upper bound on the magnitude of translations drawn in voxels.
        max_rotate: Upper bound on the magnitude of rotations drawn in degrees.
        max_scale: Upper bound on the difference of the scaling magnitude from 1.
        max_shear: Upper bound on the shearing magnitude drawn.
        axes_flip: Whether to randomly flip axes of the outputs.
        axes_swap: Whether to randomly permute axes of the outputs. Requires
            isotropic output dimensions.
        warp_res: List of factors N determining the resolution 1/N relative to
            the inputs at which the SVF is drawn.
        warp_min: Lower bound on the SDs used when drawing the SVF.
        warp_max: Upper bound on the SDs used when drawing the SVF.
        warp_zero_mean: Ensure that the SVF components have zero mean.
        crop_min: Lower bound on the proportion of the FOV to crop.
        crop_max: Upper bound on the proportion of the FOV to crop.
        crop_prob: Probability that we crop the FOV along an axis.
        crop_axes: Axes from which to draw for FOV cropping. None means all spatial axes.
        lut_blur_min: Lower bound on the smoothing SD for random-contrast lookup.
        lut_blur_max: Upper bound on the smoothing SD for random-contrast lookup. Disabled if zero.
        noise_min: Lower bound on the noise SDs relative to the absolute maximum
            intensity across the batch. A value of 0.01 means 1%.
        noise_max: Upper bound on the noise SDs relative to the absolute maximum
            intensity across the batch. A value of 0.01 means 1%.
        blur_min: Lower bound on the smoothing SDs. Can be a scalar or list.
        blur_max: Upper bound on the smoothing SDs. Can be a scalar or list.
        bias_res: List of factors N determining the resolution 1/N relative to
            the inputs at which the bias field is drawn.
        bias_min: Lower bound on the bias SDs.
        bias_max: Upper bound on the bias SDs.
        bias_func: Function applied voxel-wise to condition the bias field.
        slice_stride_min: Lower bound on slice thickness in original voxel units.
        slice_stride_max: Upper bound on slice thickness in original voxel units.
        slice_prob: Probability that we subsample to create thick slices.
        slice_axes: Axes from which to draw slice normal direction. None means all spatial axes.
        clip_max: Integer value at which the image intensities are clipped.
        normalize: Whether the image is min-max normalized.
        gamma: SD of random global intensity exponentiation, i.e. gamma
            augmentation.
        seeds: Dictionary of seeds for synchronizing randomization across models.
        return_vel: Whether to append the half-resolution SVF to the model outputs.
        return_def: Whether to append the combined displacement field to the
            model outputs.
        return_aff: Whether to append the (N+1)x(N+1) affine transformation to
            the model outputs.
        name: Model identifier used to create unique layer names for including
            this model multiple times.
    Returns:
        Image-augmentation model.
    """
    # Keywords.
    known = ('lut_std', 'gamma_std', 'dc_offset')
    for k in kwargs:
        assert k in known, f'unknown argument {k}'

    # Deprecation.
    def warn(old, new):
        warnings.warn(f'Argument {old} to nes.models.image_to_image is '
                      f'deprecated and will be removed. Use {new} instead.')

    if 'lut_std' in kwargs:
        warn(old='lut_std', new='lut_blur_max')
        lut_blur_min = kwargs['lut_std']
        lut_blur_max = kwargs['lut_std']

    if 'gamma_std' in kwargs:
        warn(old='gamma_std', new='gamma')
        gamma = kwargs['gamma_std']

    if 'dc_offset' in kwargs:
        warnings.warn('Argument dc_offset to nes.models.image_to_image is '
                      'deprecated and will be removed in the future.')
        dc_offset = kwargs['dc_offset']

    if seeds is None:
        seeds = {}

    # Compute type.
    major, minor, _ = map(int, tf.__version__.split('.'))
    if major > 2 or (major == 2 and minor > 3):
        compute_type = tf.keras.mixed_precision.global_policy().compute_dtype
        compute_type = tf.dtypes.as_dtype(compute_type)
    else:
        compute_type = tf.float32
    integer_type = tf.int32

    # Input model.
    if input_model is None:
        image = KL.Input(shape=(*in_shape, 1), name=f'input_{name}', dtype=compute_type)
        input_model = tf.keras.Model(*[image] * 2)
    image = input_model.output
    if image.dtype != compute_type:
        image = tf.cast(image, compute_type)

    # Dimensions.
    in_shape = np.asarray(image.shape[1:-1])
    if out_shape is None:
        out_shape = in_shape
    out_shape = np.array(out_shape) // (2 if half_res else 1)
    num_dim = len(in_shape)
    batch_size = tf.shape(image)[0]

    # Affine transform.
    def sample_motion(shape, lim, normal=False, truncate=False, seed=None, dtype=tf.float32):
        """ Wrap TensorFlow function for random-number generation."""
        prop = dict(shape=shape, seed=seed, dtype=dtype)
        if normal:
            func = 'truncated_normal' if truncate else 'normal'
            prop.update(dict(stddev=lim))
        else:
            func = 'uniform'
            prop.update(dict(minval=-lim, maxval=lim))
        return getattr(tf.random, func)(**prop)

    parameters = tf.concat((sample_motion(shape=(batch_size, num_dim), dtype=compute_type,
                                          lim=max_shift, normal=normal_shift, seed=seeds.get('shift')),
                            sample_motion(shape=(batch_size, 1 if num_dim == 2 else 3), dtype=compute_type,
                                          lim=max_rotate, normal=normal_rotate, seed=seeds.get('rotate')),
                            sample_motion(shape=(batch_size, num_dim), truncate=True, dtype=compute_type,
                                          lim=max_scale, normal=normal_scale, seed=seeds.get('scale')),
                            sample_motion(shape=(batch_size, 1 if num_dim == 2 else 3), dtype=compute_type,
                                          lim=max_shear, normal=normal_shear, seed=seeds.get('shear'))),
                           axis=-1)
    affine = vxm.layers.ParamsToAffineMatrix(ndims=num_dim, deg=True, shift_scale=True, last_row=True)(parameters)

    # Shift origin to rotate about image center.
    origin = np.eye(num_dim + 1)
    origin[:num_dim, -1] = -0.5 * (in_shape - 1)

    # Symmetric padding: center output volume within input volume.
    center = np.eye(num_dim + 1)
    center[:num_dim, -1] = np.round(0.5 * (in_shape - (2 if half_res else 1) * out_shape))

    # Apply transform in input space to avoid rescaling translations at half resolution.
    scale = np.diag((*[2 if half_res else 1] * num_dim, 1))
    trans = np.linalg.inv(origin) @ affine @ origin @ center @ scale

    # Axis randomization.
    if axes_flip:
        trans = KL.Lambda(lambda x: x @ draw_flip_matrix(
            out_shape, shift_center=False, dtype=compute_type, seed=seeds.get('flip')))(trans)
    if axes_swap:
        assert all(x == out_shape[0] for x in out_shape), 'non-isotropic output shape'
        trans = KL.Lambda(lambda x: x @ draw_swap_matrix(
            num_dim, dtype=compute_type, seed=seeds.get('swap')))(trans)

    # Convert to dense transform.
    trans = vxm.layers.AffineToDenseShift(out_shape, shift_center=False)(trans[:, :num_dim, :])

    # Diffeomorphic deformation.
    if warp_max > 0:
        # Velocity field.
        vel_draw = lambda x: ne.utils.augment.draw_perlin(
            out_shape=(*out_shape // (1 if half_res else 2), num_dim),
            scales=np.asarray(warp_res) / 2,
            min_std=warp_min, max_std=warp_max,
            dtype=compute_type, seed=seeds.get('warp'))
        vel_field = KL.Lambda(lambda x: tf.map_fn(vel_draw, x), name=f'vel_{name}')(image)
        if warp_zero_mean:
            vel_field -= tf.reduce_mean(vel_field, axis=range(1, num_dim + 1))
        def_field = vxm.layers.VecInt(int_steps=5, name=f'vec_int_{name}')(vel_field)
        if not half_res:
            def_field = vxm.layers.RescaleTransform(zoom_factor=2, name=f'def_{name}')(def_field)

        # Compose with affine.
        trans = vxm.layers.ComposeTransform()([trans, def_field])

    # Apply transform.
    image = vxm.layers.SpatialTransformer(interp_method='linear', fill_value=0, name=f'trans_{name}')([image, trans])

    # Cropping.
    if crop_prob > 0:
        crop_rand = tf.random.Generator.from_non_deterministic_state()
        if seeds.get('crop') is not None:
            crop_rand.reset_from_seed(seeds.get('crop'))
        crop_func = lambda x: x * draw_crop_mask(x, crop_min=crop_min, crop_max=crop_max, prob=crop_prob,
                                                 rand=crop_rand,
                                                 axis=crop_axes if crop_axes else range(1, num_dim + 1), )
        image = KL.Lambda(crop_func, name=f'crop_{name}')(image)

    # Random contrast lookup.
    if lut_blur_max > 0:
        lut_num = 256
        lut_max = lut_num - 1
        lut_draw = 1024

        # Smooth table. Filter shape: space, in, out.
        kernel = KL.Lambda(lambda x: ne.utils.gaussian_kernel(lut_blur_max, min_sigma=lut_blur_min,
                                                              random=lut_blur_min != lut_blur_max, dtype=image.dtype,
                                                              seed=seeds.get('lut')))(image)
        kernel = tf.reshape(kernel, shape=(-1, 1, 1))
        lut = tf.random.uniform(shape=(batch_size, lut_draw, 1), minval=0, maxval=lut_max, dtype=compute_type,
                                seed=seeds.get('lut'))
        lut = tf.nn.convolution(lut, kernel, padding='SAME')[..., 0]

        # Cut tapered edges.
        lut_cen = np.arange(lut_num) + (lut_draw - lut_num) // 2
        lut = tf.gather(lut, indices=lut_cen, axis=1)

        # Normalize and apply.
        lut_norm = KL.Lambda(lambda x: tf.map_fn(ne.utils.minmax_norm, x) * lut_max)
        lut, image = map(lut_norm, (lut, image))
        lut_func = lambda x: tf.gather(x[0], indices=tf.cast(x[1], integer_type), axis=0)
        image = KL.Lambda(lambda x: tf.map_fn(lut_func, x, fn_output_signature=compute_type))([lut, image])

    # Noise.
    image = GaussianNoise(noise_min, noise_max, seed=seeds.get('noise'))(image)

    # Blur.
    blur_min, blur_max = map(lambda x: [x] if np.isscalar(x) else x, (blur_min, blur_max))
    blur_min, blur_max = map(lambda x: np.ravel(x) / (2 if half_res else 1), (blur_min, blur_max))
    blur_min, blur_max = map(np.ndarray.tolist, (blur_min, blur_max))
    blur_min, blur_max = map(lambda x: x if len(x) > 1 else x * num_dim, (blur_min, blur_max))
    assert len(blur_min) == len(blur_max) == num_dim, 'unacceptable number of blurring sigmas'
    if any(x > 0 for x in blur_max):
        kernels = KL.Lambda(lambda x: ne.utils.gaussian_kernel(
            blur_max, min_sigma=blur_min, separate=True, random=blur_min != blur_max,
            dtype=x.dtype, seed=seeds.get('blur')))(image)
        image = ne.utils.separable_conv(image, kernels, batched=True)

    # Bias field.
    if bias_max > 0:
        bias_draw = lambda x: ne.utils.augment.draw_perlin(
            out_shape=(*out_shape, 1), min_std=bias_min, max_std=bias_max,
            scales=np.asarray(bias_res) / (2 if half_res else 1),
            dtype=image.dtype, seed=seeds.get('bias'), )
        bias_field = KL.Lambda(lambda x: tf.map_fn(bias_draw, x), name=f'bias_draw_{name}')(image)
        bias_field = KL.Lambda(bias_func, name=f'bias_func_{name}')(bias_field)
        image *= bias_field

    # Create thick slices.
    if slice_prob > 0:
        if half_res and slice_stride_max > 1:
            slice_stride_min = max(1, slice_stride_min // 2)
            slice_stride_max //= 2
        slice_rand = tf.random.Generator.from_non_deterministic_state()
        if seeds.get('slice') is not None:
            slice_rand.reset_from_seed(seeds.get('slice'))
        image = KL.Lambda(lambda x: subsample_axis(
            x, stride_min=slice_stride_min, stride_max=slice_stride_max,
            prob=slice_prob, rand=slice_rand,
            axes=slice_axes if slice_axes else range(1, num_dim + 1),
        ), name=f'slice_{name}')(image)
        image = KL.Reshape((*out_shape, -1))(image)

    # Intensity manipulations.
    image = tf.clip_by_value(image, clip_value_min=0, clip_value_max=clip_max, name=f'clip_{name}')
    if normalize:
        image = KL.Lambda(lambda x: tf.map_fn(ne.utils.minmax_norm, x))(image)
    if gamma > 0:
        gamma = tf.random.normal(shape=(batch_size, *[1] * num_dim, 1), stddev=gamma, dtype=image.dtype,
                                 seed=seeds.get('gamma'))
        image = tf.pow(image, tf.exp(gamma), name=f'gamma_{name}')
    if 'dc_offset' in locals() and dc_offset > 0:
        image += tf.random.uniform(shape=(batch_size, *[1] * num_dim, 1), maxval=dc_offset, dtype=image.dtype,
                                   seed=seeds.get('dc_offset'))

    outputs = [image]
    if return_vel:
        outputs.append(vel_field)
    if return_def:
        outputs.append(def_field)
    if return_aff:
        outputs.append(affine)

    return tf.keras.Model(input_model.inputs, outputs)
