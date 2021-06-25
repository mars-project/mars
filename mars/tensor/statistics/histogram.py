# Copyright 1999-2021 Alibaba Group Holding Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import warnings
import operator

import numpy as np

from ... import opcodes as OperandDef
from ... import tensor as mt
from ...core import recursive_tile
from ...core.context import get_context
from ...serialization.serializables import AnyField, TupleField, \
    KeyField, BoolField
from ...utils import has_unknown_shape
from ..core import TENSOR_TYPE, TENSOR_CHUNK_TYPE, TensorOrder
from ..operands import TensorOperand, TensorOperandMixin
from ..datasource import tensor as astensor
from ..arithmetic.utils import chunk_tree_add
from ..utils import is_asc_sorted
from ..array_utils import as_same_device, device

# note: some logic of this file were adopted from `numpy/lib/histograms`


def _ptp(range_):
    """Peak-to-peak value of x.

    This implementation avoids the problem of signed integer arrays having a
    peak-to-peak value that cannot be represented with the array's data type.
    This function returns an unsigned value for signed integer arrays.
    """
    return _unsigned_subtract(*range_[::-1])


class HistBinSelector:
    def __init__(self, histogram_bin_edges_op, x, range, raw_range):
        self._op = histogram_bin_edges_op
        self._x = x
        self._range = range
        self._raw_range = raw_range
        self._width = None

    def check(self):
        # not checked before
        width = self()
        if width is None:
            return
        self._width = width = yield from recursive_tile(width)
        yield [c.data for c in width.chunks]

    def __call__(self):
        return

    def get_result(self):
        ctx = get_context()
        width = ctx.get_chunks_result([self._width.chunks[0].key])[0]
        return width


class HistBinSqrtSelector(HistBinSelector):
    """
    Square root histogram bin estimator.

    Bin width is inversely proportional to the data size. Used by many
    programs for its simplicity.
    """

    def get_result(self):
        return _ptp(self._raw_range) / np.sqrt(self._x.size)


class HistBinSturgesSelector(HistBinSelector):
    """
    Sturges histogram bin estimator.

    A very simplistic estimator based on the assumption of normality of
    the data. This estimator has poor performance for non-normal data,
    which becomes especially obvious for large data sets. The estimate
    depends only on size of the data.
    """

    def get_result(self):
        return _ptp(self._raw_range) / (np.log2(self._x.size) + 1.0)


class HistBinRiceSelector(HistBinSelector):
    """
    Rice histogram bin estimator.

    Another simple estimator with no normality assumption. It has better
    performance for large data than Sturges, but tends to overestimate
    the number of bins. The number of bins is proportional to the cube
    root of data size (asymptotically optimal). The estimate depends
    only on size of the data.
    """

    def get_result(self):
        return _ptp(self._raw_range) / (2.0 * self._x.size ** (1.0 / 3))


class HistBinScottSelector(HistBinSelector):
    """
    Scott histogram bin estimator.

    The binwidth is proportional to the standard deviation of the data
    and inversely proportional to the cube root of data size
    (asymptotically optimal).
    """

    def __call__(self):
        return (24.0 * np.pi**0.5 / self._x.size)**(1.0 / 3.0) * mt.std(self._x)


class HistBinStoneSelector(HistBinSelector):
    """
    Histogram bin estimator based on minimizing the estimated integrated squared error (ISE).

    The number of bins is chosen by minimizing the estimated ISE against the unknown true distribution.
    The ISE is estimated using cross-validation and can be regarded as a generalization of Scott's rule.
    https://en.wikipedia.org/wiki/Histogram#Scott.27s_normal_reference_rule

    This paper by Stone appears to be the origination of this rule.
    http://digitalassets.lib.berkeley.edu/sdtr/ucb/text/34.pdf
    """

    def __call__(self):
        n = self._x.size
        ptp_x = _ptp(self._raw_range)

        if n <= 1 or ptp_x == 0:
            return

        nbins_upper_bound = max(100, int(np.sqrt(n)))
        candidates = []
        for nbins in range(1, nbins_upper_bound + 1):
            hh = ptp_x / nbins
            p_k = histogram(self._x, bins=nbins, range=self._range)[0] / n
            candidate = (2 - (n + 1) * p_k.dot(p_k)) / hh
            candidates.append(candidate)
        nbins = mt.stack(candidates).argmin() + 1
        return ptp_x / nbins

    def get_result(self):
        ptp_x = _ptp(self._raw_range)
        if self._x.size <= 1 or ptp_x == 0:
            return 0.0
        else:
            return super().get_result()


class HistBinDoaneSelector(HistBinSelector):
    """
    Doane's histogram bin estimator.

    Improved version of Sturges' formula which works better for
    non-normal data. See
    stats.stackexchange.com/questions/55134/doanes-formula-for-histogram-binning
    """

    def __call__(self):
        x = self._x
        if x.size <= 2:
            return

        sg1 = np.sqrt(6.0 * (x.size - 2) / ((x.size + 1.0) * (x.size + 3)))
        sigma = mt.std(x)
        g1 = mt.mean(((x - mt.mean(x)) / sigma)**3)
        ret = _ptp(self._raw_range) / (1.0 + np.log2(x.size) +
                                      mt.log2(1.0 + mt.absolute(g1) / sg1))
        return mt.where(sigma > 0.0, ret, 0.0)

    def get_result(self):
        if self._x.size <= 2:
            return 0.0
        else:
            return super().get_result()


class HistBinFdSelector(HistBinSelector):
    """
    The Freedman-Diaconis histogram bin estimator.

    The Freedman-Diaconis rule uses interquartile range (IQR) to
    estimate binwidth. It is considered a variation of the Scott rule
    with more robustness as the IQR is less affected by outliers than
    the standard deviation. However, the IQR depends on fewer points
    than the standard deviation, so it is less accurate, especially for
    long tailed distributions.

    If the IQR is 0, this function returns 1 for the number of bins.
    Binwidth is inversely proportional to the cube root of data size
    (asymptotically optimal).
    """

    def __call__(self):
        iqr = mt.subtract(*mt.percentile(self._x, [75, 25]))
        return 2.0 * iqr * self._x.size ** (-1.0 / 3.0)


class HistBinAutoSelector(HistBinSelector):
    """
    Histogram bin estimator that uses the minimum width of the
    Freedman-Diaconis and Sturges estimators if the FD bandwidth is non zero
    and the Sturges estimator if the FD bandwidth is 0.

    The FD estimator is usually the most robust method, but its width
    estimate tends to be too large for small `x` and bad for data with limited
    variance. The Sturges estimator is quite good for small (<1000) datasets
    and is the default in the R language. This method gives good off the shelf
    behaviour.

    If there is limited variance the IQR can be 0, which results in the
    FD bin width being 0 too. This is not a valid bin width, so
    ``np.histogram_bin_edges`` chooses 1 bin instead, which may not be optimal.
    If the IQR is 0, it's unlikely any variance based estimators will be of
    use, so we revert to the sturges estimator, which only uses the size of the
    dataset in its calculation.
    """

    def __init__(self, histogram_bin_edges_op, x, range, raw_range):
        super().__init__(histogram_bin_edges_op, x, range, raw_range)
        self._bin_fd = HistBinFdSelector(
            histogram_bin_edges_op, x, range, raw_range)
        self._bin_sturges = HistBinSturgesSelector(
            histogram_bin_edges_op, x, range, raw_range)

    def __call__(self):
        return self._bin_fd()

    def get_result(self):
        fd_bw = super().get_result()
        sturges_bw = self._bin_sturges.get_result()
        if fd_bw:
            return min(fd_bw, sturges_bw)
        else:
            # limited variance, so we return a len dependent bw estimator
            return sturges_bw


# Private dict initialized at module load time
_hist_bin_selectors = {'stone': HistBinStoneSelector,
                       'auto': HistBinAutoSelector,
                       'doane': HistBinDoaneSelector,
                       'fd': HistBinFdSelector,
                       'rice': HistBinRiceSelector,
                       'scott': HistBinScottSelector,
                       'sqrt': HistBinSqrtSelector,
                       'sturges': HistBinSturgesSelector}


def _ravel_and_check_weights(a, weights):
    """ Check a and weights have matching shapes, and ravel both """
    a = astensor(a)

    # Ensure that the array is a "subtractable" dtype
    if a.dtype == np.bool_:
        warnings.warn(f"Converting input from {a.dtype} to {np.uint8} for compatibility.",
                      RuntimeWarning, stacklevel=3)
        a = a.astype(np.uint8)

    if weights is not None:
        weights = astensor(weights)
        if weights.shape != a.shape:
            raise ValueError(
                'weights should have the same shape as a.')
        weights = weights.ravel()
    a = a.ravel()
    return a, weights


def _check_range(range):
    first_edge, last_edge = range
    if first_edge > last_edge:
        raise ValueError(
            'max must be larger than min in range parameter.')
    if not (np.isfinite(first_edge) and np.isfinite(last_edge)):
        raise ValueError(
            f"supplied range of [{first_edge}, {last_edge}] is not finite")
    return first_edge, last_edge


def _get_outer_edges(a, range):
    """
    Determine the outer bin edges to use, from either the data or the range
    argument
    """
    if range is not None:
        first_edge, last_edge = _check_range(range)
    else:
        assert a.size == 0
        # handle empty arrays. Can't determine range, so use 0-1.
        first_edge, last_edge = 0, 1

    # expand empty range to avoid divide by zero
    if first_edge == last_edge:
        first_edge = first_edge - 0.5
        last_edge = last_edge + 0.5

    return first_edge, last_edge


def _unsigned_subtract(a, b):
    """
    Subtract two values where a >= b, and produce an unsigned result

    This is needed when finding the difference between the upper and lower
    bound of an int16 histogram
    """
    # coerce to a single type
    signed_to_unsigned = {
        np.byte: np.ubyte,
        np.short: np.ushort,
        np.intc: np.uintc,
        np.int_: np.uint,
        np.longlong: np.ulonglong
    }
    dt = np.result_type(a, b)
    try:
        dt = signed_to_unsigned[dt.type]
    except KeyError:  # pragma: no cover
        return np.subtract(a, b, dtype=dt)
    else:
        # we know the inputs are integers, and we are deliberately casting
        # signed to unsigned
        return np.subtract(a, b, casting='unsafe', dtype=dt)


def _get_bin_edges(op, a, bins, range, weights):
    # parse the overloaded bins argument
    n_equal_bins = None
    bin_edges = None
    first_edge = None
    last_edge = None

    if isinstance(bins, str):
        # when `bins` is str, x.min() and x.max()
        # will be calculated in advance
        bin_name = bins
        if a.size > 0:
            assert range is not None

        raw_range = range
        first_edge, last_edge = _get_outer_edges(a, range)

        if a.size == 0:
            n_equal_bins = 1
        else:
            # Do not call selectors on empty arrays
            selector = _hist_bin_selectors[bin_name](op, a, (first_edge, last_edge), raw_range)
            yield from selector.check()
            width = selector.get_result()
            if width:
                n_equal_bins = int(np.ceil(_unsigned_subtract(last_edge, first_edge) / width))
            else:
                # Width can be zero for some estimators, e.g. FD when
                # the IQR of the data is zero.
                n_equal_bins = 1

    elif mt.ndim(bins) == 0:
        first_edge, last_edge = _get_outer_edges(a, range)
        n_equal_bins = bins

    else:
        # cannot be Tensor, must be calculated first
        assert mt.ndim(bins) == 1 and not isinstance(bins, TENSOR_TYPE)
        bin_edges = np.asarray(bins)
        if not is_asc_sorted(bin_edges):
            raise ValueError(
                '`bins` must increase monotonically, when an array')

    if n_equal_bins is not None:
        # numpy gh-10322 means that type resolution rules are dependent on array
        # shapes. To avoid this causing problems, we pick a type now and stick
        # with it throughout.
        bin_type = np.result_type(first_edge, last_edge, a)
        if np.issubdtype(bin_type, np.integer):
            bin_type = np.result_type(bin_type, float)

        # bin edges must be computed
        bin_edges = mt.linspace(
            first_edge, last_edge, n_equal_bins + 1,
            endpoint=True, dtype=bin_type, gpu=op.gpu)
        return bin_edges, (first_edge, last_edge, n_equal_bins)
    else:
        return mt.tensor(bin_edges), None


class TensorHistogramBinEdges(TensorOperand, TensorOperandMixin):
    _op_type_ = OperandDef.HISTOGRAM_BIN_EDGES

    _input = KeyField('input')
    _bins = AnyField('bins')
    _range = TupleField('range')
    _weights = KeyField('weights')
    _uniform_bins = TupleField('uniform_bins')

    def __init__(self, input=None, bins=None, range=None, weights=None,
                 input_min=None, input_max=None, **kw):
        super().__init__(_input=input, _bins=bins, _range=range,
                         _weights=weights, **kw)

    @property
    def input(self):
        return self._input

    @property
    def bins(self):
        return self._bins

    @property
    def range(self):
        return self._range

    @property
    def weights(self):
        return self._weights

    def _set_inputs(self, inputs):
        super()._set_inputs(inputs)
        inputs_iter = iter(self._inputs)
        self._input = next(inputs_iter)
        if isinstance(self._bins, (TENSOR_TYPE, TENSOR_CHUNK_TYPE)):
            self._bins = next(inputs_iter)
        if self._weights is not None:
            self._weights = next(inputs_iter)

    def __call__(self, a, bins, range, weights):
        if range is not None:
            _check_range(range)
        if isinstance(bins, str):
            # string, 'auto', 'stone', ...
            # shape is unknown
            bin_name = bins
            # if `bins` is a string for an automatic method,
            # this will replace it with the number of bins calculated
            if bin_name not in _hist_bin_selectors:
                raise ValueError(
                    f"{bin_name!r} is not a valid estimator for `bins`")
            if weights is not None:
                raise TypeError("Automated estimation of the number of "
                                "bins is not supported for weighted data")
            if isinstance(range, tuple) and len(range) == 2:
                # if `bins` is a string, e.g. 'auto', 'stone'...,
                # and `range` provided as well,
                # `a` should be trimmed first
                first_edge, last_edge = _get_outer_edges(a, range)
                a = a[(a >= first_edge) & (a <= last_edge)]
            shape = (np.nan,)
        elif mt.ndim(bins) == 0:
            try:
                n_equal_bins = operator.index(bins)
            except TypeError:  # pragma: no cover
                raise TypeError(
                    '`bins` must be an integer, a string, or an array')
            if n_equal_bins < 1:
                raise ValueError('`bins` must be positive, when an integer')
            shape = (bins + 1,)
        elif mt.ndim(bins) == 1:
            if not isinstance(bins, TENSOR_TYPE):
                bins = np.asarray(bins)
                if not is_asc_sorted(bins):
                    raise ValueError(
                        '`bins` must increase monotonically, when an array')
            shape = astensor(bins).shape
        else:
            raise ValueError('`bins` must be 1d, when an array')

        inputs = [a]
        if isinstance(bins, TENSOR_TYPE):
            inputs.append(bins)
        if weights is not None:
            inputs.append(weights)

        return self.new_tensor(inputs, shape=shape, order=TensorOrder.C_ORDER)

    @classmethod
    def tile(cls, op):
        ctx = get_context()
        a = op.input
        range_ = op.range
        bins = op.bins

        if isinstance(bins, str):
            if has_unknown_shape(op.input):
                yield
        if (a.size > 0 or np.isnan(a.size)) and \
                (isinstance(bins, str) or mt.ndim(bins) == 0) and not range_:
            input_min = a.min(keepdims=True)
            input_max = a.max(keepdims=True)
            input_min, input_max = yield from recursive_tile(
                input_min, input_max)
            chunks = [input_min.chunks[0], input_max.chunks[0]]
            yield chunks
            range_results = ctx.get_chunks_result([c.key for c in chunks])
            # make sure returned bounds are valid
            if all(x.size > 0 for x in range_results):
                range_ = tuple(x[0] for x in range_results)
        if isinstance(bins, TENSOR_TYPE):
            # `bins` is a Tensor, needs to be calculated first
            yield
            bin_datas = ctx.get_chunks_result([c.key for c in bins.chunks])
            bins = np.concatenate(bin_datas)
        else:
            bins = op.bins

        bin_edges, _ = yield from _get_bin_edges(
            op, op.input, bins, range_, op.weights)
        bin_edges = yield from recursive_tile(bin_edges)
        return [bin_edges]


def histogram_bin_edges(a, bins=10, range=None, weights=None):
    r"""
    Function to calculate only the edges of the bins used by the `histogram`
    function.

    Parameters
    ----------
    a : array_like
        Input data. The histogram is computed over the flattened tensor.
    bins : int or sequence of scalars or str, optional
        If `bins` is an int, it defines the number of equal-width
        bins in the given range (10, by default). If `bins` is a
        sequence, it defines the bin edges, including the rightmost
        edge, allowing for non-uniform bin widths.

        If `bins` is a string from the list below, `histogram_bin_edges` will use
        the method chosen to calculate the optimal bin width and
        consequently the number of bins (see `Notes` for more detail on
        the estimators) from the data that falls within the requested
        range. While the bin width will be optimal for the actual data
        in the range, the number of bins will be computed to fill the
        entire range, including the empty portions. For visualisation,
        using the 'auto' option is suggested. Weighted data is not
        supported for automated bin size selection.

        'auto'
            Maximum of the 'sturges' and 'fd' estimators. Provides good
            all around performance.

        'fd' (Freedman Diaconis Estimator)
            Robust (resilient to outliers) estimator that takes into
            account data variability and data size.

        'doane'
            An improved version of Sturges' estimator that works better
            with non-normal datasets.

        'scott'
            Less robust estimator that that takes into account data
            variability and data size.

        'stone'
            Estimator based on leave-one-out cross-validation estimate of
            the integrated squared error. Can be regarded as a generalization
            of Scott's rule.

        'rice'
            Estimator does not take variability into account, only data
            size. Commonly overestimates number of bins required.

        'sturges'
            R's default method, only accounts for data size. Only
            optimal for gaussian data and underestimates number of bins
            for large non-gaussian datasets.

        'sqrt'
            Square root (of data size) estimator, used by Excel and
            other programs for its speed and simplicity.

    range : (float, float), optional
        The lower and upper range of the bins.  If not provided, range
        is simply ``(a.min(), a.max())``.  Values outside the range are
        ignored. The first element of the range must be less than or
        equal to the second. `range` affects the automatic bin
        computation as well. While bin width is computed to be optimal
        based on the actual data within `range`, the bin count will fill
        the entire range including portions containing no data.

    weights : array_like, optional
        A tensor of weights, of the same shape as `a`.  Each value in
        `a` only contributes its associated weight towards the bin count
        (instead of 1). This is currently not used by any of the bin estimators,
        but may be in the future.

    Returns
    -------
    bin_edges : tensor of dtype float
        The edges to pass into `histogram`

    See Also
    --------
    histogram

    Notes
    -----
    The methods to estimate the optimal number of bins are well founded
    in literature, and are inspired by the choices R provides for
    histogram visualisation. Note that having the number of bins
    proportional to :math:`n^{1/3}` is asymptotically optimal, which is
    why it appears in most estimators. These are simply plug-in methods
    that give good starting points for number of bins. In the equations
    below, :math:`h` is the binwidth and :math:`n_h` is the number of
    bins. All estimators that compute bin counts are recast to bin width
    using the `ptp` of the data. The final bin count is obtained from
    ``np.round(np.ceil(range / h))``.

    'auto' (maximum of the 'sturges' and 'fd' estimators)
        A compromise to get a good value. For small datasets the Sturges
        value will usually be chosen, while larger datasets will usually
        default to FD.  Avoids the overly conservative behaviour of FD
        and Sturges for small and large datasets respectively.
        Switchover point is usually :math:`a.size \approx 1000`.

    'fd' (Freedman Diaconis Estimator)
        .. math:: h = 2 \frac{IQR}{n^{1/3}}

        The binwidth is proportional to the interquartile range (IQR)
        and inversely proportional to cube root of a.size. Can be too
        conservative for small datasets, but is quite good for large
        datasets. The IQR is very robust to outliers.

    'scott'
        .. math:: h = \sigma \sqrt[3]{\frac{24 * \sqrt{\pi}}{n}}

        The binwidth is proportional to the standard deviation of the
        data and inversely proportional to cube root of ``x.size``. Can
        be too conservative for small datasets, but is quite good for
        large datasets. The standard deviation is not very robust to
        outliers. Values are very similar to the Freedman-Diaconis
        estimator in the absence of outliers.

    'rice'
        .. math:: n_h = 2n^{1/3}

        The number of bins is only proportional to cube root of
        ``a.size``. It tends to overestimate the number of bins and it
        does not take into account data variability.

    'sturges'
        .. math:: n_h = \log _{2}n+1

        The number of bins is the base 2 log of ``a.size``.  This
        estimator assumes normality of data and is too conservative for
        larger, non-normal datasets. This is the default method in R's
        ``hist`` method.

    'doane'
        .. math:: n_h = 1 + \log_{2}(n) +
                        \log_{2}(1 + \frac{|g_1|}{\sigma_{g_1}})

            g_1 = mean[(\frac{x - \mu}{\sigma})^3]

            \sigma_{g_1} = \sqrt{\frac{6(n - 2)}{(n + 1)(n + 3)}}

        An improved version of Sturges' formula that produces better
        estimates for non-normal datasets. This estimator attempts to
        account for the skew of the data.

    'sqrt'
        .. math:: n_h = \sqrt n

        The simplest and fastest estimator. Only takes into account the
        data size.

    Examples
    --------
    >>> import mars.tensor as mt
    >>> arr = mt.array([0, 0, 0, 1, 2, 3, 3, 4, 5])
    >>> mt.histogram_bin_edges(arr, bins='auto', range=(0, 1)).execute()
    array([0.  , 0.25, 0.5 , 0.75, 1.  ])
    >>> mt.histogram_bin_edges(arr, bins=2).execute()
    array([0. , 2.5, 5. ])

    For consistency with histogram, a tensor of pre-computed bins is
    passed through unmodified:

    >>> mt.histogram_bin_edges(arr, [1, 2]).execute()
    array([1, 2])

    This function allows one set of bins to be computed, and reused across
    multiple histograms:

    >>> shared_bins = mt.histogram_bin_edges(arr, bins='auto')
    >>> shared_bins.execute()
    array([0., 1., 2., 3., 4., 5.])

    >>> group_id = mt.array([0, 1, 1, 0, 1, 1, 0, 1, 1])
    >>> a = arr[group_id == 0]
    >>> a.execute()
    array([0, 1, 3])
    >>> hist_0, _ = mt.histogram(a, bins=shared_bins).execute()
    >>> b = arr[group_id == 1]
    >>> b.execute()
    array([0, 0, 2, 3, 4, 5])
    >>> hist_1, _ = mt.histogram(b, bins=shared_bins).execute()

    >>> hist_0; hist_1
    array([1, 1, 0, 1, 0])
    array([2, 0, 1, 1, 2])

    Which gives more easily comparable results than using separate bins for
    each histogram:

    >>> hist_0, bins_0 = mt.histogram(a, bins='auto').execute()
    >>> hist_1, bins_1 = mt.histogram(b, bins='auto').execute()
    >>> hist_0; hist_1
    array([1, 1, 1])
    array([2, 1, 1, 2])
    >>> bins_0; bins_1
    array([0., 1., 2., 3.])
    array([0.  , 1.25, 2.5 , 3.75, 5.  ])

    """
    a, weights = _ravel_and_check_weights(a, weights)
    op = TensorHistogramBinEdges(input=a, bins=bins,
                                 range=range, weights=weights,
                                 dtype=a.dtype)
    return op(a, bins, range, weights)


class TensorHistogram(TensorOperand, TensorOperandMixin):
    _op_type_ = OperandDef.HISTOGRAM

    _input = KeyField('input')
    _bins = AnyField('bins')
    _range = TupleField('range')
    _weights = KeyField('weights')
    _density = BoolField('density')
    _ret_bins = BoolField('ret_bins')

    def __init__(self, input=None, bins=None, range=None, weights=None,
                 density=None, ret_bins=None, **kw):
        super().__init__(_input=input, _bins=bins, _range=range, _weights=weights,
                         _density=density, _ret_bins=ret_bins, **kw)

    @property
    def input(self):
        return self._input

    @property
    def bins(self):
        return self._bins

    @property
    def range(self):
        return self._range

    @property
    def weights(self):
        return self._weights

    @property
    def density(self):
        return self._density

    @property
    def ret_bins(self):
        return self._ret_bins

    @property
    def output_limit(self):
        return 1 if not self._ret_bins else 2

    def _set_inputs(self, inputs):
        super()._set_inputs(inputs)
        inputs_iter = iter(self._inputs)
        self._input = next(inputs_iter)
        if isinstance(self._bins, (TENSOR_TYPE, TENSOR_CHUNK_TYPE)):
            self._bins = next(inputs_iter)
        if self._weights is not None:
            self._weights = next(inputs_iter)

    def __call__(self, a, bins, range, weights):
        a, weights = _ravel_and_check_weights(a, weights)
        histogram_bin_edges_op = TensorHistogramBinEdges(
            input=a, bins=bins, range=range, weights=weights,
            dtype=np.dtype(np.float64))
        bins = self._bins = histogram_bin_edges_op(a, bins, range, weights)

        inputs = [histogram_bin_edges_op.input]
        if isinstance(bins, TENSOR_TYPE):
            inputs.append(bins)
        # Histogram is an integer or a float array depending on the weights.
        if weights is None:
            dtype = np.dtype(np.intp)
        else:
            inputs.append(weights)
            dtype = weights.dtype
        self.dtype = dtype

        hist = self.new_tensor(inputs, shape=(bins.size - 1,),
                               order=TensorOrder.C_ORDER)
        return mt.ExecutableTuple([hist, bins])

    @classmethod
    def tile(cls, op):
        bins = op.bins.rechunk(op.bins.shape)
        shape = (bins.size - 1,)
        out = op.outputs[0]
        weights = None
        if op.weights is not None:
            # make input and weights have the same nsplits
            weights = yield from recursive_tile(
                op.weights.rechunk(op.input.nsplits))

        out_chunks = []
        for chunk in op.input.chunks:
            chunk_op = op.copy().reset_key()
            chunk_op._range = None
            chunk_op._ret_bins = False
            chunk_op._density = False
            chunk_inputs = [chunk, bins.chunks[0]]
            if weights is not None:
                weights_chunk = weights.cix[chunk.index]
                chunk_inputs.append(weights_chunk)
            out_chunk = chunk_op.new_chunk(chunk_inputs, shape=shape,
                                           index=chunk.index, order=out.order)
            out_chunks.append(out_chunk)

        # merge chunks together
        chunk = chunk_tree_add(out.dtype, out_chunks, (0,), shape)
        new_op = op.copy()
        n = new_op.new_tensor(op.inputs, shape=shape, order=out.order,
                              chunks=[chunk], nsplits=((shape[0],),))
        if op.density:
            db = mt.array(mt.diff(bins), float)
            hist = n / db / n.sum()
            hist = yield from recursive_tile(hist)
            return [hist]
        else:
            return [n]

    @classmethod
    def execute(cls, ctx, op):
        inputs, device_id, xp = as_same_device(
            [ctx[inp.key] for inp in op.inputs], device=op.device, ret_extra=True)
        a = inputs[0]
        bins = inputs[1] if isinstance(op.bins, TENSOR_CHUNK_TYPE) else op.bins
        weights = None
        if op.weights is not None:
            weights = inputs[-1]
        with device(device_id):
            hist, bin_edges = xp.histogram(a, bins=bins, range=op.range,
                                           weights=weights, density=op.density)
            ctx[op.outputs[0].key] = hist
            if op.ret_bins:
                ctx[op.outputs[1].key] = bin_edges


def histogram(a, bins=10, range=None, weights=None, density=None):
    r"""
    Compute the histogram of a set of data.

    Parameters
    ----------
    a : array_like
        Input data. The histogram is computed over the flattened tensor.
    bins : int or sequence of scalars or str, optional
        If `bins` is an int, it defines the number of equal-width
        bins in the given range (10, by default). If `bins` is a
        sequence, it defines a monotonically increasing tensor of bin edges,
        including the rightmost edge, allowing for non-uniform bin widths.

        If `bins` is a string, it defines the method used to calculate the
        optimal bin width, as defined by `histogram_bin_edges`.

    range : (float, float), optional
        The lower and upper range of the bins.  If not provided, range
        is simply ``(a.min(), a.max())``.  Values outside the range are
        ignored. The first element of the range must be less than or
        equal to the second. `range` affects the automatic bin
        computation as well. While bin width is computed to be optimal
        based on the actual data within `range`, the bin count will fill
        the entire range including portions containing no data.

    weights : array_like, optional
        A tensor of weights, of the same shape as `a`.  Each value in
        `a` only contributes its associated weight towards the bin count
        (instead of 1). If `density` is True, the weights are
        normalized, so that the integral of the density over the range
        remains 1.
    density : bool, optional
        If ``False``, the result will contain the number of samples in
        each bin. If ``True``, the result is the value of the
        probability *density* function at the bin, normalized such that
        the *integral* over the range is 1. Note that the sum of the
        histogram values will not be equal to 1 unless bins of unity
        width are chosen; it is not a probability *mass* function.

        Overrides the ``normed`` keyword if given.

    Returns
    -------
    hist : tensor
        The values of the histogram. See `density` and `weights` for a
        description of the possible semantics.
    bin_edges : tensor of dtype float
        Return the bin edges ``(length(hist)+1)``.


    See Also
    --------
    histogramdd, bincount, searchsorted, digitize, histogram_bin_edges

    Notes
    -----
    All but the last (righthand-most) bin is half-open.  In other words,
    if `bins` is::

      [1, 2, 3, 4]

    then the first bin is ``[1, 2)`` (including 1, but excluding 2) and
    the second ``[2, 3)``.  The last bin, however, is ``[3, 4]``, which
    *includes* 4.


    Examples
    --------
    >>> import mars.tensor as mt
    >>> mt.histogram([1, 2, 1], bins=[0, 1, 2, 3]).execute()
    (array([0, 2, 1]), array([0, 1, 2, 3]))
    >>> mt.histogram(mt.arange(4), bins=mt.arange(5), density=True).execute()
    (array([0.25, 0.25, 0.25, 0.25]), array([0, 1, 2, 3, 4]))
    >>> mt.histogram([[1, 2, 1], [1, 0, 1]], bins=[0,1,2,3]).execute()
    (array([1, 4, 1]), array([0, 1, 2, 3]))

    >>> a = mt.arange(5)
    >>> hist, bin_edges = mt.histogram(a, density=True)
    >>> hist.execute()
    array([0.5, 0. , 0.5, 0. , 0. , 0.5, 0. , 0.5, 0. , 0.5])
    >>> hist.sum().execute()
    2.4999999999999996
    >>> mt.sum(hist * mt.diff(bin_edges)).execute()
    1.0

    Automated Bin Selection Methods example, using 2 peak random data
    with 2000 points:

    >>> import matplotlib.pyplot as plt
    >>> rng = mt.random.RandomState(10)  # deterministic random data
    >>> a = mt.hstack((rng.normal(size=1000),
    ...                rng.normal(loc=5, scale=2, size=1000)))
    >>> _ = plt.hist(np.asarray(a), bins='auto')  # arguments are passed to np.histogram
    >>> plt.title("Histogram with 'auto' bins")
    Text(0.5, 1.0, "Histogram with 'auto' bins")
    >>> plt.show()

    """
    a, weights = _ravel_and_check_weights(a, weights)
    op = TensorHistogram(input=a, bins=bins, range=range,
                         weights=weights, density=density)
    return op(a, bins, range, weights)
