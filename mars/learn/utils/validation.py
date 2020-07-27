# Copyright 1999-2020 Alibaba Group Holding Ltd.
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
import numbers

import numpy as np
from numpy.core.numeric import ComplexWarning
try:
    from sklearn.utils.validation import check_is_fitted
    from sklearn.exceptions import DataConversionWarning
except ImportError:  # pragma: no cover
    check_is_fitted = None
    DataConversionWarning = UserWarning

from ... import dataframe as md
from ... import tensor as mt
from ...core import ExecutableTuple
from ...lib.sparse import issparse
from ...tensor import Tensor
from .checks import check_non_negative_then_return_value, \
    assert_all_finite, AssertAllFinite


# ---------------------------------------------------------
# Original implementation is in `sklearn.utils.validation`.
# ---------------------------------------------------------

assert_all_finite = _assert_all_finite = assert_all_finite


def _num_samples(x):
    """Return number of samples in array-like x."""
    if hasattr(x, 'fit') and callable(x.fit):
        # Don't get num_samples from an ensembles length!
        raise TypeError('Expected sequence or array-like, got '
                        'estimator %s' % x)
    if not hasattr(x, '__len__') and not hasattr(x, 'shape'):
        if hasattr(x, '__array__'):
            x = mt.asarray(x)
        else:
            raise TypeError("Expected sequence or array-like, got %s" %
                            type(x))
    if hasattr(x, 'shape'):
        if len(x.shape) == 0:
            if isinstance(x.op, AssertAllFinite):
                x = x.op.x
            if hasattr(x.op, 'data') and x.op.data is not None:
                x = np.asarray(x.op.data)
            raise TypeError("Singleton array %r cannot be considered"
                            " a valid collection." % x)
        # Check that shape is returning an integer or default to len
        if isinstance(x.shape[0], numbers.Integral):
            return x.shape[0]
        elif np.isnan(x.shape[0]):
            return x.shape[0]
        else:
            return len(x)
    else:
        return len(x)


def check_consistent_length(*arrays, session=None, run_kwargs=None):
    """Check that all arrays have consistent first dimensions.

    Checks whether all objects in arrays have the same shape or length.

    Parameters
    ----------
    *arrays : list or tuple of input objects.
        Objects that will be checked for consistent length.
    """

    new_arrays = []
    lengths = []
    to_execute = []
    for X in arrays:
        if X is not None:
            n = _num_samples(X)
            if np.isnan(n):
                to_execute.append(X)
            new_arrays.append(X)
            lengths.append(n)
    # unknown length exists
    if len(to_execute) > 0:
        # update shape
        ExecutableTuple(to_execute).execute(session=session,
                                            **(run_kwargs or dict()))
        # get length again
        lengths = [_num_samples(X) for X in new_arrays]

    uniques = np.unique(lengths)
    if len(uniques) > 1:
        raise ValueError("Found input variables with inconsistent numbers of"
                         " samples: %r" % [int(length) for length in lengths])


def _make_indexable(iterable):
    """Ensure iterable supports indexing or convert to an indexable variant.

    Convert sparse matrices to csr and other non-indexable iterable to arrays.
    Let `None` and indexable objects (e.g. pandas dataframes) pass unchanged.

    Parameters
    ----------
    iterable : {list, dataframe, array, sparse} or None
        Object to be converted to an indexable iterable.
    """
    if issparse(iterable):
        return mt.tensor(iterable)
    elif hasattr(iterable, "iloc"):
        return md.DataFrame(iterable)
    elif hasattr(iterable, "__getitem__"):
        return mt.tensor(iterable)
    elif iterable is None:
        return iterable
    return mt.tensor(iterable)


def indexable(*iterables, session=None, run_kwargs=None):
    """Make arrays indexable for cross-validation.

    Checks consistent length, passes through None, and ensures that everything
    can be indexed by converting sparse matrices to csr and converting
    non-interable objects to arrays.

    Parameters
    ----------
    *iterables : lists, dataframes, arrays, sparse matrices
        List of objects to ensure sliceability.
    """
    result = [_make_indexable(X) for X in iterables]
    check_consistent_length(*result, session=session,
                            run_kwargs=run_kwargs)
    return result


def _ensure_no_complex_data(array):
    if hasattr(array, 'dtype') and array.dtype is not None \
            and hasattr(array.dtype, 'kind') and array.dtype.kind == "c":
        raise ValueError("Complex data not supported\n"
                         "{}\n".format(array))


def _ensure_sparse_format(spmatrix, accept_sparse, dtype, copy,
                          force_all_finite, accept_large_sparse):
    """Convert a sparse matrix to a given format.

    Checks the sparse format of spmatrix and converts if necessary.

    Parameters
    ----------
    spmatrix : scipy sparse matrix
        Input to validate and convert.

    accept_sparse : string, boolean or list/tuple of strings
        String[s] representing allowed sparse matrix formats ('csc',
        'csr', 'coo', 'dok', 'bsr', 'lil', 'dia'). If the input is sparse but
        not in the allowed format, it will be converted to the first listed
        format. True allows the input to be any format. False means
        that a sparse matrix input will raise an error.

    dtype : string, type or None
        Data type of result. If None, the dtype of the input is preserved.

    copy : boolean
        Whether a forced copy will be triggered. If copy=False, a copy might
        be triggered by a conversion.

    force_all_finite : boolean or 'allow-nan', (default=True)
        Whether to raise an error on np.inf and np.nan in X. The possibilities
        are:

        - True: Force all values of X to be finite.
        - False: accept both np.inf and np.nan in X.
        - 'allow-nan': accept only np.nan values in X. Values cannot be
          infinite.

    Returns
    -------
    spmatrix_converted : scipy sparse matrix.
        Matrix that is ensured to have an allowed type.
    """
    if dtype is None:
        dtype = spmatrix.dtype

    changed_format = False

    if isinstance(accept_sparse, str):
        accept_sparse = [accept_sparse]

    # Indices dtype validation
    # _check_large_sparse(spmatrix, accept_large_sparse)

    if accept_sparse is False:
        raise TypeError('A sparse tensor was passed, but dense '
                        'data is required. Use X.todense() to '
                        'convert to a dense tensor.')
    elif isinstance(accept_sparse, (list, tuple)):
        if len(accept_sparse) == 0:
            raise ValueError("When providing 'accept_sparse' "
                             "as a tuple or list, it must contain at "
                             "least one string value.")
        # # ensure correct sparse format
        # if spmatrix.format not in accept_sparse:
        #     # create new with correct sparse
        #     spmatrix = spmatrix.asformat(accept_sparse[0])
        #     changed_format = True
    elif accept_sparse is not True:
        # any other type
        raise ValueError("Parameter 'accept_sparse' should be a string, "
                         "boolean or list of strings. You provided "
                         "'accept_sparse={}'.".format(accept_sparse))

    if dtype != spmatrix.dtype:
        # convert dtype
        spmatrix = spmatrix.astype(dtype)
    elif copy and not changed_format:
        # force copy
        spmatrix = spmatrix.copy()

    if force_all_finite:
        spmatrix = assert_all_finite(
            spmatrix, allow_nan=force_all_finite == 'allow-nan', check_only=False)

    return spmatrix


def check_array(array, accept_sparse=False, accept_large_sparse=True,
                dtype="numeric", order=None, copy=False, force_all_finite=True,
                ensure_2d=True, allow_nd=False, ensure_min_samples=1,
                ensure_min_features=1, estimator=None) -> Tensor:

    """Input validation on a tensor, list, sparse matrix or similar.

    By default, the input is checked to be a non-empty 2D array containing
    only finite values. If the dtype of the tensor is object, attempt
    converting to float, raising on failure.

    Parameters
    ----------
    array : object
        Input object to check / convert.

    accept_sparse : string, boolean or list/tuple of strings (default=False)
        String[s] representing allowed sparse matrix formats, such as 'csc',
        'csr', etc. If the input is sparse but not in the allowed format,
        it will be converted to the first listed format. True allows the input
        to be any format. False means that a sparse matrix input will
        raise an error.

    accept_large_sparse : bool (default=True)
        If a CSR, CSC, COO or BSR sparse matrix is supplied and accepted by
        accept_sparse, accept_large_sparse=False will cause it to be accepted
        only if its indices are stored with a 32-bit dtype.

    dtype : string, type, list of types or None (default="numeric")
        Data type of result. If None, the dtype of the input is preserved.
        If "numeric", dtype is preserved unless array.dtype is object.
        If dtype is a list of types, conversion on the first type is only
        performed if the dtype of the input is not in the list.

    order : 'F', 'C' or None (default=None)
        Whether a tenor will be forced to be fortran or c-style.
        When order is None (default), then if copy=False, nothing is ensured
        about the memory layout of the output tensor; otherwise (copy=True)
        the memory layout of the returned tensor is kept as close as possible
        to the original tensor.

    copy : boolean (default=False)
        Whether a forced copy will be triggered. If copy=False, a copy might
        be triggered by a conversion.

    force_all_finite : boolean or 'allow-nan', (default=True)
        Whether to raise an error on np.inf and np.nan in tensor. The
        possibilities are:

        - True: Force all values of tensor to be finite.
        - False: accept both np.inf and np.nan in tensor.
        - 'allow-nan': accept only np.nan values in tensor. Values cannot
          be infinite.

        For object dtyped data, only np.nan is checked and not np.inf.

    ensure_2d : boolean (default=True)
        Whether to raise a value error if tensor is not 2D.

    allow_nd : boolean (default=False)
        Whether to allow tensor.ndim > 2.

    ensure_min_samples : int (default=1)
        Make sure that the tensor has a minimum number of samples in its first
        axis (rows for a 2D tensor). Setting to 0 disables this check.

    ensure_min_features : int (default=1)
        Make sure that the 2D tensor has some minimum number of features
        (columns). The default value of 1 rejects empty datasets.
        This check is only enforced when the input data has effectively 2
        dimensions or is originally 1D and ``ensure_2d`` is True. Setting to 0
        disables this check.

    estimator : str or estimator instance (default=None)
        If passed, include the name of the estimator in warning messages.

    Returns
    -------
    array_converted : object
        The converted and validated tensor.
    """

    # store whether originally we wanted numeric dtype
    dtype_numeric = isinstance(dtype, str) and dtype == "numeric"

    dtype_orig = getattr(array, "dtype", None)
    if not hasattr(dtype_orig, 'kind'):
        # not a data type (e.g. a column named dtype in a pandas DataFrame)
        dtype_orig = None

    if dtype_numeric:
        if dtype_orig is not None and dtype_orig.kind == "O":
            # if input is object, convert to float.
            dtype = np.float64
        else:
            dtype = None

    if isinstance(dtype, (list, tuple)):
        if dtype_orig is not None and dtype_orig in dtype:
            # no dtype conversion required
            dtype = None
        else:
            # dtype conversion required. Let's select the first element of the
            # list of accepted types.
            dtype = dtype[0]

    if force_all_finite not in (True, False, 'allow-nan'):
        raise ValueError('force_all_finite should be a bool or "allow-nan"'
                         '. Got {!r} instead'.format(force_all_finite))

    if estimator is not None:
        if isinstance(estimator, str):
            estimator_name = estimator
        else:
            estimator_name = estimator.__class__.__name__
    else:
        estimator_name = "Estimator"
    context = " by %s" % estimator_name if estimator is not None else ""

    if (hasattr(array, 'issparse') and array.issparse()) or issparse(array):
        _ensure_no_complex_data(array)
        array = mt.asarray(array)
        array = _ensure_sparse_format(array, accept_sparse=accept_sparse,
                                      dtype=dtype, copy=copy,
                                      force_all_finite=force_all_finite,
                                      accept_large_sparse=accept_large_sparse)
    else:
        # If np.array(..) gives ComplexWarning, then we convert the warning
        # to an error. This is needed because specifying a non complex
        # dtype to the function converts complex to real dtype,
        # thereby passing the test made in the lines following the scope
        # of warnings context manager.
        with warnings.catch_warnings():
            try:
                warnings.simplefilter('error', ComplexWarning)
                array = mt.asarray(array, dtype=dtype, order=order)
            except ComplexWarning:
                raise ValueError("Complex data not supported\n"
                                 "{}\n".format(array))

        # It is possible that the np.array(..) gave no warning. This happens
        # when no dtype conversion happened, for example dtype = None. The
        # result is that np.array(..) produces an array of complex dtype
        # and we need to catch and raise exception for such cases.
        _ensure_no_complex_data(array)

        if ensure_2d:
            # If input is scalar raise error
            if array.ndim == 0:
                raise ValueError(
                    "Expected 2D array, got scalar array instead:\narray={}.\n"
                    "Reshape your data either using array.reshape(-1, 1) if "
                    "your data has a single feature or array.reshape(1, -1) "
                    "if it contains a single sample.".format(array))
            # If input is 1D raise error
            if array.ndim == 1:
                raise ValueError(
                    "Expected 2D array, got 1D array instead:\narray={}.\n"
                    "Reshape your data either using array.reshape(-1, 1) if "
                    "your data has a single feature or array.reshape(1, -1) "
                    "if it contains a single sample.".format(array))

        # in the future np.flexible dtypes will be handled like object dtypes
        if dtype_numeric and np.issubdtype(array.dtype, np.flexible):
            warnings.warn(
                "Beginning in version 0.22, arrays of bytes/strings will be "
                "converted to decimal numbers if dtype='numeric'. "
                "It is recommended that you convert the array to "
                "a float dtype before using it in scikit-learn, "
                "for example by using "
                "your_array = your_array.astype(np.float64).",
                FutureWarning)

        # make sure we actually converted to numeric:
        if dtype_numeric and array.dtype.kind == "O":
            array = array.astype(np.float64)
        if not allow_nd and array.ndim >= 3:
            raise ValueError("Found array with dim %d. %s expected <= 2."
                             % (array.ndim, estimator_name))
        if force_all_finite:
            array = _assert_all_finite(
                array, allow_nan=force_all_finite == 'allow-nan', check_only=False)

    if ensure_min_samples > 0:
        n_samples = _num_samples(array)
        if n_samples < ensure_min_samples:
            raise ValueError("Found array with %d sample(s) (shape=%s) while a"
                             " minimum of %d is required%s."
                             % (n_samples, array.shape, ensure_min_samples,
                                context))

    if ensure_min_features > 0 and array.ndim == 2:
        n_features = array.shape[1]
        if n_features < ensure_min_features:
            raise ValueError("Found array with %d feature(s) (shape=%s) while"
                             " a minimum of %d is required%s."
                             % (n_features, array.shape, ensure_min_features,
                                context))

    if copy:
        array = mt.array(array, dtype=dtype, order=order)

    return array


def check_X_y(X, y, accept_sparse=False, accept_large_sparse=True,
              dtype="numeric", order=None, copy=False, force_all_finite=True,
              ensure_2d=True, allow_nd=False, multi_output=False,
              ensure_min_samples=1, ensure_min_features=1, y_numeric=False,
              estimator=None):
    """Input validation for standard estimators.

    Checks X and y for consistent length, enforces X to be 2D and y 1D. By
    default, X is checked to be non-empty and containing only finite values.
    Standard input checks are also applied to y, such as checking that y
    does not have np.nan or np.inf targets. For multi-label y, set
    multi_output=True to allow 2D and sparse y. If the dtype of X is
    object, attempt converting to float, raising on failure.

    Parameters
    ----------
    X : tensor, list or sparse tensor
        Input data.

    y : tensor, list or sparse tensor
        Labels.

    accept_sparse : string, boolean or list of string (default=False)
        String[s] representing allowed sparse matrix formats, such as 'csc',
        'csr', etc. If the input is sparse but not in the allowed format,
        it will be converted to the first listed format. True allows the input
        to be any format. False means that a sparse matrix input will
        raise an error.

    accept_large_sparse : bool (default=True)
        If a CSR, CSC, COO or BSR sparse matrix is supplied and accepted by
        accept_sparse, accept_large_sparse will cause it to be accepted only
        if its indices are stored with a 32-bit dtype.

    dtype : string, type, list of types or None (default="numeric")
        Data type of result. If None, the dtype of the input is preserved.
        If "numeric", dtype is preserved unless array.dtype is object.
        If dtype is a list of types, conversion on the first type is only
        performed if the dtype of the input is not in the list.

    order : 'F', 'C' or None (default=None)
        Whether an array will be forced to be fortran or c-style.

    copy : boolean (default=False)
        Whether a forced copy will be triggered. If copy=False, a copy might
        be triggered by a conversion.

    force_all_finite : boolean or 'allow-nan', (default=True)
        Whether to raise an error on np.inf and np.nan in X. This parameter
        does not influence whether y can have np.inf or np.nan values.
        The possibilities are:

        - True: Force all values of X to be finite.
        - False: accept both np.inf and np.nan in X.
        - 'allow-nan': accept only np.nan values in X. Values cannot be
          infinite.

    ensure_2d : boolean (default=True)
        Whether to raise a value error if X is not 2D.

    allow_nd : boolean (default=False)
        Whether to allow X.ndim > 2.

    multi_output : boolean (default=False)
        Whether to allow 2D y (array or sparse matrix). If false, y will be
        validated as a vector. y cannot have np.nan or np.inf values if
        multi_output=True.

    ensure_min_samples : int (default=1)
        Make sure that X has a minimum number of samples in its first
        axis (rows for a 2D array).

    ensure_min_features : int (default=1)
        Make sure that the 2D array has some minimum number of features
        (columns). The default value of 1 rejects empty datasets.
        This check is only enforced when X has effectively 2 dimensions or
        is originally 1D and ``ensure_2d`` is True. Setting to 0 disables
        this check.

    y_numeric : boolean (default=False)
        Whether to ensure that y has a numeric type. If dtype of y is object,
        it is converted to float64. Should only be used for regression
        algorithms.

    estimator : str or estimator instance (default=None)
        If passed, include the name of the estimator in warning messages.

    Returns
    -------
    X_converted : object
        The converted and validated X.

    y_converted : object
        The converted and validated y.
    """
    if y is None:
        raise ValueError('y cannot be None')

    X = check_array(X, accept_sparse=accept_sparse,
                    accept_large_sparse=accept_large_sparse,
                    dtype=dtype, order=order, copy=copy,
                    force_all_finite=force_all_finite,
                    ensure_2d=ensure_2d, allow_nd=allow_nd,
                    ensure_min_samples=ensure_min_samples,
                    ensure_min_features=ensure_min_features,
                    estimator=estimator)
    if multi_output:
        y = check_array(y, True, force_all_finite=True, ensure_2d=False,
                        dtype=None)
    else:
        y = column_or_1d(y, warn=True)
        y = _assert_all_finite(y, check_only=False)
    if y_numeric and y.dtype.kind == 'O':
        y = y.astype(np.float64)

    check_consistent_length(X, y)

    return X, y


def check_non_negative(X, whom):
    """
    Check if there is any negative value in a tensor.

    Parameters
    ----------
    X : array-like or sparse matrix
        Input data.

    whom : string
        Who passed X to this function.
    """
    return check_non_negative_then_return_value(X, X, whom)


def column_or_1d(y, warn=False):
    """ Ravel column or 1d numpy array, else raises an error

    Parameters
    ----------
    y : array-like

    warn : boolean, default False
       To control display of warnings.

    Returns
    -------
    y : array

    """
    y = mt.tensor(y)
    shape = y.shape
    if len(shape) == 1:
        return mt.ravel(y)
    if len(shape) == 2 and shape[1] == 1:
        if warn:
            warnings.warn("A column-vector y was passed when a 1d array was"
                          " expected. Please change the shape of y to "
                          "(n_samples, ), for example using ravel().",
                          DataConversionWarning, stacklevel=2)
        return mt.ravel(y)

    raise ValueError("bad input shape {0}".format(shape))


check_is_fitted = check_is_fitted


def _check_sample_weight(sample_weight, X, dtype=None):
    """Validate sample weights.

    Note that passing sample_weight=None will output an array of ones.
    Therefore, in some cases, you may want to protect the call with:
    if sample_weight is not None:
        sample_weight = _check_sample_weight(...)

    Parameters
    ----------
    sample_weight : {ndarray, Number or None}, shape (n_samples,)
       Input sample weights.

    X : nd-array, list or sparse matrix
        Input data.

    dtype: dtype
       dtype of the validated `sample_weight`.
       If None, and the input `sample_weight` is an array, the dtype of the
       input is preserved; otherwise an array with the default numpy dtype
       is be allocated.  If `dtype` is not one of `float32`, `float64`,
       `None`, the output will be of dtype `float64`.

    Returns
    -------
    sample_weight : ndarray, shape (n_samples,)
       Validated sample weight. It is guaranteed to be "C" contiguous.
    """
    n_samples = _num_samples(X)

    if dtype is not None and dtype not in [np.float32, np.float64]:
        dtype = np.float64

    if sample_weight is None or isinstance(sample_weight, numbers.Number):
        if sample_weight is None:
            sample_weight = mt.ones(n_samples, dtype=dtype)
        else:
            sample_weight = mt.full(n_samples, sample_weight,
                                    dtype=dtype)
    else:
        if dtype is None:
            dtype = [np.float64, np.float32]
        sample_weight = check_array(
            sample_weight, accept_sparse=False, ensure_2d=False, dtype=dtype,
            order="C"
        )
        if sample_weight.ndim != 1:
            raise ValueError("Sample weights must be 1D array or scalar")

        if sample_weight.shape != (n_samples,):
            raise ValueError("sample_weight.shape == {}, expected {}!"
                             .format(sample_weight.shape, (n_samples,)))
    return sample_weight
