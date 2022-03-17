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

from typing import Union

import numpy as np
import scipy.sparse as sp
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.sparsefuncs import min_max_axis

from ... import execute as execute_tileable, fetch as fetch_tileable
from ... import opcodes
from ... import tensor as mt
from ...core import ENTITY_TYPE, OutputType, recursive_tile
from ...core.context import get_context, Context
from ...lib.sparse import SparseNDArray
from ...serialization.serializables import AnyField, BoolField, Int32Field, StringField
from ...tensor.core import TensorOrder
from ...typing import TileableType
from ...utils import has_unknown_shape
from ..operands import LearnOperand, LearnOperandMixin
from ..utils import column_or_1d
from ..utils._encode import _unique, _encode
from ..utils.multiclass import unique_labels, type_of_target
from ..utils.validation import _num_samples, check_is_fitted, check_array


class LabelEncoder(TransformerMixin, BaseEstimator):
    """Encode target labels with value between 0 and n_classes-1.

    This transformer should be used to encode target values, *i.e.* `y`, and
    not the input `X`.

    Read more in the :ref:`User Guide <preprocessing_targets>`.

    Attributes
    ----------
    classes_ : ndarray of shape (n_classes,)
        Holds the label for each class.

    See Also
    --------
    OrdinalEncoder : Encode categorical features using an ordinal encoding
        scheme.
    OneHotEncoder : Encode categorical features as a one-hot numeric array.

    Examples
    --------
    `LabelEncoder` can be used to normalize labels.

    >>> from sklearn import preprocessing
    >>> le = preprocessing.LabelEncoder()
    >>> le.fit([1, 2, 2, 6])
    LabelEncoder()
    >>> le.classes_
    array([1, 2, 6])
    >>> le.transform([1, 1, 2, 6])
    array([0, 0, 1, 2]...)
    >>> le.inverse_transform([0, 0, 1, 2])
    array([1, 1, 2, 6])

    It can also be used to transform non-numerical labels (as long as they are
    hashable and comparable) to numerical labels.

    >>> le = preprocessing.LabelEncoder()
    >>> le.fit(["paris", "paris", "tokyo", "amsterdam"])
    LabelEncoder()
    >>> list(le.classes_)
    ['amsterdam', 'paris', 'tokyo']
    >>> le.transform(["tokyo", "tokyo", "paris"])
    array([2, 2, 1]...)
    >>> list(le.inverse_transform([2, 2, 1]))
    ['tokyo', 'tokyo', 'paris']
    """

    def fit(self, y, session=None, run_kwargs=None, execute=True):
        """Fit label encoder.

        Parameters
        ----------
        y : array-like of shape (n_samples,)
            Target values.

        Returns
        -------
        self : returns an instance of self.
            Fitted label encoder.
        """
        y = column_or_1d(y, warn=True)
        self.classes_ = _unique(y)
        if execute:
            self.classes_ = execute_tileable(
                self.classes_, session=session, **(run_kwargs or dict())
            )
        return self

    def fit_transform(self, y, session=None, run_kwargs=None):
        """Fit label encoder and return encoded labels.

        Parameters
        ----------
        y : array-like of shape (n_samples,)
            Target values.

        Returns
        -------
        y : array-like of shape (n_samples,)
            Encoded labels.
        """
        y = column_or_1d(y, warn=True)
        self.classes_, y = execute_tileable(
            _unique(y, return_inverse=True), session=session, **(run_kwargs or dict())
        )
        return y

    def transform(self, y, session=None, run_kwargs=None, execute=True):
        """Transform labels to normalized encoding.

        Parameters
        ----------
        y : array-like of shape (n_samples,)
            Target values.

        Returns
        -------
        y : array-like of shape (n_samples,)
            Labels as normalized encodings.
        """
        check_is_fitted(self)
        y = column_or_1d(y, warn=True)
        # transform of empty array is empty array
        if _num_samples(y) == 0:
            return mt.array([])

        t = _encode(y, uniques=self.classes_)
        if execute:
            t = t.execute(session=session, **(run_kwargs or dict()))
        return t

    def inverse_transform(self, y, session=None, run_kwargs=None):
        """Transform labels back to original encoding.

        Parameters
        ----------
        y : ndarray of shape (n_samples,)
            Target values.

        Returns
        -------
        y : ndarray of shape (n_samples,)
            Original encoding.
        """
        check_is_fitted(self)
        y = column_or_1d(y, warn=True)
        # inverse transform of empty array is empty array
        if _num_samples(y) == 0:
            return mt.array([])

        def _class_checker(chunk_data, classes_data):
            diff = np.setdiff1d(chunk_data, np.arange(len(classes_data)))
            if len(diff):
                raise ValueError("y contains previously unseen labels: %s" % str(diff))
            return chunk_data

        y = mt.asarray(y).map_chunk(_class_checker, args=(self.classes_,))
        return self.classes_[y].execute(session=session, **(run_kwargs or dict()))

    def _more_tags(self):
        return {"X_types": ["1dlabels"]}


class LabelBinarizer(TransformerMixin, BaseEstimator):
    """Binarize labels in a one-vs-all fashion.

    Several regression and binary classification algorithms are
    available in scikit-learn. A simple way to extend these algorithms
    to the multi-class classification case is to use the so-called
    one-vs-all scheme.

    At learning time, this simply consists in learning one regressor
    or binary classifier per class. In doing so, one needs to convert
    multi-class labels to binary labels (belong or does not belong
    to the class). LabelBinarizer makes this process easy with the
    transform method.

    At prediction time, one assigns the class for which the corresponding
    model gave the greatest confidence. LabelBinarizer makes this easy
    with the inverse_transform method.

    Read more in the :ref:`User Guide <preprocessing_targets>`.

    Parameters
    ----------

    neg_label : int, default=0
        Value with which negative labels must be encoded.

    pos_label : int, default=1
        Value with which positive labels must be encoded.

    sparse_output : bool, default=False
        True if the returned array from transform is desired to be in sparse
        CSR format.

    Attributes
    ----------

    classes_ : ndarray of shape (n_classes,)
        Holds the label for each class.

    y_type_ : str
        Represents the type of the target data as evaluated by
        utils.multiclass.type_of_target. Possible type are 'continuous',
        'continuous-multioutput', 'binary', 'multiclass',
        'multiclass-multioutput', 'multilabel-indicator', and 'unknown'.

    sparse_input_ : bool
        True if the input data to transform is given as a sparse matrix, False
        otherwise.

    Examples
    --------
    >>> from mars.learn import preprocessing
    >>> lb = preprocessing.LabelBinarizer()
    >>> lb.fit([1, 2, 6, 4, 2])
    LabelBinarizer()
    >>> lb.classes_
    array([1, 2, 4, 6])
    >>> lb.transform([1, 6])
    array([[1, 0, 0, 0],
           [0, 0, 0, 1]])

    Binary targets transform to a column vector

    >>> lb = preprocessing.LabelBinarizer()
    >>> lb.fit_transform(['yes', 'no', 'no', 'yes'])
    array([[1],
           [0],
           [0],
           [1]])

    Passing a 2D matrix for multilabel classification

    >>> import numpy as np
    >>> lb.fit(np.array([[0, 1, 1], [1, 0, 0]]))
    LabelBinarizer()
    >>> lb.classes_
    array([0, 1, 2])
    >>> lb.transform([0, 1, 2, 1])
    array([[1, 0, 0],
           [0, 1, 0],
           [0, 0, 1],
           [0, 1, 0]])

    See Also
    --------
    label_binarize : Function to perform the transform operation of
        LabelBinarizer with fixed classes.
    OneHotEncoder : Encode categorical features using a one-hot aka one-of-K
        scheme.
    """

    def __init__(self, *, neg_label=0, pos_label=1, sparse_output=False):
        if neg_label >= pos_label:
            raise ValueError(
                "neg_label={0} must be strictly less than "
                "pos_label={1}.".format(neg_label, pos_label)
            )

        if sparse_output and (pos_label == 0 or neg_label != 0):
            raise ValueError(
                "Sparse binarization is only supported with non "
                "zero pos_label and zero neg_label, got "
                "pos_label={0} and neg_label={1}"
                "".format(pos_label, neg_label)
            )

        self.neg_label = neg_label
        self.pos_label = pos_label
        self.sparse_output = sparse_output

    def fit(self, y, session=None, run_kwargs=None):
        """Fit label binarizer.

        Parameters
        ----------
        y : ndarray of shape (n_samples,) or (n_samples, n_classes)
            Target values. The 2-d matrix should only contain 0 and 1,
            represents multilabel classification.

        Returns
        -------
        self : returns an instance of self.
        """
        self.y_type_ = fetch_tileable(
            execute_tileable(
                type_of_target(y), session=session, **(run_kwargs or dict())
            )
        )
        if "multioutput" in self.y_type_:
            raise ValueError(
                "Multioutput target data is not supported with " "label binarization"
            )
        if _num_samples(y) == 0:  # pragma: no cover
            raise ValueError("y has 0 samples: %r" % y)

        self.sparse_input_ = mt.tensor(y).issparse()
        self.classes_ = unique_labels(y).execute(
            session=session, **(run_kwargs or dict())
        )
        return self

    def fit_transform(self, y, session=None, run_kwargs=None):
        """Fit label binarizer and transform multi-class labels to binary
        labels.

        The output of transform is sometimes referred to as
        the 1-of-K coding scheme.

        Parameters
        ----------
        y : {ndarray, sparse matrix} of shape (n_samples,) or \
                (n_samples, n_classes)
            Target values. The 2-d matrix should only contain 0 and 1,
            represents multilabel classification. Sparse matrix can be
            CSR, CSC, COO, DOK, or LIL.

        Returns
        -------
        Y : {ndarray, sparse matrix} of shape (n_samples, n_classes)
            Shape will be (n_samples, 1) for binary problems. Sparse matrix
            will be of CSR format.
        """
        return self.fit(y, session=session, run_kwargs=run_kwargs).transform(
            y, session=session, run_kwargs=run_kwargs
        )

    def transform(self, y, session=None, run_kwargs=None):
        """Transform multi-class labels to binary labels.

        The output of transform is sometimes referred to by some authors as
        the 1-of-K coding scheme.

        Parameters
        ----------
        y : {array, sparse matrix} of shape (n_samples,) or \
                (n_samples, n_classes)
            Target values. The 2-d matrix should only contain 0 and 1,
            represents multilabel classification. Sparse matrix can be
            CSR, CSC, COO, DOK, or LIL.

        Returns
        -------
        Y : {ndarray, sparse matrix} of shape (n_samples, n_classes)
            Shape will be (n_samples, 1) for binary problems. Sparse matrix
            will be of CSR format.
        """
        check_is_fitted(self)

        target = fetch_tileable(
            execute_tileable(
                type_of_target(y), session=session, **(run_kwargs or dict())
            )
        )
        y_is_multilabel = target.startswith("multilabel")
        if y_is_multilabel and not self.y_type_.startswith("multilabel"):
            raise ValueError("The object was not fitted with multilabel" " input.")

        return label_binarize(
            y,
            classes=self.classes_,
            pos_label=self.pos_label,
            neg_label=self.neg_label,
            sparse_output=self.sparse_output,
        )

    def inverse_transform(self, Y, threshold=None):
        """Transform binary labels back to multi-class labels.

        Parameters
        ----------
        Y : {ndarray, sparse matrix} of shape (n_samples, n_classes)
            Target values. All sparse matrices are converted to CSR before
            inverse transformation.

        threshold : float, default=None
            Threshold used in the binary and multi-label cases.

            Use 0 when ``Y`` contains the output of decision_function
            (classifier).
            Use 0.5 when ``Y`` contains the output of predict_proba.

            If None, the threshold is assumed to be half way between
            neg_label and pos_label.

        Returns
        -------
        y : {ndarray, sparse matrix} of shape (n_samples,)
            Target values. Sparse matrix will be of CSR format.

        Notes
        -----
        In the case when the binary labels are fractional
        (probabilistic), inverse_transform chooses the class with the
        greatest value. Typically, this allows to use the output of a
        linear model's decision_function method directly as the input
        of inverse_transform.
        """
        check_is_fitted(self)

        if threshold is None:
            threshold = (self.pos_label + self.neg_label) / 2.0

        Y = mt.asarray(Y)
        if self.y_type_ == "multiclass":
            y_inv = Y.map_chunk(
                _inverse_binarize_multiclass,
                args=(self.classes_,),
                dtype=self.classes_.dtype,
                shape=(Y.shape[0],),
            )
        else:
            shape = (Y.shape[0],) if self.y_type_ != "multilabel-indicator" else Y.shape
            y_inv = Y.map_chunk(
                _inverse_binarize_thresholding,
                args=(self.y_type_, self.classes_, threshold),
                dtype=self.classes_.dtype,
                shape=shape,
            )

        if self.sparse_input_:
            y_inv = y_inv.tosparse()
        elif y_inv.issparse():
            y_inv = y_inv.todense()

        return y_inv

    def _more_tags(
        self,
    ):  # pragma: no cover  # noqa: R0201  # pylint: disable=no-self-use
        return {"X_types": ["1dlabels"]}


class LabelBinarize(LearnOperand, LearnOperandMixin):
    _op_type_ = opcodes.LABEL_BINARIZE

    y = AnyField("y")
    classes = AnyField("classes")
    neg_label = Int32Field("neg_label")
    pos_label = Int32Field("pos_label")
    sparse_output = BoolField("sparse_output")
    # for chunk
    y_type = StringField("y_type")
    pos_switch = BoolField("pos_switch")

    def __call__(self, y: TileableType, classes: TileableType):
        inputs = []
        if isinstance(y, ENTITY_TYPE):
            inputs.append(y)
        if isinstance(classes, ENTITY_TYPE):
            inputs.append(classes)
        self.sparse = self.sparse_output
        self.output_types = [OutputType.tensor]
        if len(classes) == 2:
            n_dim1 = 1
        else:
            n_dim1 = len(classes)
        return self.new_tileable(
            inputs,
            shape=(np.nan, n_dim1),
            dtype=np.dtype(int),
            order=TensorOrder.C_ORDER,
        )

    def _set_inputs(self, inputs):
        super()._set_inputs(inputs)
        if isinstance(self.y, ENTITY_TYPE):
            self.y = self._inputs[0]
        if isinstance(self.classes, ENTITY_TYPE):
            self.classes = self._inputs[-1]

    @classmethod
    def tile(cls, op: "LabelBinarize"):
        y = op.y
        classes = op.classes
        neg_label = op.neg_label
        pos_label = op.pos_label
        sparse_output = op.sparse_output
        out = op.outputs[0]
        ctx = get_context()

        if (isinstance(y, ENTITY_TYPE) and has_unknown_shape(y)) or (
            isinstance(classes, ENTITY_TYPE) and has_unknown_shape(classes)
        ):  # pragma: no cover
            yield
        if (
            isinstance(classes, ENTITY_TYPE) and len(classes.chunks) > 1
        ):  # pragma: no cover
            classes = yield from recursive_tile(classes.rechunk(classes.shape))

        if not isinstance(y, list):
            # XXX Workaround that will be removed when list of list format is
            # dropped
            y = check_array(y, accept_sparse=True, ensure_2d=False, dtype=None)
        else:
            if _num_samples(y) == 0:
                raise ValueError("y has 0 samples: %r" % y)

        y = yield from recursive_tile(mt.tensor(y))

        if neg_label >= pos_label:
            raise ValueError(
                "neg_label={0} must be strictly less than "
                "pos_label={1}.".format(neg_label, pos_label)
            )

        if sparse_output and (pos_label == 0 or neg_label != 0):
            raise ValueError(
                "Sparse binarization is only supported with non "
                "zero pos_label and zero neg_label, got "
                "pos_label={0} and neg_label={1}"
                "".format(pos_label, neg_label)
            )

        # To account for pos_label == 0 in the dense case
        pos_switch = pos_label == 0
        if pos_switch:
            pos_label = -neg_label

        y_type = yield from recursive_tile(type_of_target(y))
        yield y_type.chunks
        y_type = ctx.get_chunks_result([y_type.chunks[0].key])[0]
        y_type = y_type.item() if hasattr(y_type, "item") else y_type
        if "multioutput" in y_type:
            raise ValueError(
                "Multioutput target data is not supported with label " "binarization"
            )
        if y_type == "unknown":
            raise ValueError("The type of target data is not known")

        n_samples = mt.tensor(y).shape[0]
        n_classes = len(classes)

        if y_type == "binary":
            if n_classes == 1:
                if sparse_output:
                    return (
                        yield from recursive_tile(
                            mt.zeros((n_samples, 1), dtype=int, sparse=True)
                        )
                    )
                else:
                    Y = mt.zeros((len(y), 1), dtype=int)
                    Y += neg_label
                    return (yield from recursive_tile(Y))
            elif len(classes) >= 3:
                y_type = "multiclass"

        if y_type == "multilabel-indicator":
            y_n_classes = y.shape[1] if hasattr(y, "shape") else len(y[0])
            if mt.tensor(classes).size != y_n_classes:
                raise ValueError(
                    "classes {0} mismatch with the labels {1}"
                    " found in the data".format(classes, unique_labels(y))
                )

        if y_type in ("binary", "multiclass"):
            y = yield from recursive_tile(column_or_1d(y))
            if y_type == "binary":
                out_shape = (n_samples, 1)
            else:
                out_shape = (n_samples, n_classes)
        elif y_type == "multilabel-indicator":
            out_shape = y.shape
        else:
            raise ValueError(
                "%s target data is not supported with label " "binarization" % y_type
            )

        out_chunks = []
        for y_chunk in y.chunks:
            chunk_inputs = [y_chunk]
            classes_chunk = classes
            if isinstance(classes, ENTITY_TYPE):
                chunk_inputs.append(classes.chunks[0])
                classes_chunk = classes.chunks[0]
            chunk_op = LabelBinarize(
                y=y_chunk,
                classes=classes_chunk,
                neg_label=neg_label,
                pos_label=pos_label,
                sparse_output=sparse_output,
                y_type=y_type,
                pos_switch=pos_switch,
                _output_types=op.output_types,
            )
            if len(out_shape) == 2:
                chunk_shape = (y_chunk.shape[0], out_shape[1])
                chunk_index = (y_chunk.index[0], 0)
            else:  # pragma: no cover
                chunk_shape = (y_chunk.shape[0],)
                chunk_index = (y_chunk.index[0],)
            out_chunk = chunk_op.new_chunk(
                chunk_inputs,
                shape=chunk_shape,
                dtype=out.dtype,
                order=out.order,
                index=chunk_index,
            )
            out_chunks.append(out_chunk)

        params = out.params.copy()
        params["chunks"] = out_chunks
        params["shape"] = out_shape
        if len(out_shape) == 2:
            nsplits = (y.nsplits[0], (out_shape[1],))
        else:  # pragma: no cover
            nsplits = (y.nsplits[0],)
        params["nsplits"] = nsplits
        return op.copy().new_tileables(op.inputs, kws=[params])

    @classmethod
    def execute(cls, ctx: Union[dict, Context], op: "LabelBinarize"):
        y = ctx[op.y.key]
        if hasattr(y, "raw"):
            # SparseNDArray
            y = y.raw
        if isinstance(op.classes, ENTITY_TYPE):
            classes = ctx[op.classes.key]
        else:
            classes = op.classes
        y_type = op.y_type
        sparse_output = op.sparse_output
        pos_label = op.pos_label
        neg_label = op.neg_label
        pos_switch = op.pos_switch

        n_samples = y.shape[0] if sp.issparse(y) else len(y)
        n_classes = len(classes)
        sorted_class = np.sort(classes)

        if y_type in ("binary", "multiclass"):
            # pick out the known labels from y
            y_in_classes = np.in1d(y, classes)
            y_seen = y[y_in_classes]
            indices = np.searchsorted(sorted_class, y_seen)
            indptr = np.hstack((0, np.cumsum(y_in_classes)))

            data = np.empty_like(indices)
            data.fill(pos_label)
            Y = sp.csr_matrix((data, indices, indptr), shape=(n_samples, n_classes))
        elif y_type == "multilabel-indicator":
            Y = sp.csr_matrix(y)
            if pos_label != 1:
                data = np.empty_like(Y.data)
                data.fill(pos_label)
                Y.data = data
        else:  # pragma: no cover
            raise ValueError(
                "%s target data is not supported with label " "binarization" % y_type
            )

        if not sparse_output:
            Y = Y.toarray()
            Y = Y.astype(int, copy=False)

            if neg_label != 0:
                Y[Y == 0] = neg_label

            if pos_switch:
                Y[Y == pos_label] = 0
        else:
            Y.data = Y.data.astype(int, copy=False)

        # preserve label ordering
        if np.any(classes != sorted_class):
            indices = np.searchsorted(sorted_class, classes)
            Y = Y[:, indices]

        if y_type == "binary":
            if sparse_output:
                Y = Y.getcol(-1)
            else:
                Y = Y[:, -1].reshape((-1, 1))

        if sp.issparse(Y):
            Y = SparseNDArray(Y)
        ctx[op.outputs[0].key] = Y


def label_binarize(
    y, *, classes, neg_label=0, pos_label=1, sparse_output=False, execute=True
):
    """Binarize labels in a one-vs-all fashion.

    Several regression and binary classification algorithms are
    available in scikit-learn. A simple way to extend these algorithms
    to the multi-class classification case is to use the so-called
    one-vs-all scheme.

    This function makes it possible to compute this transformation for a
    fixed set of class labels known ahead of time.

    Parameters
    ----------
    y : array-like
        Sequence of integer labels or multilabel data to encode.

    classes : array-like of shape (n_classes,)
        Uniquely holds the label for each class.

    neg_label : int, default=0
        Value with which negative labels must be encoded.

    pos_label : int, default=1
        Value with which positive labels must be encoded.

    sparse_output : bool, default=False,
        Set to true if output binary array is desired in CSR sparse format.

    Returns
    -------
    Y : {tensor, sparse tensor} of shape (n_samples, n_classes)
        Shape will be (n_samples, 1) for binary problems.

    Examples
    --------
    >>> from mars.learn.preprocessing import label_binarize
    >>> label_binarize([1, 6], classes=[1, 2, 4, 6])
    array([[1, 0, 0, 0],
           [0, 0, 0, 1]])

    The class ordering is preserved:

    >>> label_binarize([1, 6], classes=[1, 6, 4, 2])
    array([[1, 0, 0, 0],
           [0, 1, 0, 0]])

    Binary targets transform to a column vector

    >>> label_binarize(['yes', 'no', 'no', 'yes'], classes=['no', 'yes'])
    array([[1],
           [0],
           [0],
           [1]])

    See Also
    --------
    LabelBinarizer : Class used to wrap the functionality of label_binarize and
        allow for fitting to classes independently of the transform operation.
    """
    op = LabelBinarize(
        y=y,
        classes=classes,
        neg_label=neg_label,
        pos_label=pos_label,
        sparse_output=sparse_output,
    )
    result = op(y, classes)
    return result.execute() if execute else result


def _inverse_binarize_multiclass(y, classes):  # pragma: no cover
    """Inverse label binarization transformation for multiclass.

    Multiclass uses the maximal score instead of a threshold.
    """
    classes = np.asarray(classes)

    if sp.issparse(y):
        # Find the argmax for each row in y where y is a CSR matrix

        y = y.tocsr()
        n_samples, n_outputs = y.shape
        outputs = np.arange(n_outputs)
        row_max = min_max_axis(y, 1)[1]
        row_nnz = np.diff(y.indptr)

        y_data_repeated_max = np.repeat(row_max, row_nnz)
        # picks out all indices obtaining the maximum per row
        y_i_all_argmax = np.flatnonzero(y_data_repeated_max == y.data)

        # For corner case where last row has a max of 0
        if row_max[-1] == 0:
            y_i_all_argmax = np.append(y_i_all_argmax, [len(y.data)])

        # Gets the index of the first argmax in each row from y_i_all_argmax
        index_first_argmax = np.searchsorted(y_i_all_argmax, y.indptr[:-1])
        # first argmax of each row
        y_ind_ext = np.append(y.indices, [0])
        y_i_argmax = y_ind_ext[y_i_all_argmax[index_first_argmax]]
        # Handle rows of all 0
        y_i_argmax[np.where(row_nnz == 0)[0]] = 0

        # Handles rows with max of 0 that contain negative numbers
        samples = np.arange(n_samples)[(row_nnz > 0) & (row_max.ravel() == 0)]
        for i in samples:
            ind = y.indices[y.indptr[i] : y.indptr[i + 1]]
            y_i_argmax[i] = classes[np.setdiff1d(outputs, ind)][0]

        return classes[y_i_argmax]
    else:
        return classes.take(y.argmax(axis=1), mode="clip")


def _inverse_binarize_thresholding(
    y, output_type, classes, threshold
):  # pragma: no cover
    """Inverse label binarization transformation using thresholding."""

    if output_type == "binary" and y.ndim == 2 and y.shape[1] > 2:
        raise ValueError("output_type='binary', but y.shape = {0}".format(y.shape))

    if output_type != "binary" and y.shape[1] != len(classes):
        raise ValueError(
            "The number of class is not equal to the number of " "dimension of y."
        )

    classes = np.asarray(classes)

    # Perform thresholding
    if sp.issparse(y):
        if threshold > 0:
            if y.format not in ("csr", "csc"):
                y = y.tocsr()
            y.data = np.array(y.data > threshold, dtype=int)
            y.eliminate_zeros()
        else:
            y = np.array(y.toarray() > threshold, dtype=int)
    else:
        y = np.array(y > threshold, dtype=int)

    # Inverse transform data
    if output_type == "binary":
        if sp.issparse(y):
            y = y.toarray()
        if y.ndim == 2 and y.shape[1] == 2:
            return classes[y[:, 1]]
        else:
            if len(classes) == 1:
                return np.repeat(classes[0], len(y))
            else:
                return classes[y.ravel()]

    elif output_type == "multilabel-indicator":
        return y

    else:
        raise ValueError("{0} format is not supported".format(output_type))
