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

from typing import List, Union

import numpy as np
from sklearn.base import BaseEstimator as SklearnBaseEstimator, clone
from sklearn.utils.validation import check_is_fitted

from ... import opcodes
from ... import execute
from ... import tensor as mt
from ...core import OutputType, ENTITY_TYPE, recursive_tile
from ...core.context import Context
from ...serialization.serializables import (
    FieldTypes,
    AnyField,
    BoolField,
    DictField,
    Int64Field,
    ListField,
    StringField,
    KeyField,
)
from ...typing import SessionType
from ...tensor.core import Tensor, TensorOrder
from ...tensor.utils import decide_unify_split
from ..operands import LearnOperand, LearnOperandMixin
from ..base import BaseEstimator, ClassifierMixin, RegressorMixin
from ..utils import check_array


class BlockwiseEnsembleFit(LearnOperand, LearnOperandMixin):
    _op_type_ = opcodes.BLOCKWISE_ENSEMBLE_FIT

    x = KeyField("x")
    y = KeyField("y")
    estimator = AnyField("estimator")
    kwargs = DictField("kwargs", default_factory=dict)

    def __call__(self):
        self._output_types = [OutputType.object]
        return self.new_tileable([self.x, self.y])

    def _set_inputs(self, inputs):
        super()._set_inputs(inputs)
        self.x = self._inputs[0]
        self.y = self._inputs[1]

    @classmethod
    def tile(cls, op: "BlockwiseEnsembleFit"):
        X, y = op.x, op.y
        x_split = X.nsplits[0]
        y_split = y.nsplits[0]
        out = op.outputs[0]

        if any(np.isnan(s) for s in x_split + y_split) or np.isnan(
            X.shape[1]
        ):  # pragma: no ccover
            yield

        if x_split != y_split or X.chunk_shape[1] > 1:
            x_split = y_split = decide_unify_split(x_split, y_split)
            X = X.rechunk({0: x_split, 1: X.shape[1]})
            y = y.rechunk({0: y_split})
            X, y = yield from recursive_tile(X, y)

        out_chunks = []
        for i, _ in enumerate(x_split):
            chunk_op = op.copy().reset_key()
            out_chunk = chunk_op.new_chunk(
                [X.cix[i, 0], y.cix[(i,)]],
                index=(i,),
            )
            out_chunks.append(out_chunk)

        params = out.params.copy()
        params["chunks"] = out_chunks
        params["nsplits"] = ((np.nan,) * len(x_split),)
        return op.copy().new_tileables(op.inputs, kws=[params])

    @classmethod
    def execute(cls, ctx: Union[dict, Context], op: "BlockwiseEnsembleFit"):
        x, y = ctx[op.inputs[0].key], ctx[op.inputs[1].key]
        estimator = clone(op.estimator)
        ctx[op.outputs[0].key] = estimator.fit(x, y, **op.kwargs)


class BlockwiseEnsemblePredict(LearnOperand, LearnOperandMixin):
    _op_type_ = opcodes.BLOCKWISE_ENSEMBLE_PREDICT

    x = KeyField("x")
    estimators = ListField("estimators", FieldTypes.key)
    voting = StringField("voting", default="hard")
    proba = BoolField("proba", default=None)
    is_classifier = BoolField("is_classifier")
    n_classes = Int64Field("n_classes")

    def _set_inputs(self, inputs):
        super()._set_inputs(inputs)
        self.x = self._inputs[0]
        self.estimators = self._inputs[1:]

    def __call__(self):
        self._output_types = [OutputType.tensor]
        x_len = self.x.shape[0]
        if self.is_classifier:
            shape = (x_len, self.n_classes) if self.proba else (x_len,)
            dtype = np.dtype(np.float64) if self.proba else np.dtype(np.int64)
        else:
            shape = (x_len,)
            dtype = np.dtype(np.float64)
        return self.new_tileable(
            [self.x] + self.estimators,
            shape=shape,
            dtype=dtype,
            order=TensorOrder.C_ORDER,
        )

    @classmethod
    def tile(cls, op: "BlockwiseEnsemblePredict"):
        out = op.outputs[0]
        x = op.x
        estimators = op.estimators[0]
        estimators_chunks = estimators.chunks

        out_chunks = []
        for chunk in x.chunks:
            chunk_op = op.copy().reset_key()
            if out.ndim == 2:
                chunk_shape = (chunk.shape[0], out.shape[1])
                chunk_index = (chunk.index[0], 0)
            else:
                chunk_shape = (chunk.shape[0],)
                chunk_index = (chunk.index[0],)
            out_chunk = chunk_op.new_chunk(
                [chunk] + estimators_chunks,
                shape=chunk_shape,
                dtype=out.dtype,
                order=out.order,
                index=chunk_index,
            )
            out_chunks.append(out_chunk)

        if out.ndim == 2:
            nsplits = (x.nsplits[0], (out.shape[1],))
        else:
            nsplits = (x.nsplits[0],)
        params = out.params.copy()
        params["nsplits"] = nsplits
        params["chunks"] = out_chunks
        return op.copy().new_tileables(op.inputs, kws=[params])

    @classmethod
    def execute(cls, ctx: Union[dict, Context], op: "BlockwiseEnsemblePredict"):
        x = ctx[op.inputs[0].key]
        estimators = [ctx[inp.key] for inp in op.inputs[1:]]
        if op.proba or op.voting == "soft":
            predictions = [estimator.predict_proba(x) for estimator in estimators]
        else:
            predictions = [estimator.predict(x) for estimator in estimators]

        if op.is_classifier:
            if not op.proba:
                result = cls._execute_classifier_predict(predictions, op)
            else:
                result = cls._execute_classifier_predict_proba(predictions, op)
        else:
            result = cls._execute_regressor_predict(predictions)
        ctx[op.outputs[0].key] = result

    @classmethod
    def _execute_classifier_predict(
        cls, predictions: List[np.ndarray], op: "BlockwiseEnsemblePredict"
    ):
        if op.voting == "soft":
            prob = np.average(np.stack(predictions), axis=0)
            return np.argmax(prob, axis=1)
        else:

            def vote(x: np.ndarray):
                return np.argmax(np.bincount(x))

            # hard voting
            prediction = np.vstack(predictions).T
            return np.apply_along_axis(vote, 1, prediction)

    @classmethod
    def _execute_classifier_predict_proba(
        cls, predictions: List[np.ndarray], op: "BlockwiseEnsemblePredict"
    ):
        assert op.voting == "soft"
        return np.average(np.stack(predictions), axis=0)

    @classmethod
    def _execute_regressor_predict(cls, predictions: List[np.ndarray]):
        return np.average(np.vstack(predictions), axis=0)


class BlockwiseBaseEstimator(BaseEstimator):
    def __init__(self, estimator: SklearnBaseEstimator):
        self.estimator = estimator

    def _fit(self, X, y, **kwargs):
        X = check_array(X)
        op = BlockwiseEnsembleFit(x=X, y=y, estimator=self.estimator, kwargs=kwargs)
        self.estimators_ = op()


class BlockwiseVotingClassifier(ClassifierMixin, BlockwiseBaseEstimator):
    """
    Blockwise training and ensemble voting classifier.

    This classifier trains on blocks / partitions of tensors or DataFrames.
    A cloned version of `estimator` will be fit *independently* on each block
    or partition of the data. This is useful when the sub estimator
    only works on small in-memory data structures like a NumPy array or pandas
    DataFrame.

    Prediction is done by the *ensemble* of learned models.

    .. warning::

       Ensure that your data are sufficiently shuffled prior to training!
       If the values of the various blocks / partitions of your dataset are not
       distributed similarly, the classifier will give poor results.

    Parameters
    ----------
    estimator : Estimator
    voting : str, {'hard', 'soft'} (default='hard')
        If 'hard', uses predicted class labels for majority rule voting.
        Else if 'soft', predicts the class label based on the argmax of
        the sums of the predicted probabilities, which is recommended for
        an ensemble of well-calibrated classifiers.
    classes : list-like, optional
        The set of classes that `y` can take. This can also be provided as
        a fit param if the underlying estimator requires `classes` at fit time.

    Attributes
    ----------
    estimators_ : list of classifiers
        The collection of fitted sub-estimators that are `estimator` fitted
        on each partition / block of the inputs.

    classes_ : array-like, shape (n_predictions,)
        The class labels.

    Examples
    --------
    >>> import mars.tensor as mt
    >>> from mars.learn.ensemble import BlockwiseVotingClassifier
    >>> from sklearn.linear_model import RidgeClassifier
    >>> from sklearn.datasets import make_classification
    >>> X, y = make_classification(n_samples=100_000)
    >>> X, y = mt.tensor(X, chunk_size=10_0000), mt.tensor(y, chunk_size=10_0000)
    >>> subestimator = RidgeClassifier(random_state=0)
    >>> clf = BlockwiseVotingClassifier(subestimator)
    >>> clf.fit(X, y)
    """

    def __init__(
        self,
        estimator: SklearnBaseEstimator,
        voting: str = "hard",
        classes: Union[np.ndarray, list, Tensor] = None,
    ):
        super().__init__(estimator=estimator)
        if voting not in ("hard", "soft"):  # pragma: no cover
            raise ValueError("`voting` could be hard or soft")
        self.voting = voting
        self.classes = None
        if classes is not None:
            self.classes = mt.tensor(classes)

    def fit(
        self,
        X,
        y,
        classes: Union[np.ndarray, list, Tensor] = None,
        session: SessionType = None,
        run_kwargs: dict = None,
        **kwargs,
    ):
        if not isinstance(y, ENTITY_TYPE):
            y = mt.tensor(y)
        if classes is None:
            classes = self.classes
        to_execute = []
        if classes is None:
            classes = mt.unique(y)
            to_execute.append(classes)
        super()._fit(X, y, **kwargs)
        to_execute.append(self.estimators_)
        execute(to_execute, session=session, **(run_kwargs or dict()))
        self.n_classes_ = len(classes)

    def predict(self, X, session: SessionType = None, run_kwargs: dict = None):
        check_is_fitted(self, attributes=["estimators_"])
        X = check_array(X)
        op = BlockwiseEnsemblePredict(
            x=X,
            estimators=[self.estimators_],
            voting=self.voting,
            proba=False,
            is_classifier=True,
            n_classes=self.n_classes_,
        )
        return op().execute(session=session, **(run_kwargs or dict()))

    def predict_proba(self, X, session: SessionType = None, run_kwargs: dict = None):
        if self.voting == "hard":
            raise AttributeError(f'predict_proba is not available when voting="hard"')

        check_is_fitted(self, attributes=["estimators_"])
        X = check_array(X)
        op = BlockwiseEnsemblePredict(
            x=X,
            estimators=[self.estimators_],
            voting=self.voting,
            proba=True,
            is_classifier=True,
            n_classes=self.n_classes_,
        )
        return op().execute(session=session, **(run_kwargs or dict()))


class BlockwiseVotingRegressor(RegressorMixin, BlockwiseBaseEstimator):
    """
    Blockwise training and ensemble voting regressor.

    This regressor trains on blocks / partitions of tensors or DataFrames.
    A cloned version of `estimator` will be fit *independently* on each block
    or partition of the data.

    Prediction is done by the *ensemble* of learned models.

    .. warning::
       Ensure that your data are sufficiently shuffled prior to training!
       If the values of the various blocks / partitions of your dataset are not
       distributed similarly, the regressor will give poor results.

    Parameters
    ----------
    estimator : Estimator

    Attributes
    ----------
    estimators_ : list of regressors
        The collection of fitted sub-estimators that are `estimator` fitted
        on each partition / block of the inputs.

    Examples
    --------
    >>> import mars.tensor as mt
    >>> from mars.learn.ensemble import BlockwiseVotingRegressor
    >>> from sklearn.linear_model import LinearRegression
    >>> from sklearn.datasets import make_classification
    >>> X, y = make_classification(n_samples=100_000)
    >>> X, y = mt.tensor(X, chunk_size=10_0000), mt.tensor(y, chunk_size=10_0000)
    >>> subestimator = LinearRegression()
    >>> clf = BlockwiseVotingRegressor(subestimator)
    >>> clf.fit(X, y)
    """

    def fit(self, X, y, session: SessionType = None, run_kwargs: dict = None, **kwargs):
        if not isinstance(y, ENTITY_TYPE):
            y = mt.tensor(y)
        super()._fit(X, y, **kwargs)
        self.estimators_.execute(session=session, **(run_kwargs or dict()))

    def predict(self, X, session: SessionType = None, run_kwargs: dict = None):
        check_is_fitted(self, attributes=["estimators_"])
        X = check_array(X)
        op = BlockwiseEnsemblePredict(
            x=X, estimators=[self.estimators_], is_classifier=False
        )
        return op().execute(session=session, **(run_kwargs or dict()))
