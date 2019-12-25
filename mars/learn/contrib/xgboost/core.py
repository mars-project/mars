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

try:
    import xgboost
except ImportError:
    xgboost = None

from .dmatrix import MarsDMatrix


XGBScikitLearnBase = None
if xgboost:
    class XGBScikitLearnBase(xgboost.XGBModel):
        """
        Base class for implementing scikit-learn interface
        """

        def fit(self, X, y, sample_weights=None, eval_set=None, sample_weight_eval_set=None, **kw):
            """
            Fit the regressor.
            Parameters
            ----------
            X : array_like
                Feature matrix
            y : array_like
                Labels
            sample_weight : array_like
                instance weights
            eval_set : list, optional
                A list of (X, y) tuple pairs to use as validation sets, for which
                metrics will be computed.
                Validation metrics will help us track the performance of the model.
            sample_weight_eval_set : list, optional
                A list of the form [L_1, L_2, ..., L_n], where each L_i is a list
                of group weights on the i-th validation set.
            """
            raise NotImplementedError

        def predict(self, data, **kw):
            """
            Predict with `data`.

            Parameters
            ----------
              data: data that can be used to perform prediction
            Returns
            -------
            prediction : mars.tensor.Tensor
            """
            raise NotImplementedError

    def evaluation_matrices(validation_set, sample_weights, session=None, run_kwargs=None):
        """
        Parameters
        ----------
        validation_set: list of tuples
            Each tuple contains a validation dataset including input X and label y.
            E.g.:
            .. code-block:: python
              [(X_0, y_0), (X_1, y_1), ... ]
        sample_weights: list of arrays
            The weight vector for validation data.
        session:
            Session to run
        run_kwargs:
            kwargs for session.run
        Returns
        -------
        evals: list of validation MarsDMatrix
        """
        evals = []
        if validation_set is not None:
            assert isinstance(validation_set, list)
            for i, e in enumerate(validation_set):
                w = (sample_weights[i]
                     if sample_weights is not None else None)
                dmat = MarsDMatrix(e[0], label=e[1], weight=w,
                                   session=session, run_kwargs=run_kwargs)
                evals.append((dmat, 'validation_{}'.format(i)))
        else:
            evals = None
        return evals
