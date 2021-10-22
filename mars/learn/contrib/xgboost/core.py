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

from typing import Any, Callable, List, Optional, Tuple

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

        def fit(
            self,
            X,
            y,
            sample_weights=None,
            eval_set=None,
            sample_weight_eval_set=None,
            **kw,
        ):
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

    def wrap_evaluation_matrices(
        missing: float,
        X: Any,
        y: Any,
        sample_weight: Optional[Any],
        base_margin: Optional[Any],
        eval_set: Optional[List[Tuple[Any, Any]]],
        sample_weight_eval_set: Optional[List[Any]],
        base_margin_eval_set: Optional[List[Any]],
        label_transform: Callable = lambda x: x,
    ) -> Tuple[Any, Optional[List[Tuple[Any, str]]]]:
        """Convert array_like evaluation matrices into DMatrix.  Perform validation on the way."""
        train_dmatrix = MarsDMatrix(
            data=X,
            label=label_transform(y),
            weight=sample_weight,
            base_margin=base_margin,
            missing=missing,
        )

        n_validation = 0 if eval_set is None else len(eval_set)

        def validate_or_none(meta: Optional[List], name: str) -> List:
            if meta is None:
                return [None] * n_validation
            if len(meta) != n_validation:
                raise ValueError(
                    f"{name}'s length does not equal `eval_set`'s length, "
                    + f"expecting {n_validation}, got {len(meta)}"
                )
            return meta

        if eval_set is not None:
            sample_weight_eval_set = validate_or_none(
                sample_weight_eval_set, "sample_weight_eval_set"
            )
            base_margin_eval_set = validate_or_none(
                base_margin_eval_set, "base_margin_eval_set"
            )

            evals = []
            for i, (valid_X, valid_y) in enumerate(eval_set):
                # Skip the duplicated entry.
                if all(
                    (
                        valid_X is X,
                        valid_y is y,
                        sample_weight_eval_set[i] is sample_weight,
                        base_margin_eval_set[i] is base_margin,
                    )
                ):
                    evals.append(train_dmatrix)
                else:
                    m = MarsDMatrix(
                        data=valid_X,
                        label=label_transform(valid_y),
                        weight=sample_weight_eval_set[i],
                        base_margin=base_margin_eval_set[i],
                        missing=missing,
                    )
                    evals.append(m)
            nevals = len(evals)
            eval_names = [f"validation_{i}" for i in range(nevals)]
            evals = list(zip(evals, eval_names))
        else:
            if any(
                meta is not None
                for meta in [
                    sample_weight_eval_set,
                    base_margin_eval_set,
                ]
            ):
                raise ValueError(
                    "`eval_set` is not set but one of the other evaluation meta info is "
                    "not None."
                )
            evals = []

        return train_dmatrix, evals
