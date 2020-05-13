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

from .core import xgboost, XGBScikitLearnBase


XGBClassifier = None
if xgboost:
    from xgboost.sklearn import XGBClassifierBase

    from .... import tensor as mt
    from .dmatrix import MarsDMatrix
    from .core import evaluation_matrices
    from .train import train
    from .predict import predict

    class XGBClassifier(XGBScikitLearnBase, XGBClassifierBase):
        """
        Implementation of the scikit-learn API for XGBoost classification.
        """

        def fit(self, X, y, sample_weights=None, eval_set=None, sample_weight_eval_set=None, **kw):
            session = kw.pop('session', None)
            run_kwargs = kw.pop('run_kwargs', dict())
            if kw:
                raise TypeError("fit got an unexpected keyword argument '{0}'".format(next(iter(kw))))

            dtrain = MarsDMatrix(X, label=y, weight=sample_weights,
                                 session=session, run_kwargs=run_kwargs)
            params = self.get_xgb_params()

            self.classes_ = mt.unique(y, aggregate_size=1).to_numpy(session=session, **run_kwargs)
            self.n_classes_ = len(self.classes_)

            if self.n_classes_ > 2:
                params['objective'] = 'multi:softprob'
                params['num_class'] = self.n_classes_
            else:
                params['objective'] = 'binary:logistic'

            evals = evaluation_matrices(eval_set, sample_weight_eval_set,
                                        session=session, run_kwargs=run_kwargs)
            self.evals_result_ = dict()
            result = train(params, dtrain, num_boost_round=self.get_num_boosting_rounds(),
                           evals=evals, evals_result=self.evals_result_,
                           session=session, run_kwargs=run_kwargs)
            self._Booster = result
            return self

        def predict(self, data, **kw):
            session = kw.pop('session', None)
            run_kwargs = kw.pop('run_kwargs', dict())
            if kw:
                raise TypeError("predict got an unexpected "
                                "keyword argument '{0}'".format(next(iter(kw))))
            prob = predict(self.get_booster(), data, run=False)
            if prob.ndim > 1:
                prediction = mt.argmax(prob, axis=1)
            else:
                prediction = (prob > 0.5).astype(mt.int64)
            prediction.execute(session=session, **run_kwargs)
            return prediction

        def predict_proba(self, data, ntree_limit=None, **kw):
            if ntree_limit is not None:
                raise NotImplementedError('ntree_limit is not currently supported')
            return predict(self.get_booster(), data, **kw)
