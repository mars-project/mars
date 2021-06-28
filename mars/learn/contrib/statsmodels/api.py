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

import pickle  # nosec  # pylint: disable=import_pickle

from .train import StatsModelsTrain
from .predict import StatsModelsPredict

try:
    from statsmodels.base.distributed_estimation import DistributedModel
except ImportError:
    DistributedModel = None


class MarsDistributedModel:
    def __init__(self, factor=None, num_partitions=None, model_class=None,
                 init_kwds=None, estimation_method=None, estimation_kwds=None,
                 join_method=None, join_kwds=None, results_class=None, results_kwds=None):
        self._factor = factor
        self._sm_model = DistributedModel(
            num_partitions or 10, model_class=model_class, init_kwds=init_kwds,
            estimation_method=estimation_method, estimation_kwds=estimation_kwds,
            join_method=join_method, join_kwds=join_kwds, results_class=results_class,
            results_kwds=results_kwds
        )

    def fit(self, endog, exog, session=None, **kwargs):
        num_partitions = None if self._factor is not None else self._sm_model.partitions
        run_kwargs = kwargs.pop('run_kwargs', dict())
        op = StatsModelsTrain(
            endog=endog, exog=exog, num_partitions=num_partitions, factor=self._factor,
            model_class=self._sm_model.model_class, init_kwds=self._sm_model.init_kwds,
            fit_kwds=kwargs, estimation_method=self._sm_model.estimation_method,
            estimation_kwds=self._sm_model.estimation_kwds,
            join_method=self._sm_model.join_method, join_kwds=self._sm_model.join_kwds,
            results_class=self._sm_model.results_class,
            results_kwds=self._sm_model.results_kwds)
        result = op(exog, endog).execute(session=session, **run_kwargs).fetch(session=session)
        return MarsResults(pickle.loads(result))  # nosec


class MarsResults:
    def __init__(self, model):
        self._model = model

    @property
    def model(self):
        return self._model

    def __getattr__(self, item):
        if item == '_model':
            raise AttributeError
        return getattr(self._model, item)

    def __mars_tokenize__(self):
        return pickle.dumps(self.model)

    def predict(self, exog, *args, **kwargs):
        session = kwargs.pop('session', None)
        run_kwargs = kwargs.pop('run_kwargs', dict())
        op = StatsModelsPredict(model_results=self, predict_args=args, predict_kwargs=kwargs)
        return op(exog).execute(session=session, **run_kwargs)
