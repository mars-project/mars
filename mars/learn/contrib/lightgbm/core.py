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

import enum
import itertools
from collections import namedtuple

import numpy as np
import pandas as pd

from ....tensor import tensor as mars_tensor
from ....dataframe import DataFrame as MarsDataFrame, Series as MarsSeries


class LGBMModelType(enum.Enum):
    CLASSIFIER = 0
    REGRESSOR = 1
    RANKER = 2


_model_type_to_model = dict()


def get_model_cls_from_type(model_type: LGBMModelType):
    import lightgbm
    if not _model_type_to_model:
        _model_type_to_model.update({
            LGBMModelType.CLASSIFIER: lightgbm.LGBMClassifier,
            LGBMModelType.REGRESSOR: lightgbm.LGBMRegressor,
            LGBMModelType.RANKER: lightgbm.LGBMRanker,
        })
    return _model_type_to_model[model_type]


TrainTuple = namedtuple('TrainTuple', 'data label sample_weight init_score')


class LGBMScikitLearnBase:
    def __init__(self, *args, **kwargs):
        if args and isinstance(args[0], self._get_lgbm_class()):
            model = args[0]
            super().__init__(**model.get_params())
            self._copy_extra_params(model, self)
        else:
            super().__init__(*args, **kwargs)

    @classmethod
    def _get_lgbm_class(cls):
        try:
            return getattr(cls, '_lgbm_class')
        except AttributeError:
            lgbm_class = next(base for base in cls.__bases__
                              if base.__module__.startswith('lightgbm'))
            cls._lgbm_class = lgbm_class
            return lgbm_class

    @classmethod
    def _get_param_names(cls):
        return cls._get_lgbm_class()._get_param_names()

    @staticmethod
    def _copy_extra_params(source, dest):
        params = source.get_params()
        attributes = source.__dict__
        extra_param_names = set(attributes.keys()).difference(params.keys())
        for name in extra_param_names:
            setattr(dest, name, attributes[name])

    @staticmethod
    def _convert_tileable(obj):
        if isinstance(obj, np.ndarray):
            return mars_tensor(obj)
        elif isinstance(obj, pd.DataFrame):
            return MarsDataFrame(obj)
        elif isinstance(obj, pd.Series):
            return MarsSeries(obj)
        return obj

    @classmethod
    def _wrap_train_tuple(cls, data, label, sample_weight=None, init_score=None):
        data = cls._convert_tileable(data)
        label = cls._convert_tileable(label)
        sample_weight = cls._convert_tileable(sample_weight)
        init_score = cls._convert_tileable(init_score)
        return TrainTuple(data, label, sample_weight, init_score)

    @staticmethod
    def _wrap_eval_tuples(eval_set=None, eval_sample_weight=None, eval_init_score=None):
        if not eval_set:
            return None

        tps = []
        for (data, label), weight, score in zip(eval_set, eval_sample_weight or itertools.repeat(None),
                                                eval_init_score or itertools.repeat(None)):
            tps.append(TrainTuple(data, label, weight, score))
        return tps

    def fit(self, X, y, sample_weight=None, **kwargs):
        raise NotImplementedError

    def predict(self, X, **kwargs):
        raise NotImplementedError

    def predict_proba(self, X, **kwargs):
        raise NotImplementedError

    def load_model(self, model):
        self.set_params(**self.get_params())
        self._copy_extra_params(model, self)
        return self
