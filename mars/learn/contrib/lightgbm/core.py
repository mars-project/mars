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

import enum

try:
    import lightgbm
except ImportError:
    lightgbm = None


class LGBMModelType(enum.Enum):
    CLASSIFIER = 0
    REGRESSOR = 1


_model_type_to_model = dict()


def get_model_cls_from_type(model_type: LGBMModelType):
    return _model_type_to_model[model_type]


LGBMScikitLearnBase = None


if lightgbm:
    _model_type_to_model = {
        LGBMModelType.CLASSIFIER: lightgbm.LGBMClassifier,
        LGBMModelType.REGRESSOR: lightgbm.LGBMRegressor,
    }

    class LGBMScikitLearnBase:
        @staticmethod
        def _copy_extra_params(source, dest):
            params = source.get_params()
            attributes = source.__dict__
            extra_param_names = set(attributes.keys()).difference(params.keys())
            for name in extra_param_names:
                setattr(dest, name, attributes[name])

        def fit(self, X, y, sample_weight=None, **kwargs):
            raise NotImplementedError

        def predict(self, X, **kwargs):
            raise NotImplementedError

        def predict_proba(self, X, **kwargs):
            raise NotImplementedError
