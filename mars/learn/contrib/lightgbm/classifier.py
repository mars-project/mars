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

from .core import lightgbm, LGBMScikitLearnBase, LGBMModelType
from .train import train
from .predict import predict


if lightgbm:
    class LGBMClassifier(LGBMScikitLearnBase, lightgbm.LGBMClassifier):
        def fit(self, X, y, sample_weight=None, **kwargs):
            params = self.get_params(True)
            model = train(params, X, y, sample_weight=sample_weight,
                          model_type=LGBMModelType.CLASSIFIER, **kwargs)

            self.set_params(**model.get_params())
            self._copy_extra_params(model, self)
            return self

        def predict(self, X, **kwargs):
            return predict(self, X, proba=False, **kwargs)

        def predict_proba(self, X, **kwargs):
            return predict(self, X, proba=True, **kwargs)

        def to_local(self):
            model = lightgbm.LGBMClassifier(**self.get_params())
            self._copy_extra_params(self, model)
            return model
