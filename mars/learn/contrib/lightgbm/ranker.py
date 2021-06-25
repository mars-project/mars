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

from ...utils import check_consistent_length
from ..utils import make_import_error_func
from .core import LGBMScikitLearnBase, LGBMModelType
from ._train import train
from ._predict import predict_base

try:
    import lightgbm
except ImportError:
    lightgbm = None


LGBMRanker = make_import_error_func('lightgbm')
if lightgbm:
    class LGBMRanker(LGBMScikitLearnBase, lightgbm.LGBMRanker):
        def fit(self, X, y, sample_weight=None, init_score=None, group=None, eval_set=None,
                eval_sample_weight=None, eval_init_score=None,
                session=None, run_kwargs=None, **kwargs):
            check_consistent_length(X, y, session=session, run_kwargs=run_kwargs)
            params = self.get_params(True)
            model = train(params, self._wrap_train_tuple(X, y, sample_weight, init_score),
                          eval_sets=self._wrap_eval_tuples(eval_set, eval_sample_weight, eval_init_score),
                          group=group, model_type=LGBMModelType.RANKER,
                          session=session, run_kwargs=run_kwargs, **kwargs)

            self.set_params(**model.get_params())
            self._copy_extra_params(model, self)
            return self

        def predict(self, X, **kw):
            session = kw.pop('session', None)
            run_kwargs = kw.pop('run_kwargs', None)
            return predict_base(self, X, session=session, run_kwargs=run_kwargs, **kw)

        def to_local(self):
            model = lightgbm.LGBMRanker(**self.get_params())
            self._copy_extra_params(self, model)
            return model
