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

try:
    import scipy

    from .err_fresnel import erf, TensorErf
    from .gamma_funcs import (
        gamma, TensorGamma,
        gammaln, TensorGammaln,
        loggamma, TensorLogGamma,
        gammasgn, TensorGammaSgn,
        gammainc, TensorGammaInc,
        gammaincinv, TensorGammaIncInv,
        gammaincc, TensorGammaIncc,
        gammainccinv, TensorGammaInccInv,
        beta, TensorBeta,
        betaln, TensorBetaLn,
        betainc, TensorBetaInc,
        betaincinv, TensorBetaIncInv,
        psi, TensorPsi,
        rgamma, TensorRGamma,
        polygamma, TensorPolyGamma,
        multigammaln, TensorMultiGammaLn,
        digamma, TensorDiGamma,
        poch, TensorPoch,
    )
    from .info_theory import (
        entr, TensorEntr,
        rel_entr, TensorRelEntr,
        kl_div, TensorKlDiv,
    )
    from .convenience import (
        xlogy, TensorXLogY,
    )
    from .bessel import (
        jv, TensorJV,
        jve, TensorJVE,
        yn, TensorYN,
        yv, TensorYV,
        yve, TensorYVE,
        kn, TensorKN,
        kv, TensorKV,
        kve, TensorKVE,
        iv, TensorIV,
        ive, TensorIVE,
        hankel1, TensorHankel1,
        hankel1e, TensorHankel1e,
        hankel2, TensorHankel2,
        hankel2e, TensorHankel2e,
    )
except ImportError:  # pragma: no cover
    pass
