Universal Functions (ufunc)
===========================

Mars tensor provides universal functions(a.k.a ufuncs) to support various elementwise operations.
Mars tensor's ufunc supports following features of Numpy's one:

- Broadcasting
- Output type determination
- Casting rules

Mars tensor's ufunc currently does not support methods
like ``reduce``, ``accumulate``, ``reduceat``, ``outer``, and ``at``.

Available ufuncs
----------------

Math operations
~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/
   :nosignatures:

   mars.tensor.add
   mars.tensor.subtract
   mars.tensor.multiply
   mars.tensor.divide
   mars.tensor.logaddexp
   mars.tensor.logaddexp2
   mars.tensor.true_divide
   mars.tensor.floor_divide
   mars.tensor.negative
   mars.tensor.power
   mars.tensor.remainder
   mars.tensor.mod
   mars.tensor.fmod
   mars.tensor.absolute
   mars.tensor.rint
   mars.tensor.sign
   mars.tensor.exp
   mars.tensor.exp2
   mars.tensor.log
   mars.tensor.log2
   mars.tensor.log10
   mars.tensor.expm1
   mars.tensor.log1p
   mars.tensor.sqrt
   mars.tensor.square
   mars.tensor.reciprocal


Trigonometric functions
~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/
   :nosignatures:

   mars.tensor.sin
   mars.tensor.cos
   mars.tensor.tan
   mars.tensor.arcsin
   mars.tensor.arccos
   mars.tensor.arctan
   mars.tensor.arctan2
   mars.tensor.hypot
   mars.tensor.sinh
   mars.tensor.cosh
   mars.tensor.tanh
   mars.tensor.arcsinh
   mars.tensor.arccosh
   mars.tensor.arctanh
   mars.tensor.deg2rad
   mars.tensor.rad2deg


Bit-twiddling functions
~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/
   :nosignatures:

   mars.tensor.bitwise_and
   mars.tensor.bitwise_or
   mars.tensor.bitwise_xor
   mars.tensor.invert
   mars.tensor.left_shift
   mars.tensor.right_shift


Comparison functions
~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/
   :nosignatures:

   mars.tensor.greater
   mars.tensor.greater_equal
   mars.tensor.less
   mars.tensor.less_equal
   mars.tensor.not_equal
   mars.tensor.equal
   mars.tensor.logical_and
   mars.tensor.logical_or
   mars.tensor.logical_xor
   mars.tensor.logical_not
   mars.tensor.maximum
   mars.tensor.minimum
   mars.tensor.fmax
   mars.tensor.fmin


Floating point values
~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/
   :nosignatures:

   mars.tensor.isfinite
   mars.tensor.isinf
   mars.tensor.isnan
   mars.tensor.signbit
   mars.tensor.copysign
   mars.tensor.nextafter
   mars.tensor.modf
   mars.tensor.ldexp
   mars.tensor.frexp
   mars.tensor.fmod
   mars.tensor.floor
   mars.tensor.ceil
   mars.tensor.trunc
