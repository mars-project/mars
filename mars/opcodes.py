# Generated automatically.  DO NOT EDIT!
NULL = 0
SCALAR = 1
TENSOR_DATA_SOURCE = 2
TENSOR_ONES = 3
TENSOR_ONES_LIKE = 4
TENSOR_ZEROS = 5
TENSOR_ZEROS_LIKE = 6
TENSOR_EMPTY = 7
TENSOR_EMPTY_LIKE = 8
TENSOR_FULL = 9
TENSOR_FULL_LIKE = 25
TENSOR_ARANGE = 10
TENSOR_INDICES = 11
TENSOR_DIAG = 12
TENSOR_EYE = 13
TENSOR_LINSPACE = 14
TENSOR_TRIU = 15
TENSOR_TRIL = 16
TENSOR_FROM_TILEDB = 18
TENSOR_STORE_TILEDB = 19
TENSOR_STORE_TILEDB_CONSOLIDATE = 20
TENSOR_FROM_DATAFRAME = 22
DATAFRAME_DATA_SOURCE = 17
DATAFRAME_FROM_TENSOR = 21
DATAFRAME_FROM_RECORDS = 24
SERIES_DATA_SOURCE = 23
RAND_RAND = 41
RAND_RANDN = 42
RAND_RANDINT = 43
RAND_RANDOM_INTEGERS = 44
RAND_RANDOM_SAMPLE = 45
RAND_RANDOM = 46
RAND_RANF = 47
RAND_SAMPLE = 48
RAND_BYTES = 49
RAND_BETA = 50
RAND_BINOMIAL = 51
RAND_CHISQUARE = 52
RAND_CHOICE = 53
RAND_DIRICHLET = 54
RAND_EXPONENTIAL = 55
RAND_F = 56
RAND_GAMMA = 57
RAND_GEOMETRIC = 58
RAND_GUMBEL = 59
RAND_HYPERGEOMETRIC = 60
RAND_LAPLACE = 61
RAND_LOGISTIC = 62
RAND_LOGNORMAL = 63
RAND_LOGSERIES = 64
RAND_MULTINOMIAL = 65
RAND_MULTIVARIATE_NORMAL = 66
RAND_NEGATIVE_BINOMIAL = 67
RAND_NONCENTRAL_CHISQURE = 68
RAND_NONCENTRAL_F = 69
RAND_NORMAL = 70
RAND_PARETO = 71
RAND_PERMUTATION = 72
RAND_POSSION = 73
RAND_POWER = 74
RAND_RAYLEIGH = 75
RAND_SHUFFLE = 76
RAND_STANDARD_CAUCHY = 77
RAND_STANDARD_EXPONENTIAL = 78
RAND_STANDARD_GAMMMA = 79
RAND_STANDARD_NORMAL = 80
RAND_STANDARD_T = 81
RAND_TOMAXINT = 82
RAND_TRIANGULAR = 83
RAND_UNIFORM = 84
RAND_VONMISES = 85
RAND_WALD = 86
RAND_WEIBULL = 87
RAND_ZIPF = 88
ADD = 101
SUB = 102
MUL = 103
DIV = 104
TRUEDIV = 105
FLOORDIV = 106
POW = 107
MOD = 108
FMOD = 109
LOGADDEXP = 110
LOGADDEXP2 = 111
NEGATIVE = 112
POSITIVE = 113
ABSOLUTE = 114
FABS = 115
ABS = 116
RINT = 117
SIGN = 118
CONJ = 119
EXP = 120
EXP2 = 121
LOG = 122
LOG2 = 123
LOG10 = 124
EXPM1 = 125
LOG1P = 126
SQRT = 127
SQUARE = 128
CBRT = 129
RECIPROCAL = 130
EQ = 131
NE = 132
LT = 133
LE = 134
GT = 135
GE = 136
SIN = 137
COS = 138
TAN = 139
ARCSIN = 140
ARCCOS = 141
ARCTAN = 142
ARCTAN2 = 143
HYPOT = 144
SINH = 145
COSH = 146
TANH = 147
ARCSINH = 148
ARCCOSH = 149
ARCTANH = 150
DEG2RAD = 151
RAD2DEG = 152
BITAND = 153
BITOR = 154
BITXOR = 155
INVERT = 156
LSHIFT = 157
RSHIFT = 158
AND = 159
OR = 160
XOR = 161
NOT = 162
MAXIMUM = 163
MINIMUM = 164
AROUND = 165
FLOAT_POWER = 166
FMAX = 167
FMIN = 168
ISFINITE = 169
ISINF = 170
ISNAN = 171
SIGNBIT = 172
COPYSIGN = 173
NEXTAFTER = 174
SPACING = 175
LDEXP = 176
FREXP = 177
MODF = 178
FLOOR = 179
CEIL = 180
TRUNC = 181
DEGREES = 182
RADIANS = 183
CLIP = 184
ISREAL = 185
ISCOMPLEX = 186
REAL = 187
IMAG = 188
FIX = 189
I0 = 190
SINC = 191
NAN_TO_NUM = 192
ISCLOSE = 193
DIVMOD = 194
ANGLE = 195
SET_REAL = 196
SET_IMAG = 197
GAMMALN = 200
ERF = 201
TREE_ADD = 251
TREE_MULTIPLY = 252
CUMSUM = 301
CUMPROD = 302
PROD = 303
SUM = 304
MAX = 305
MIN = 306
ALL = 307
ANY = 308
MEAN_CHUNK = 309
MEAN_COMBINE = 310
MEAN = 311
ARGMAX = 312
ARGMAX_CHUNK = 313
ARGMAX_COMBINE = 314
ARGMIN = 315
ARGMIN_CHUNK = 316
ARGMIN_COMBINE = 317
NANSUM = 318
NANMAX = 319
NANMIN = 320
NANPROD = 321
NANMEAN = 322
NANMEAN_CHUNK = 323
NANARGMAX = 324
NANARGMAX_CHUNK = 325
NANARGMAX_COMBINE = 326
NANARGMIN = 327
NANARGMIN_CHUNK = 328
NANARGMIN_COMBINE = 329
COUNT_NONZERO = 330
MOMENT_CHUNK = 331
NANMOMENT_CHUNK = 332
MOMENT_COMBINE = 333
NANMOMENT_COMBINE = 334
MOMENT = 335
NANMOMENT = 336
VAR = 337
STD = 338
NANVAR = 339
NANSTD = 340
NANCUMSUM = 341
NANCUMPROD = 342
RESHAPE = 401
SLICE = 402
INDEX = 403
INDEXSETVALUE = 404
CONCATENATE = 405
RECHUNK = 406
ASTYPE = 407
TRANSPOSE = 408
SWAPAXES = 409
BROADCAST_TO = 410
STACK = 411
WHERE = 412
CHOOSE = 413
NONZERO = 414
ARGWHERE = 415
UNRAVEL_INDEX = 416
RAVEL_MULTI_INDEX = 417
ARRAY_SPLIT = 418
SQUEEZE = 419
DIGITIZE = 420
REPEAT = 421
COPYTO = 422
ISIN = 423
SEARCHSORTED = 428
FANCY_INDEX_DISTRIBUTE_MAP = 424
FANCY_INDEX_DISTRIBUTE_REDUCE = 425
FANCY_INDEX_CONCAT_MAP = 426
FANCY_INDEX_CONCAT_REDUCE = 427
TENSORDOT = 501
DOT = 502
MATMUL = 503
CHOLESKY = 510
QR = 511
SVD = 512
LU = 513
SOLVE_TRIANGULAR = 520
INV = 521
NORM = 530
FFT = 601
IFFT = 602
FFT2 = 603
IFFT2 = 604
FFTN = 605
IFFTN = 606
RFFT = 607
IRFFT = 608
RFFT2 = 609
IRFFT2 = 610
RFFTN = 611
IRFFTN = 612
HFFT = 613
IHFFT = 614
FFTFREQ = 615
FFTFREQ_CHUNK = 616
RFFTFREQ = 617
FFTSHIFT = 618
IFFTSHIFT = 619
SPARSE_MATRIX_DATA_SOURCE = 701
DENSE_TO_SPARSE = 702
SPARSE_TO_DENSE = 703
FUSE = 801
ENTER = 901
LEAVE = 902
FIX_LATEST = 903
IF_ELSE = 904
NEXT_ITER = 905
TABLE_COO = 1003
STORE_COO = 1004
SHUFFLE_PROXY = 2001
RESHAPE_MAP = 2002
RESHAPE_REDUCE = 2003
DATAFRAME_INDEX_ALIGN_MAP = 2004
DATAFRAME_INDEX_ALIGN_REDUCE = 2005
DATAFRAME_SHUFFLE_MERGE = 2010
DATAFRAME_SHUFFLE_MERGE_ALIGN_MAP = 2011
DATAFRAME_SHUFFLE_MERGE_ALIGN_REDUCE = 2012
FETCH_SHUFFLE = 999998
FETCH = 999999
