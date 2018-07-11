try:
    import numba
    from numba.types import float64, float32, int64, int32
    jit = numba.njit(nogil=True, fastmath=True)

    def guvec(sigs, layout, func):
        return numba.guvectorize(
            sigs, layout, nopython=1, fastmath=1)(func)  # nogil not supported

except ImportError:
    import numpy
    # dummy
    float64 = float32 = int64 = int32 = numpy.empty((1, 1, 1, 1, 1, 1, 1))
    jit = lambda f: None

    def guvec(sigs, layout, func):
        return None