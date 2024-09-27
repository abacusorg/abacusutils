import numba as nb


@nb.njit
def cumsum(arr, out, initial=False, final=True, offset=0):
    """
    Compute the cumulative sum of an array, storing the result in the output array.
    The (scalar) total is returned.

    The length of the output array depends on the values of initial and final.

    Defaults conform to numpy.cumsum().

    Parameters
    ----------
    arr : array-like
        The input array.

    out : array-like
        The output array.

    initial : bool, optional
        If True, the first element of the output array is set to 0.
        Defaults to False.

    final : bool, optional
        If True, the last element of the output array is set to the total sum.
        Defaults to True.

    offset : scalar, optional
        The initial value of the total sum.
        Defaults to 0.

    Returns
    -------
    total : scalar
        The total sum of the input array (plus the offset).
    """

    N = len(arr)
    N_out = N - 1 + int(initial) + int(final)
    if len(out) != N_out:
        raise ValueError('Output array has incorrect length')

    dtype = out.dtype.type
    total = dtype(offset)

    if initial:
        out[0] = total

    for i in range(N - 1):
        total += arr[i]
        out[i + int(initial)] = total

    total += arr[-1]
    if final:
        out[-1] = total

    return total
