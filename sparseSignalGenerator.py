import numpy as np

def sparseSignal(N, K=None, sigma_inactive=0.01, sigma_active=0.5, fixedActiveValue=None):
    """
    Generates a K-sparse signal with N-K inactive components, chosen randomly. The inactive components 
    are nearly zero, while the active components are either a fixed value or generated from a Gaussian
    distribution.

    Parameters:
    ----------
    N : int
        The total number of components in the signal.
    
    K : int, optional
        The number of active (non-zero) components in the signal. If not provided, defaults to 10% of `N`.
    
    sigma_inactive : float, optional (default=0.01)
        The standard deviation of the Gaussian noise added to the inactive components.
    
    sigma_active : float, optional (default=0.5)
        The standard deviation of the Gaussian noise for generating the active components, if `fixedActiveValue` 
        is not provided.
    
    fixedActiveValue : float, optional
        If provided, this fixed value is assigned to all active components instead of generating them randomly.

    Returns:
    -------
    signal : numpy array
        The generated signal of length `N` with `K` active components.
    
    active_indices : numpy array
        The indices of the active components in the signal.

    Notes:
    -----
    - The signal is constructed by first randomly selecting `K` indices as active components.
    - If `fixedActiveValue` is `None`, the active components are drawn from a Gaussian distribution 
      with standard deviation `sigma_active`.
    - Gaussian noise with standard deviation `sigma_inactive` is then added to the inactive components, 
      ensuring they have small random values near zero.

    Example:
    --------

    >>> signal, active_indices = sparseSignal(10, K=3, sigma_inactive=0.01, sigma_active=0.5)
    >>> print(signal)
    [ 0.          0.          0.          0.          0.          0.          0.
      0.          0.          0.45470039]
    >>> print(active_indices)
    [9]

    """

    if K is None:
        N = int(N)
        K = int(0.1 * N)
    else:
        N = int(N)
        K = int(K)

    active_indexes = np.zeros(N)
    active_indexes[:K] = 1
    np.random.shuffle(active_indexes)

    signal = np.zeros(N)
    
    if fixedActiveValue is None:
        # Generate active components with Gaussian noise
        signal[active_indexes == 1] = np.random.randn(K) * sigma_active
    else:
        # Use fixed value for active components
        signal[active_indexes == 1] = fixedActiveValue
    
    # Add Gaussian noise only to inactive components
    signal[active_indexes == 0] += np.random.randn(N - K) * sigma_inactive

    return signal, np.where(active_indexes == 1)[0]
