import numpy as np
from typing import Tuple

try:
    from .gall_filter import gall_filter
    from .reference import reference_matrix
except ImportError:
    from gall_filter import gall_filter
    from reference import reference_matrix


def _resolve_reference(ref_x=None, ref_y=None, ref_z=None, reference=None,
                       SVM_condition: bool = False) -> np.ndarray:
    '''
    Reference channels as a (num_channels, N) matrix, from either calling convention.

    `reference` takes an array already in that layout and accepts any number of channels,
    which is what two accelerometers require: an ECG is a difference of two potentials and
    a motion artefact arises at the interface of each electrode, so one sensor can explain
    only one of the two terms.

    The three separate axes remain accepted so that existing callers keep working.
    '''
    if reference is not None:
        if ref_x is not None or ref_y is not None or ref_z is not None:
            raise ValueError('pass either reference or the separate axes, not both')
        return reference_matrix(reference, svm=SVM_condition)

    if ref_x is None or ref_y is None or ref_z is None:
        raise ValueError('three axes are required when reference is not given')
    return reference_matrix(np.asarray(ref_x).ravel(),
                            np.asarray(ref_y).ravel(),
                            np.asarray(ref_z).ravel(), svm=SVM_condition)


def _regressor(ref: np.ndarray, n: int, filter_order: int) -> np.ndarray:
    '''
    Causal regressor u(n) = [r(n), r(n-1), ..., r(n-filter_order+1)] for every reference channel,
    flattened channel by channel.
    '''
    window = ref[:, n - filter_order + 1 : n + 1][:, ::-1]
    return window.reshape(-1)


def modified_lms_anc(raw_ecg: np.ndarray, lowpass_ecg: np.ndarray, ref_x=None, ref_y=None, ref_z=None, mu: float = 0.001, filter_order: int = 5, SVM_condition: bool = False, reference=None) -> np.ndarray: #[1]
    '''
    Function to perform modified multi-reference LMS adaptive filtering.
    Uses a lowpass-filtered ECG for weight adaptation and raw ECG for final noise cancellation.

    Parameters
    ----------
    raw_ecg : np.ndarray
        The raw ECG signal to be denoised.
    lowpass_ecg : np.ndarray
        The lowpass-filtered ECG signal used for weight adaptation.
    ref_x : np.ndarray
        The reference signal from the x-axis accelerometer.
    ref_y : np.ndarray
        The reference signal from the y-axis accelerometer.
    ref_z : np.ndarray
        The reference signal from the z-axis accelerometer.
    mu : float, optional
        The step size for the LMS algorithm. Default is 0.001.
    filter_order : int, optional
        The order of the adaptive filter. Default is 5.
    SVM_condition : bool, optional
        If True, uses Signal Vector Magnitude (SVM) instead of 3 separate axes. Default is False.
    
    Returns
    -------
    clean_ecg : np.ndarray
        The denoised ECG signal after adaptive filtering.
    '''

    raw_ecg = np.asarray(raw_ecg, dtype=np.float64).ravel()
    lowpass_ecg = np.asarray(lowpass_ecg, dtype=np.float64).ravel()
    ref = _resolve_reference(ref_x, ref_y, ref_z, reference, SVM_condition)

    N = raw_ecg.size
    if lowpass_ecg.size != N or ref.shape[1] != N:
        raise ValueError("raw_ecg, lowpass_ecg and the reference channels must have the same length")
    if filter_order > N:
        raise ValueError(f"filter_order ({filter_order}) exceeds signal length ({N})")

    w = np.zeros(ref.shape[0] * filter_order)

    y = np.zeros(N)
    e_learning = np.zeros(N)
    clean_ecg = np.copy(raw_ecg)

    for n in range(filter_order - 1, N):
        u_n = _regressor(ref, n, filter_order)

        y[n] = np.dot(w, u_n)
        e_learning[n] = lowpass_ecg[n] - y[n]
        w += 2 * mu * e_learning[n] * u_n
        clean_ecg[n] = raw_ecg[n] - y[n]

    return clean_ecg


def rls_anc(raw_ecg: np.ndarray, ref_x=None, ref_y=None, ref_z=None, filter_order: int = 5, lam: float = 0.99, delta: float = 1.0, SVM_condition: bool = False, reference=None) -> np.ndarray: #[2]
    '''
    Function to perform multi-reference RLS adaptive filtering.
    Uses raw ECG and accelerometer data for noise cancellation.

    Parameters
    ----------
    raw_ecg : np.ndarray
        The raw ECG signal to be denoised.
    ref_x : np.ndarray
        The reference signal from the x-axis accelerometer.
    ref_y : np.ndarray
        The reference signal from the y-axis accelerometer.
    ref_z : np.ndarray
        The reference signal from the z-axis accelerometer.
    filter_order : int, optional
        The order of the adaptive filter. Default is 5.
    lam : float, optional
        The forgetting factor (lambda) for RLS. Default is 0.99.
    delta : float, optional
        Initialization constant for the inverse correlation matrix P. Default is 1.0.
    SVM_condition : bool, optional
        If True, uses Signal Vector Magnitude (SVM) instead of 3 separate axes. Default is False.
    
    Returns
    -------
    clean_ecg : np.ndarray
        The denoised ECG signal after adaptive filtering.
    '''

    raw_ecg = np.asarray(raw_ecg, dtype=np.float64).ravel()
    ref = _resolve_reference(ref_x, ref_y, ref_z, reference, SVM_condition)

    N = raw_ecg.size
    if ref.shape[1] != N:
        raise ValueError("raw_ecg and the reference channels must have the same length")
    if filter_order > N:
        raise ValueError(f"filter_order ({filter_order}) exceeds signal length ({N})")
    if not 0.0 < lam <= 1.0:
        raise ValueError("lam must lie in (0, 1]")

    M = ref.shape[0] * filter_order
    w = np.zeros(M)
    P = np.eye(M) * delta

    y = np.zeros(N)
    clean_ecg = np.copy(raw_ecg)

    for n in range(filter_order - 1, N):
        u_n = _regressor(ref, n, filter_order)

        y[n] = np.dot(w, u_n)
        clean_ecg[n] = raw_ecg[n] - y[n]

        Pi = P @ u_n
        k = Pi / (lam + np.dot(u_n, Pi))
        w += k * clean_ecg[n]
        P = (P - np.outer(k, Pi)) / lam
        P = 0.5 * (P + P.T)

    return clean_ecg


def blms_ecg_filter(raw_ecg: np.ndarray, ref_x=None, ref_y=None, ref_z=None, L: int = 10, mu: float = 0.01, filter_order: int = 32, SVM_condition: bool = False, reference=None) -> np.ndarray: #[4]
    '''
    Block Least Mean Square (BLMS) adaptive filter for ECG motion artifact removal.

    Parameters
    ----------
    raw_ecg : np.ndarray
        The noisy ECG signal (primary input).
    ref_x : np.ndarray
        The reference signal from accelerometer (should be same length as raw_ecg).
    ref_y : np.ndarray
        The reference signal from accelerometer (should be same length as raw_ecg).
    ref_z : np.ndarray
        The reference signal from accelerometer (should be same length as raw_ecg).
    L : int
        Block size for weight updates.
    mu : float
        Step size.
    filter_order : int
        Number of filter taps per reference channel.
    SVM_condition : bool, optional
        If True, uses Signal Vector Magnitude (SVM) instead of 3 separate axes. Default is False.

    Returns
    -------
    e : np.ndarray
        The denoised ECG signal.
    '''

    raw_ecg = np.asarray(raw_ecg, dtype=np.float64).ravel()
    ref = _resolve_reference(ref_x, ref_y, ref_z, reference, SVM_condition)

    N = raw_ecg.size
    if ref.shape[1] != N:
        raise ValueError("raw_ecg and the reference channels must have the same length")
    if filter_order > N:
        raise ValueError(f"filter_order ({filter_order}) exceeds signal length ({N})")
    if L < 1:
        raise ValueError("L must be a positive integer")

    w = np.zeros(ref.shape[0] * filter_order)
    e = np.copy(raw_ecg)

    start = filter_order - 1
    for k in range(start, N, L):
        weight_update_sum = np.zeros_like(w)

        for idx in range(k, min(k + L, N)):
            u_window = _regressor(ref, idx, filter_order)

            y = np.dot(w, u_window)
            e[idx] = raw_ecg[idx] - y
            weight_update_sum += u_window * e[idx]

        w += (mu / L) * weight_update_sum

    return e


def hybrid_gall_kalman_ecg_filter(
    raw_ecg: np.ndarray,
    ref_x=None,
    ref_y=None,
    ref_z=None,
    M_gall: int = 5,
    beta_gall: float = 1.0,
    alpha_gall: float = 0.0,
    epsi_gall: float = 1e-3,
    lambda_K: float = 0.99,
    delta_K: float = 1.0,
    reference=None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]: #[3]
    '''
    Hybrid Filter Fusion: 3x Parallel GALL Filters combined by an Unforced Kalman Filter.
    
    Parameters
    ----------
    raw_ecg : np.ndarray
        The noisy ECG signal (Primary input).
    ref_x, ref_y, ref_z : np.ndarray, optional
        The reference signals from one 3-axis accelerometer.
    reference : np.ndarray, optional
        Reference channels of shape (num_channels, N), for any number of accelerometers.
    M_gall : int, optional
        Filter order for the local GALL filters. Default is 5.
    beta_gall : float, optional
        Forgetting factor for the GALL filters. Default is 1.0.
    alpha_gall : float, optional
        Laguerre pole magnitude for GALL filters. Default is 0.0.
    epsi_gall : float, optional
        Initialization constant for GALL filters. Default is 1e-3.
    lambda_K : float, optional
        Scalar forgetting factor for the Unforced Kalman Filter (RLS combiner). Default is 0.99.
    delta_K : float, optional
        Initialization constant for the Kalman state correlation matrix K. Default is 1.0.

    Returns
    -------
    clean_ecg : np.ndarray
        The final denoised ECG signal.
    y_total : np.ndarray
        The global noise estimate computed by the Kalman Filter.
    weight_history : np.ndarray
        Matrix of shape (N, num_channels) tracking the Kalman weights over time.
    '''

    raw_ecg = np.asarray(raw_ecg, dtype=np.float64).ravel()
    ref = _resolve_reference(ref_x, ref_y, ref_z, reference, SVM_condition=False)

    N = raw_ecg.size
    if ref.shape[1] != N:
        raise ValueError("raw_ecg and the reference channels must have the same length")
    if not 0.0 < lambda_K <= 1.0:
        raise ValueError("lambda_K must lie in (0, 1]")

    
    #Parallel Noise Estimation (GALL)

    # One GALL filter per reference channel, however many there are
    C = ref.shape[0]
    estimates = np.empty((C, N))
    for channel in range(C):
        _, estimates[channel], _, _ = gall_filter(
            x=ref[channel], d=raw_ecg, M=M_gall, beta=beta_gall,
            alpha=alpha_gall, epsi=epsi_gall)


    #Unforced Kalman Filter Fusion

    w = np.ones(C) / C           # State vector, one weight per channel
    K_mat = np.eye(C) * delta_K  # State error correlation matrix (K[n])

    y_total = np.zeros(N)
    clean_ecg = np.zeros(N)
    weight_history = np.zeros((N, C))

    for n in range(N):
        # 1. Input vector u[n] holding the GALL output of every channel
        u_n = estimates[:, n]
        
        # 2. Calculate the global noise estimate y_total[n] using the current weights w
        y_total[n] = np.dot(w, u_n)
        
        
        #Global Summer
        
        clean_ecg[n] = raw_ecg[n] - y_total[n]
        
        
        # KALMAN STATE UPDATE (RLS Logic)
        
        # Równania śledzące z zerowym szumem procesu (process noise = 0)
        Pi = K_mat @ u_n
        
        # Calculate Kalman Gain
        g_n = Pi / (lambda_K + np.dot(u_n, Pi))
        
        # Update weights based on the global error (clean_ecg)
        w = w + g_n * clean_ecg[n]
        
        # Update the correlation matrix K[n]
        K_mat = (K_mat - np.outer(g_n, Pi)) / lambda_K
        K_mat = 0.5 * (K_mat + K_mat.T)
        
        # Save the weight history for analysis
        weight_history[n, :] = w

    return clean_ecg, y_total, weight_history
