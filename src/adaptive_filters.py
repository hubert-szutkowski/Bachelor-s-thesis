import numpy as np
from gall_filter import gall_filter

def modified_lms_anc(raw_ecg: np.ndarray, lowpass_ecg: np.ndarray, ref_x: np.ndarray, ref_y: np.ndarray, ref_z: np.ndarray, mu: float = 0.001, filter_order: int = 5, SVM_condition: bool = False) -> np.ndarray:
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
    
    N = len(raw_ecg)
    
    if SVM_condition:
        ref_svm = np.sqrt(ref_x**2 + ref_y**2 + ref_z**2)
        num_channels = 1
    else:
        num_channels = 3
        
    w = np.zeros(num_channels * filter_order)
    
    y = np.zeros(N)
    e_learning = np.zeros(N)
    clean_ecg = np.zeros(N)
    
    for n in range(filter_order, N):
        if SVM_condition:
            u_n = ref_svm[n - filter_order : n][::-1]
        else:
            x_n = ref_x[n - filter_order : n][::-1]
            y_n = ref_y[n - filter_order : n][::-1]
            z_n = ref_z[n - filter_order : n][::-1]
            u_n = np.concatenate((x_n, y_n, z_n))
        
        y[n] = np.dot(w, u_n)
        e_learning[n] = lowpass_ecg[n] - y[n]
        w = w + 2 * mu * e_learning[n] * u_n
        clean_ecg[n] = raw_ecg[n] - y[n]
        
    return clean_ecg


def rls_anc(raw_ecg: np.ndarray, ref_x: np.ndarray, ref_y: np.ndarray, ref_z: np.ndarray, filter_order: int = 5, lam: float = 0.99, delta: float = 1.0, SVM_condition: bool = False) -> np.ndarray:
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
    
    N = len(raw_ecg)
    
    if SVM_condition:
        ref_svm = np.sqrt(ref_x**2 + ref_y**2 + ref_z**2)
        num_channels = 1
    else:
        num_channels = 3
        
    M = num_channels * filter_order
    w = np.zeros(M)
    P = np.eye(M) * delta
    
    y = np.zeros(N)
    clean_ecg = np.zeros(N)
    
    for n in range(filter_order, N):
        if SVM_condition:
            u_n = ref_svm[n - filter_order : n][::-1]
        else:
            x_n = ref_x[n - filter_order : n][::-1]
            y_n = ref_y[n - filter_order : n][::-1]
            z_n = ref_z[n - filter_order : n][::-1]
            u_n = np.concatenate((x_n, y_n, z_n))
        
        y[n] = np.dot(w, u_n)
        clean_ecg[n] = raw_ecg[n] - y[n]
        
        Pi = np.dot(P, u_n)
        k = Pi / (lam + np.dot(u_n, Pi))
        w = w + k * clean_ecg[n]
        P = (P - np.outer(k, Pi)) / lam
        
    return clean_ecg



def blms_ecg_filter(raw_ecg: np.ndarray, ref_x: np.ndarray, ref_y: np.ndarray, ref_z: np.ndarray, L: int = 10, mu: float = 0.01, filter_order: int = 32, SVM_condition: bool = False) -> np.ndarray:
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
        Number of filter taps.
    SVM_condition : bool, optional
        If True, uses Signal Vector Magnitude (SVM) instead of 3 separate axes. Default is False.
    Returns
    -------
    e : np.ndarray
        The denoised ECG signal.
    '''
    N = len(raw_ecg)
    # Weights vector initialized to zeros
    w = np.zeros(filter_order)
    
    e = np.zeros(N)

    if SVM_condition:
        ref_svm = np.sqrt(ref_x**2 + ref_y**2 + ref_z**2)
    
    # We process in blocks of size L
    for k in range(0, N - filter_order, L):
        # Accumulator for weight update: mu * sum(x(i) * e(i))
        weight_update_sum = np.zeros(filter_order)
        
        # Process samples within the block
        for i in range(L):
            idx = k + i
            if idx + filter_order >= N:
                break
                
            # Current window of reference signal
            if SVM_condition:
                u_window = ref_svm[idx : idx + filter_order][::-1]
            else:
                x_window = ref_x[idx : idx + filter_order][::-1]
                y_window = ref_y[idx : idx + filter_order][::-1]
                z_window = ref_z[idx : idx + filter_order][::-1]
                u_window = np.concatenate((x_window, y_window, z_window))
            
            # Output: y(k) = w^T * x
            y = np.dot(w, u_window)
            
            # Error: e(k) = r(k) - y(k)
            e[idx] = raw_ecg[idx] - y
            
            # Accumulate: x(i) * e(i)
            weight_update_sum += u_window * e[idx]
            
        # Block weight update: w(k+1) = w(k) + mu * sum(...)
        w = w + mu * weight_update_sum
        
    return e

import numpy as np
from typing import Tuple

def hybrid_gall_kalman_ecg_filter(
    raw_ecg: np.ndarray, 
    ref_x: np.ndarray, 
    ref_y: np.ndarray, 
    ref_z: np.ndarray,
    M_gall: int = 5,
    beta_gall: float = 1.0,
    alpha_gall: float = 0.0,
    epsi_gall: float = 1e-3,
    lambda_K: float = 0.99,
    delta_K: float = 1.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    '''
    Hybrid Filter Fusion: 3x Parallel GALL Filters combined by an Unforced Kalman Filter.
    
    Parameters
    ----------
    raw_ecg : np.ndarray
        The noisy ECG signal (Primary input).
    ref_x, ref_y, ref_z : np.ndarray
        The reference signals from the 3-axis accelerometer.
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
        Matrix of shape (N, 3) tracking the Kalman weights for X, Y, Z axes over time.
    '''
    
    N = len(raw_ecg)

    
    #Parallel Noise Estimation (GALL)
    
    # Gall filter for each axis 
    _, y_x, _, _ = gall_filter(x=ref_x, d=raw_ecg, M=M_gall, beta=beta_gall, alpha=alpha_gall, epsi=epsi_gall)
    _, y_y, _, _ = gall_filter(x=ref_y, d=raw_ecg, M=M_gall, beta=beta_gall, alpha=alpha_gall, epsi=epsi_gall)
    _, y_z, _, _ = gall_filter(x=ref_z, d=raw_ecg, M=M_gall, beta=beta_gall, alpha=alpha_gall, epsi=epsi_gall)

    
    #Unforced Kalman Filter Fusion
    
    w = np.ones(3) / 3.0         # State vector (Wagi dla X, Y, Z) 
    K_mat = np.eye(3) * delta_K  # State error correlation matrix (K[n])
    
    y_total = np.zeros(N)
    clean_ecg = np.zeros(N)
    weight_history = np.zeros((N, 3))

    for n in range(N):
        # 1. Input vector u[n] containing the GALL outputs for X, Y, Z axes
        u_n = np.array([y_x[n], y_y[n], y_z[n]])
        
        # 2. Calculate the global noise estimate y_total[n] using the current weights w
        y_total[n] = np.dot(w, u_n)
        
        
        #Global Summer
        
        clean_ecg[n] = raw_ecg[n] - y_total[n]
        
        
        # KALMAN STATE UPDATE (RLS Logic)
        
        # Równania śledzące z zerowym szumem procesu (process noise = 0)
        Pi = np.dot(K_mat, u_n)
        
        # Calculate Kalman Gain
        g_n = Pi / (lambda_K + np.dot(u_n, Pi))
        
        # Update weights based on the global error (clean_ecg)
        w = w + g_n * clean_ecg[n]
        
        # Update the correlation matrix K[n]
        K_mat = (K_mat - np.outer(g_n, Pi)) / lambda_K
        
        # Save the weight history for analysis
        weight_history[n, :] = w
        
    return clean_ecg, y_total, weight_history