import numpy as np
from typing import Tuple, Optional

def gall_filter(x: np.ndarray, d: np.ndarray, M: Optional[int] = None, beta: float = 1.0, 
                alpha: float = 0.0, epsi: float = 1e-3, k0: Optional[np.ndarray] = None, 
                h0: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    '''
    Gradient Adaptive Laguerre Lattice (GALL) Filter.
    Converted from MATLAB Fezjo & Lev-Ari (1997) implementation.
    Citation:
    Ikaro Silva (2026). Gradient Adaptive Laguerre Lattice Filter (https://www.mathworks.com/matlabcentral/fileexchange/19817-gradient-adaptive-laguerre-lattice-filter), MATLAB Central File Exchange. Retrieved July 8, 2026.

    Parameters
    ----------
    x : np.ndarray
        Measurement data (e.g., Accelerometer input).
    d : np.ndarray
        Desired response (e.g., Noisy ECG). Same size as x.
    M : int, optional
        Filter order. If None, the filter operates in Freeze Mode (Mode 2) using k0 and h0.
    beta : float
        Forgetting factor (0 <= beta <= 1).
    alpha : float
        Pole magnitude (0 <= alpha < 1). alpha=0 is standard GAL.
    epsi : float
        Small positive constant for initialization.
    k0 : np.ndarray, optional
        Initial reflection coefficients vector.
    h0 : np.ndarray, optional
        Initial FIR ladder coefficients vector.

    Returns
    -------
    err : np.ndarray
        Error signal of shape (N,), i.e. the output of the last ladder stage (filtered ECG).
    y : np.ndarray
        Filter prediction (Noise estimate).
    h : np.ndarray
        FIR coefficients of the ladder.
    k : np.ndarray
        Lattice reflection coefficients.
    '''

    x = np.asarray(x, dtype=np.float64).ravel()
    d = np.asarray(d, dtype=np.float64).ravel()
    if x.size != d.size:
        raise ValueError(f"x and d must have the same length, got {x.size} and {d.size}")
    if not 0.0 <= alpha < 1.0:
        raise ValueError("alpha must lie in [0, 1)")

    N = x.size

    
    #LEARNING MODE
    
    if M is not None:
        f = np.zeros(M + 1)
        b = np.zeros(M + 1)
        bt = np.zeros(M + 1)
        Q = np.zeros(M + 1) + epsi
        delta = np.zeros(M + 1)
        
        y = np.zeros(N)
        D = np.zeros(M + 2)
        err_out = np.zeros(N)
        err_stage = np.zeros(M + 2)
        
        b_old = np.zeros(M + 1)
        bt_old = np.zeros(M + 1)
        
        k = np.zeros(M + 1) if k0 is None else np.array(k0, dtype=np.float64, copy=True)
        h = np.zeros(M + 2) if h0 is None else np.array(h0, dtype=np.float64, copy=True)
        
        sqrt_alpha = np.sqrt(1 - alpha**2)

        for n in range(N):
            # Input preparation with Laguerre pole
            f[0] = alpha * b_old[0] + sqrt_alpha * x[n]
            b[0] = f[0]
            Q[0] = beta * Q[0] + b[0]**2
            err_stage[0] = d[n]
            
            # Save state for next iteration
            bt_old[:] = bt[:]
            b_old[:] = b[:]

            # Lattice Section
            for m in range(1, M + 1):
                bt[m-1] = b_old[m-1] + alpha * (bt_old[m-1] - b[m-1])
                Q[m-1] = beta * Q[m-1] + (f[m-1]**2 + bt[m-1]**2) / 2.0
                delta[m] = beta * delta[m] + f[m-1] * bt[m-1]
                
                k[m] = delta[m] / Q[m-1] if Q[m-1] > 0 else 0
                
                f[m] = f[m-1] - k[m] * bt[m-1]
                b[m] = bt[m-1] - k[m] * f[m-1]

            Q[-1] = beta * Q[-1] + (f[-1]**2 + bt[-1]**2) / 2.0

            # Ladder Section
            for m in range(M + 1):
                D[m+1] = beta * D[m+1] + err_stage[m] * b[m]
                h[m+1] = D[m+1] / Q[m] if Q[m] > 0 else 0
                err_stage[m+1] = err_stage[m] - h[m+1] * b[m]

            err_out[n] = err_stage[-1]
            y[n] = np.dot(h[1:], b[:M+1])

        return err_out, y, h, k

    
    #FREEZE (OUTPUT) MODE
    
    else:
        if k0 is None or h0 is None:
            raise ValueError("In Mode 2 (Freeze), k0 and h0 must be provided.")
            
        k = np.array(k0, dtype=np.float64, copy=True)
        h = np.array(h0, dtype=np.float64, copy=True)
        M_freeze = len(k) - 1
        if h.size != M_freeze + 2:
            raise ValueError(f"h0 must have length len(k0) + 1 = {M_freeze + 2}, got {h.size}")
        
        f = np.zeros(M_freeze + 1)
        b = np.zeros(M_freeze + 1)
        bt = np.zeros(M_freeze + 1)
        y = np.zeros(N)
        err_out = np.zeros(N)
        err_stage = np.zeros(M_freeze + 2)
        
        b_old = np.zeros(M_freeze + 1)
        bt_old = np.zeros(M_freeze + 1)
        
        sqrt_alpha = np.sqrt(1 - alpha**2)

        for n in range(N):
            f[0] = alpha * b_old[0] + sqrt_alpha * x[n]
            b[0] = f[0]
            err_stage[0] = d[n]
            
            bt_old[:] = bt[:]
            b_old[:] = b[:]

            # Lattice Section
            for m in range(1, M_freeze + 1):
                bt[m-1] = b_old[m-1] + alpha * (bt_old[m-1] - b[m-1])
                f[m] = f[m-1] - k[m] * bt[m-1]
                b[m] = bt[m-1] - k[m] * f[m-1]

            # Ladder Section
            for m in range(M_freeze + 1):
                err_stage[m+1] = err_stage[m] - h[m+1] * b[m]

            err_out[n] = err_stage[-1]
            y[n] = np.dot(h[1:], b[:M_freeze+1])

        return err_out, y, h, k
