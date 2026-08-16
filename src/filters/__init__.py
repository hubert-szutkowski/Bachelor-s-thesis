'''
Denoising methods.

Three families share this package:

    static_filters        FIR, IIR, moving average / median, wavelet, EMD
    adaptive_filters      LMS, RLS, BLMS and the GALL + Kalman hybrid driven by the
    gall_filter           accelerometer reference channel
    *_model, miemd_cnn    deep architectures (PyTorch)

`signal_transforms` holds the NumPy / SciPy signal processing shared by the deep
architectures and by the training pipeline. It is deliberately free of any PyTorch import,
so building a cache or computing metrics does not require a CUDA-capable install.

Submodules are not imported eagerly: `import filters` must stay cheap and must not pull in
PyTorch for callers that only need the classical filters.
'''
