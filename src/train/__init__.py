'''
Training pipeline.

    signal_selection    maps a model name onto the tensor representation it consumes,
                        together with a matched encode / decode pair
    training            representation-agnostic dataset, trainer, losses and metrics
    cli                 command line entry point

This package depends on `filters`; the dependency never runs the other way.
'''
