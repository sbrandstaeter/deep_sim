import os

import numpy as np
import tensorflow as tf
import gpflow


def configure_tensorflow_device(device_preference: str = "auto") -> str:
    """
    Configure TensorFlow device placement.

    Args:
        device_preference:
            - "auto": use GPU if available, else CPU
            - "gpu": require GPU
            - "cpu": force CPU

    Returns:
        TensorFlow device string, e.g. "/GPU:0" or "/CPU:0"
    """
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

    gpus = tf.config.list_physical_devices("GPU")
    cpus = tf.config.list_physical_devices("CPU")

    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    if device_preference == "cpu":
        print("Forcing CPU execution.")
        return "/CPU:0"

    if device_preference == "gpu":
        if not gpus:
            raise RuntimeError(
                "GPU was requested, but TensorFlow does not see any GPU devices."
            )
        print(f"Using GPU: {gpus[0].name}")
        return "/GPU:0"

    # auto
    if gpus:
        print(f"Using GPU: {gpus[0].name}")
        return "/GPU:0"

    print("No GPU detected by TensorFlow. Falling back to CPU.")
    if not cpus:
        raise RuntimeError("TensorFlow does not report any CPU devices either.")
    return "/CPU:0"


def build_gpflow_gpr(
    X_train_scaled: np.ndarray,
    y_train_log: np.ndarray,
    num_features: int,
) -> gpflow.models.GPR:
    """
    Build a GPflow exact GPR model.

    GPflow expects:
      X: shape [N, D]
      Y: shape [N, P]
    """
    X_tf = X_train_scaled.astype(np.float64)
    y_tf = y_train_log.reshape(-1, 1).astype(np.float64)

    # kernel = gpflow.kernels.Constant(
    #     variance=float(num_features)
    # ) * gpflow.kernels.SquaredExponential(
    #     lengthscales=np.ones(num_features, dtype=np.float64),
    #     variance=1.0,
    # )
    kernel = gpflow.kernels.SquaredExponential(
        lengthscales=np.ones(num_features, dtype=np.float64),
        variance=1.0,
    )

    model = gpflow.models.GPR(
        data=(X_tf, y_tf),
        kernel=kernel,
        mean_function=None,
        noise_variance=1e-5,
    )

    gpflow.set_trainable(model.likelihood.variance, False)
    return model
