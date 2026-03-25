import numpy as np
import tensorflow as tf


def tar_at_far(y_true: np.ndarray, scores: np.ndarray, far: float = 0.01) -> tuple[float, float]:
    y_true = np.asarray(y_true).astype(np.int32).reshape(-1)
    scores = np.asarray(scores).astype(np.float32).reshape(-1)

    neg_scores = scores[y_true == 0]
    pos_scores = scores[y_true == 1]

    if neg_scores.size == 0 or pos_scores.size == 0:
        return float("nan"), float("nan")

    q = 1.0 - float(far)
    q = min(max(q, 0.0), 1.0)
    threshold = float(np.quantile(neg_scores, q, method="higher"))

    tar = float(np.mean(pos_scores >= threshold))
    return tar, threshold


class TarAtFarCallback(tf.keras.callbacks.Callback):
    def __init__(self, dataset: tf.data.Dataset, far: float = 0.01, name: str | None = None):
        super().__init__()
        self.dataset = dataset
        self.far = float(far)
        self.name = name if name is not None else f"tar_at_far_{self.far:g}"

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}

        y_true_all = []
        score_all = []

        for (img_a, img_b), y in self.dataset:
            logits = self.model((img_a, img_b), training=False)
            scores = tf.sigmoid(logits)
            y_true_all.append(tf.reshape(tf.cast(y, tf.float32), (-1,)))
            score_all.append(tf.reshape(tf.cast(scores, tf.float32), (-1,)))

        y_true_np = tf.concat(y_true_all, axis=0).numpy()
        score_np = tf.concat(score_all, axis=0).numpy()

        tar, thr = tar_at_far(y_true_np, score_np, far=self.far)

        logs[self.name] = tar
        logs[f"{self.name}_threshold"] = thr
