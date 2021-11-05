from typing import Dict
import tensorflow as tf

class MyLogger:
    """Simple tensorboard logger."""
    
    def __init__(
        self,
        logdir,
    ):
        self.file_writer = tf.summary.create_file_writer(logdir)
        self.file_writer.set_as_default()

        self.step = 0

    def write(self, logs: Dict[str, float]):
        for key, val in logs.items():
            tf.summary.scalar(key, data=val, step=self.step)

        self.step += 1