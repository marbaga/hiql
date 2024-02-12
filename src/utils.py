from typing import Iterable, Optional
import os
import numpy as np
import jax.numpy as jnp
import wandb
from aim import Run

Array = jnp.ndarray


def prepare_video(v, n_cols=None):
    orig_ndim = v.ndim
    if orig_ndim == 4:
        v = v[None, ]

    _, t, c, h, w = v.shape

    if v.dtype == np.uint8:
        v = np.float32(v) / 255.

    if n_cols is None:
        if v.shape[0] <= 4:
            n_cols = 2
        elif v.shape[0] <= 9:
            n_cols = 3
        else:
            n_cols = 6
    if v.shape[0] % n_cols != 0:
        len_addition = n_cols - v.shape[0] % n_cols
        v = np.concatenate(
            (v, np.zeros(shape=(len_addition, t, c, h, w))), axis=0)
    n_rows = v.shape[0] // n_cols

    v = np.reshape(v, newshape=(n_rows, n_cols, t, c, h, w))
    v = np.transpose(v, axes=(2, 0, 4, 1, 5, 3))
    v = np.reshape(v, newshape=(t, n_rows * h, n_cols * w, c))

    return v


def save_video(label, step, tensor, fps=15, n_cols=None):
    def _to_uint8(t):
        # If user passes in uint8, then we don't need to rescale by 255
        if t.dtype != np.uint8:
            t = (t * 255.0).astype(np.uint8)
        return t
    if tensor.dtype in [object]:
        tensor = [_to_uint8(prepare_video(t, n_cols)) for t in tensor]
    else:
        tensor = prepare_video(tensor, n_cols)
        tensor = _to_uint8(tensor)

    tensor = tensor.transpose(0, 3, 1, 2)
    return wandb.Video(tensor, fps=15, format='mp4')


def record_video(label, step, renders=None, n_cols=None, skip_frames=1):
    max_length = max([len(render) for render in renders])
    for i, render in enumerate(renders):
        renders[i] = np.concatenate([render, np.zeros((max_length - render.shape[0], *render.shape[1:]), dtype=render.dtype)], axis=0)
        renders[i] = renders[i][::skip_frames]
    renders = np.array(renders)
    return save_video(label, step, renders, n_cols=n_cols)


class Logger:
    def __init__(self, kwargs):
        self.kwargs = kwargs
        self.path = {k: os.path.join(kwargs.working_dir, f'{k}.csv') for k in ['train', 'eval']}
        self.header = {'train': None, 'eval': None}
        self.file = {'train': None, 'eval': None}
        self.disallowed_types = (wandb.Image, wandb.Video, wandb.Histogram)
        self.run = Run(repo = kwargs.working_dir, experiment='test')
        self.run['hparams'] = kwargs.to_dict()

    def log(self, row, step, mode='eval'):
        assert mode in ['eval', 'train']
        row['step'] = step
        if self.file[mode] is None:
            self.file[mode] = open(self.path[mode], 'w')
            if self.header[mode] is None:
                self.header[mode] = [k for k, v in row.items() if not isinstance(v, self.disallowed_types)]
                self.file[mode].write(','.join(self.header[mode]) + '\n')
        filtered_row = {k: v for k, v in row.items() if not isinstance(v, self.disallowed_types)}
        [self.run.track(v, name=k) for k, v in filtered_row.items()]
        self.file[mode].write(','.join([str(filtered_row.get(k, '')) for k in self.header[mode]]) + '\n')
        self.file[mode].flush()

    def close(self):
        [f.close() for f in self.file.values() if f is not None]
