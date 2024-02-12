import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import matplotlib.gridspec as gridspec
import math
import numpy as np
from functools import partial


def most_squarelike(n):
    c = int(n ** 0.5)
    while c > 0:
        if n % c in [0, c - 1]:
            return (c, int(math.ceil(n / c)))
        c -= 1

def make_visual_no_image(metrics):
    visualization_methods = [partial(visualize_metric, metric_name=k) for k in metrics.keys()]
    w, h = most_squarelike(len(visualization_methods))
    gs = gridspec.GridSpec(h, w)
    fig = plt.figure(tight_layout=True)
    canvas = FigureCanvas(fig)

    for i in range(len(visualization_methods)):
        wi, hi = i % w, i // w
        ax = fig.add_subplot(gs[hi, wi])
        visualization_methods[i](ax=ax, metrics=metrics)

    plt.tight_layout()
    canvas.draw()
    out_image = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
    out_image = out_image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return out_image


def visualize_metric(ax, metrics, *, metric_name, linestyle='--', marker='o', **kwargs):
    metric = metrics[metric_name]
    ax.plot(metric, linestyle=linestyle, marker=marker, **kwargs)
    ax.set_ylabel(metric_name)
