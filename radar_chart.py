import sys


sys.path.append('./')

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, RegularPolygon
from matplotlib.path import Path
from matplotlib.projections.polar import PolarAxes
from matplotlib.projections import register_projection
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D

from utils import metrics
import config as cfg


def radar_factory(num_vars, frame='circle'):
    """
    Create a radar chart with `num_vars` axes.

    This function creates a RadarAxes projection and registers it.

    Parameters
    ----------
    num_vars : int
        Number of variables for radar chart.
    frame : {'circle', 'polygon'}
        Shape of frame surrounding axes.

    """
    # calculate evenly-spaced axis angles
    theta = np.linspace(0, 2*np.pi, num_vars, endpoint=False)

    class RadarTransform(PolarAxes.PolarTransform):

        def transform_path_non_affine(self, path):
            # Paths with non-unit interpolation steps correspond to gridlines,
            # in which case we force interpolation (to defeat PolarTransform's
            # autoconversion to circular arcs).
            if path._interpolation_steps > 1:
                path = path.interpolated(num_vars)
            return Path(self.transform(path.vertices), path.codes)

    class RadarAxes(PolarAxes):

        name = 'radar'
        # use 1 line segment to connect specified points
        RESOLUTION = 1
        PolarTransform = RadarTransform

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # rotate plot such that the first axis is at the top
            self.set_theta_zero_location('N')

        def fill(self, *args, closed=True, **kwargs):
            """Override fill so that line is closed by default"""
            return super().fill(closed=closed, *args, **kwargs)

        def plot(self, *args, **kwargs):
            """Override plot so that line is closed by default"""
            lines = super().plot(*args, **kwargs)
            for line in lines:
                self._close_line(line)

        def _close_line(self, line):
            x, y = line.get_data()
            # FIXME: markers at x[0], y[0] get doubled-up
            if x[0] != x[-1]:
                x = np.append(x, x[0])
                y = np.append(y, y[0])
                line.set_data(x, y)

        def set_varlabels(self, labels):
            self.set_thetagrids(np.degrees(theta), labels)

        def _gen_axes_patch(self):
            # The Axes patch must be centered at (0.5, 0.5) and of radius 0.5
            # in axes coordinates.
            if frame == 'circle':
                return Circle((0.5, 0.5), 0.5)
            elif frame == 'polygon':
                return RegularPolygon((0.5, 0.5), num_vars,
                                      radius=.5, edgecolor="k")
            else:
                raise ValueError("Unknown value for 'frame': %s" % frame)

        def _gen_axes_spines(self):
            if frame == 'circle':
                return super()._gen_axes_spines()
            elif frame == 'polygon':
                # spine_type must be 'left'/'right'/'top'/'bottom'/'circle'.
                spine = Spine(axes=self,
                              spine_type='circle',
                              path=Path.unit_regular_polygon(num_vars))
                # unit_regular_polygon gives a polygon of radius 1 centered at
                # (0, 0) but we want a polygon of radius 0.5 centered at (0.5,
                # 0.5) in axes coordinates.
                spine.set_transform(Affine2D().scale(.5).translate(.5, .5)
                                    + self.transAxes)
                return {'polar': spine}
            else:
                raise ValueError("Unknown value for 'frame': %s" % frame)

    register_projection(RadarAxes)
    return theta


def convert_data(data_list):
    new_data_list = []

    #for i in range(len(data_list)):
    #    for j in range(len(data_list[i])):
    #        data_list[i][j] = (data_list[i][j] - min(data_list[i])) / max(data_list[i])

    for i in range(len(data_list[0])):
        new_data = []
        for j in range(len(data_list)):
            new_data.append(data_list[j][i])
        new_data_list.append(new_data)

    return new_data_list


def create_radar_chart(gt, outputs):
    labels = []

    sum_gt = int(metrics.total_sum(gt))

    rmses = []
    rmses_over_mean = []
    time_cors = []
    total_prs = []
    mean_fld_cors = []
    fld_cor_total_sum = []

    # define output metrics
    for output_name, output in outputs.items():
        # append values
        labels.append(output_name)
        rmses.append(metrics.rmse(gt, output))
        rmses_over_mean.append(metrics.rmse_over_mean(gt, output))
        time_cors.append(metrics.timcor(gt, output))
        total_prs.append(abs(sum_gt - int(metrics.total_sum(output))))
        mean_fld_cors.append(metrics.timmean_fldor(gt, output))
        fld_cor_total_sum.append(metrics.fldor_timsum(gt, output))

    N = 6
    theta = radar_factory(N, frame='circle')

    spoke_labels = ['RMSE', 'AME', 'Temporal Correlation', 'Spatial Correlation', 'Correlation Sum', 'Total Precipitation']

    data_list = [rmses, rmses_over_mean, time_cors, mean_fld_cors, fld_cor_total_sum, total_prs]
    data = convert_data(data_list)

    fig, ax = plt.subplots(figsize=(6, 6),
                           subplot_kw=dict(projection='radar'))
    fig.subplots_adjust(wspace=0.25, hspace=0.20, top=0.85, bottom=0.05)

    # Plot the four cases from the example data on separate axes
    for d, color in zip(data, cfg.graph_colors):
        ax.plot(theta, d, color=color)
        ax.fill(theta, d, facecolor=color, alpha=0.25, label='_nolegend_')
    ax.set_varlabels(spoke_labels)

    # add legend relative to top-left plot
    legend = ax.legend(labels, loc=(0.9, .95),
                       labelspacing=0.1, fontsize='small')

    fig.text(0.5, 0.965, 'Radar chart of evaluation metrics',
             horizontalalignment='center', color='black', weight='bold',
             size='large')

    plt.savefig('{}/radar/RadarChart.pdf'.format(cfg.evaluation_dirs[0]), bbox_inches="tight")
