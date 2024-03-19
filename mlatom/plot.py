'''
  !---------------------------------------------------------------------------! 
  ! plot.py: Plotting routines                                                ! 
  ! Implementations by: Pavlo O. Dral                                         ! 
  !---------------------------------------------------------------------------! 
'''

import numpy as np

class plot(object):
    x2ys = []
    y2ys = []
    x3y = []
    y3y = []
    xfs = []
    yfs = []
    delta = 0.0

    def __init__(self):
        self.xs = []
        self.ys = []

        self.plottype = 'linechart'
        self.trendline = None
        self.normalize = False
        self.plot2Ys = False
        self.plot3Ys = False

        self.plotstart = None
        self.plotend = None

        self.savein = None

        self.title=None
        self.xaxis_caption = ''
        self.yaxis_caption = ''
        self.labels = []
        self.colors = []
        self.edgecolors = []
        self.markers = []
        self.linewidths = []

    def make_figure(self):
        import matplotlib
        import matplotlib.pyplot as plt
        matplotlib.rcParams['axes.linewidth'] = 2

        fig, ax = plt.subplots()
        fig.subplots_adjust(right=0.85)
        lines = []

        if self.colors == []:
            self.colors = [None for xx in self.xs]
        if self.edgecolors == []:
            self.edgecolors = [None for xx in self.xs]
        if self.labels == []:
            self.labels = [None for xx in self.xs]
        if self.linewidths == []:
            self.linewidths = [None for xx in self.xs]
        if self.markers == []:
            self.markers = [None for xx in self.xs]
            
        if self.plottype == 'linechart':
            plt.plot(self.xs[0], self.ys[0], 'ro')
        if self.plottype == 'linechart_without_points':
            for ii in range(len(self.xs)):
                if ii < len(self.labels): label=self.labels[ii]
                else: label=None
                if ii < len(self.linewidths): linewidth=self.linewidths[ii]
                else: linewidth=None
                if ii < len(self.markers): marker=self.markers[ii]
                else: marker=None
                if ii < len(self.colors): color=self.colors[ii]
                else: color=None
                lines.append(ax.plot(self.xs[ii], self.ys[ii], color=color, label=label,
                             linewidth=linewidth, marker=marker, markersize=10, mfc='none')[0])
        elif self.plottype == 'scatter':
            for ii in range(len(self.xs)):
                if ii < len(self.labels): label=self.labels[ii]
                else: label=None
                if ii < len(self.linewidths): linewidth=self.linewidths[ii]
                else: linewidth=None
                if ii < len(self.markers): marker=self.markers[ii]
                else: marker=None
                if ii < len(self.edgecolors): edgecolor=self.edgecolors[ii]
                else: edgecolor=None
                plt.scatter(self.xs[ii], self.ys[ii],
                            edgecolor=edgecolor, marker='.', zorder=10)

        if self.title != None: plt.title(self.title)
        plt.xlabel(self.xaxis_caption, fontsize=18)
        plt.ylabel(self.yaxis_caption, fontsize=18)

        # zed = [tick.label.set_fontsize(14)
        #        for tick in ax.xaxis.get_major_ticks()]
        # zed = [tick.label.set_fontsize(14)
        #        for tick in ax.yaxis.get_major_ticks()]

        if self.trendline == 'linear':
            # Calculate the trendline
            z = np.polyfit(self.xs[0], self.ys[0], 1)
            p = np.poly1d(z)
            plt.plot(self.xs[0], p(self.xs[0]), "r-")
            try:
                from . import stats
            except:
                import stats
            r_squared = (stats.correlation_coefficient(
                self.xs[0], self.ys[0])) ** 2
            plt.text(0, 1, r'$R^2=%.6f$' % r_squared, color='red')

        if self.plot2Ys:
            colors = ['r']
            labels = ['TD $\omega$B97-XD/def2-TZVP', 'TD $\omega$B97XD']
            ax2 = ax.twinx()
            ax2.set_ylabel('Oscillator strength $f$',
                           fontsize=18, color='black')
            ax2.set_ylim(ymin=0.0, ymax=max(self.yfs) * 1.1)
            for ii in range(len(self.xfs)):
                ax2.plot((self.xfs[ii], self.xfs[ii]),
                         (0.0, self.yfs[ii]), 'r', linewidth=1)
            zed = [tick.set_fontsize(14)
                   for tick in ax2.yaxis.get_ticklabels()]

        if self.plot3Ys:
            ax3 = ax.twinx()
            # Solution from http://stackoverflow.com/questions/9103166/multiple-axis-in-matplotlib-with-different-scales
            ax3.spines['right'].set_position(('axes', 1.13))
            ax3.set_frame_on(True)
            ax3.patch.set_visible(False)
            ax3.yaxis.set_major_formatter(
                matplotlib.ticker.OldScalarFormatter())

            ax3.set_ylabel(
                'Cross section, $\AA^2$ molecule$^{-1}$', fontsize=18, color='black')
            ax3.set_ylim(ymin=0.0, ymax=max(self.y3y) * 1.1)
            ax3.get_yaxis().set_visible(True)
            lines.append(ax3.plot(self.x3y, self.y3y,
                         'b.', label='Cross section')[0])
            # zed = [tick.set_fontsize(14)
            #        for tick in ax3.yaxis.get_ticklabels()]
            ax3.spines['right'].set_color('k')

        if self.plotstart:
            plt.xlim(left=self.plotstart)
        if self.plotend:
            plt.xlim(right=self.plotend)

        if not all(label == None for label in self.labels):
            plt.legend(lines, [ll.get_label() for ll in lines],
                       frameon=False, fontsize=18, loc='best')

        if self.savein:
            plt.savefig('%s' % self.savein, dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()

