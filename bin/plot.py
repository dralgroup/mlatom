#!/usr/bin/python
'''
  !----------------------------------------------------------------------------!
  !             Script to make common types of plots                           !
  !       Pavlo O. Dral, Max-Planck-Institut fuer Kohlenforschung, 2016        !
  !----------------------------------------------------------------------------!
               
  Usage:
    plot.py [arguments]
  
  Arguments:
    -help             - print this help and exit
    
    Plot types:
    -type=<type>      - type of plots. Allowed types:
                          linechart                    [default]
                          linechartwopoints            [optional]
                          scatter                      [optional]

    
    -xy=<name>        - file names (separated by comma) with XY data points  [default]
    -x=<name>         - file name with X data points  [optional]
    -y=<name>         - file name with Y data points  [optional]
    -normalize        - normalizes plots
    
    -2ys              - plot with two Y axes
    -x2y=<name>       - file names (separated by comma) with XY data points for plot with two Y axes
    -fs=<name>        - file name with the oscillator strengths
    -shift            - shifts the spectrum so that the maxima coincide
    -shiftby=<value>  - shifts the spectrum to the right by specified amount
    
    -3ys              - plot with two Y axes
    -x3y=<name>       - file name with XY data points for plot with two Y axes
    
    -plotstart=<value> - start plotting with <value>                  [optional]
    -plotend=<value>   - stop plotting with <value>                   [optional]
    
    -trendline=<type> - type of trendline
    
    -savein=<name>    - save plot in file             [required]
  
  All arguments are case-insensitive except for file names
  
  Example:
    plot.py -type=scatter -trendline=linear -xy=xy.dat -savein=scatter.png
'''

import sys
import os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
try:
    from . import stats
except:
    import stats


class args(object):
    # Default values:
    plottype = 'linechart'
    trendline = None

    fxys = []
    xy = None
    x = None
    y = None
    normalize = False

    plot2Ys = False
    fx2ys = []
    x2y = None
    fs = None
    shift = False
    shiftby = None

    plot3Ys = False
    x3y = None

    plotstart = None
    plotend = None

    savein = None

    # Allowed values:
    allowedArgs = {'plottype': ['linechart', 'linechartwopoints', 'scatter'],
                   'trendline': [None, 'linear']}

    @classmethod
    def parse(cls, argsraw):
        for arg in argsraw:
            if arg[0:1] == '-':
                if arg.lower() == '-help':
                    printHelp(error=False)
                    sys.exit()
                elif arg.lower()[0:len('-type=')] == '-type=':
                    cls.plottype = arg[len('-type='):].lower()
                elif arg.lower()[0:len('-trendline=')] == '-trendline=':
                    cls.trendline = arg[len('-trendline='):].lower()
                elif arg.lower()[0:len('-xy=')] == '-xy=':
                    cls.xy = arg[len('-xy='):]
                elif arg.lower()[0:len('-x=')] == '-x=':
                    cls.x = arg[len('-x='):]
                elif arg.lower()[0:len('-y=')] == '-y=':
                    cls.y = arg[len('-y='):]
                elif arg.lower() == '-2ys':
                    cls.plot2Ys = True
                elif arg.lower()[0:len('-x2y=')] == '-x2y=':
                    cls.x2y = arg[len('-x2y='):]
                elif arg.lower()[0:len('-fs=')] == '-fs=':
                    cls.fs = arg[len('-fs='):]
                elif arg.lower() == '-normalize':
                    cls.normalize = True
                elif arg.lower() == '-shift':
                    cls.shift = True
                elif arg.lower() == '-3ys':
                    cls.plot3Ys = True
                elif arg.lower()[0:len('-x3y=')] == '-x3y=':
                    cls.x3y = arg[len('-x3y='):]
                elif arg.lower()[0:11] == '-plotstart=':
                    cls.plotstart = float(arg[11:])
                elif arg.lower()[0:9] == '-plotend=':
                    cls.plotend = float(arg[9:])
                elif arg.lower()[0:len('-shiftby=')] == '-shiftby=':
                    cls.shiftby = float(arg[len('-shiftby='):])
                elif arg.lower()[0:len('-savein=')] == '-savein=':
                    cls.savein = arg[len('-savein='):]
                else:
                    print('Argument "%s" is not recognized' %
                          arg, file=sys.stderr)
                    printHelp(error=True)
                    sys.exit()
            else:
                print('Argument "%s" is not recognized' % arg, file=sys.stderr)
                printHelp(error=True)
                sys.exit()
        cls.checkArgs()

    @classmethod
    def checkArgs(cls):
        if not cls.trendline in cls.allowedArgs['trendline']:
            print('Trendline type "%s" is not supported' %
                  cls.trendline, file=sys.stderr)
            printHelp(error=True)
            sys.exit()

        if cls.xy:
            if cls.x or cls.y:
                print('Options -xy, -x, and -y cannot be used together',
                      file=sys.stderr)
                printHelp(error=True)
                sys.exit()
            for ff in cls.xy.split(','):
                cls.fxys.append(ff)
                if not os.path.exists(ff):
                    print('File %s does not exist' % ff, file=sys.stderr)
                    printHelp(error=True)
                    sys.exit()

        if cls.x or cls.y:
            if not (cls.x and cls.y):
                print('Options -x and -y should be provided together',
                      file=sys.stderr)
                printHelp(error=True)
                sys.exit()
            elif cls.xy:
                print('Options -xy, -x, and -y cannot be used together',
                      file=sys.stderr)
                printHelp(error=True)
                sys.exit()
            if not os.path.exists(cls.x):
                print('File %s does not exist' % cls.x, file=sys.stderr)
                printHelp(error=True)
                sys.exit()
            if not os.path.exists(cls.y):
                print('File %s does not exist' % cls.y, file=sys.stderr)
                printHelp(error=True)
                sys.exit()

        if not (cls.xy or cls.x or cls.y):
            print(
                'No data file provided. Please use options -xy or -x and -y', file=sys.stderr)
            printHelp(error=True)
            sys.exit()

        if cls.x2y:
            for ff in cls.x2y.split(','):
                cls.fx2ys.append(ff)
                if not os.path.exists(ff):
                    print('File %s does not exist' % cls.x2y, file=sys.stderr)
                    printHelp(error=True)
                    sys.exit()

        if cls.fs:
            if not os.path.exists(cls.fs):
                print('File %s does not exist' % cls.fs, file=sys.stderr)
                printHelp(error=True)
                sys.exit()

        if cls.x3y:
            if not os.path.exists(cls.x3y):
                print('File %s does not exist' % cls.x3y, file=sys.stderr)
                printHelp(error=True)
                sys.exit()

        if not (cls.savein):
            print(
                'Please specify where plots should be saved using option -savein', file=sys.stderr)
            printHelp(error=True)
            sys.exit()


def printHelp(error=False):
    if error:
        print(__doc__, file=sys.stderr)
    else:
        print(__doc__)


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
        self.markers = []
        self.linewidths = []

    def read_data(self):
        # Get data
        self.xs = []
        self.ys = []
        if args.xy:
            for fxy in args.fxys:
                xx = []
                yy = []
                with open(fxy, 'r') as ff:
                    for line in ff:
                        xx.append(float(line.split()[0]))
                        yy.append(float(line.split()[1]))
                self.xs.append(xx)
                self.ys.append(yy)
        else:
            with open(args.x, 'r') as ff:
                for line in ff:
                    self.xx.append(float(line))
            with open(args.y, 'r') as ff:
                for line in ff:
                    self.yy.append(float(line))
        if args.plot2Ys:
            for fx2y in args.fx2ys:
                x2y = []
                y2y = []
                with open(fx2y, 'r') as ff:
                    for line in ff:
                        x2y.append(float(line.split()[0]))
                        y2y.append(float(line.split()[1]))
                self.x2ys.append(x2y)
                self.y2ys.append(y2y)
            if args.fs:
                with open(args.fs, 'r') as ff:
                    for line in ff:
                        self.xfs.append(float(line.split()[0]))
                        self.yfs.append(float(line.split()[1]))

        if args.plot3Ys:
            with open(args.x3y, 'r') as ff:
                for line in ff:
                    self.x3y.append(float(line.split()[0]))
                    self.y3y.append(float(line.split()[1]))

        ev2nm = 1240.0

        #self.xx = [ev2nm / ii for ii in self.xx]
        # for jj in range(len(self.xs)):
        #self.xs[jj] = [ev2nm / ii for ii in self.xs[jj]]
        #self.ys[jj] = [ii / 10000 for ii in self.ys[jj]]
        '''
        for jj in range(len(self.x2ys)):
            self.x2ys[jj] = [ev2nm / ii for ii in self.x2ys[jj]]
            if args.fs: 
                self.xfs = [ev2nm / xtmp for xtmp in self.xfs]
        '''
        #self.x2y = [ev2nm / ii for ii in self.x2y]
        #self.x3y = [ev2nm / ii for ii in self.x3y]

        if args.normalize:
            for ii in range(len(self.ys)):
                ymax = max(self.ys[ii])
                self.ys[ii] = [zz / ymax for zz in self.ys[ii]]

        if args.shift:
            # Superimposes global maxima
            ix1m = self.ys[0].index(max(self.ys[0]))
            #ix2m = self.y2ys[0].index(max(self.y2ys[0]))
            ix2m = self.ys[1].index(max(self.ys[1]))

            #self.delta = self.xs[0][ix1m] - self.x2ys[0][ix2m]
            self.delta = self.xs[0][ix1m] - self.xs[1][ix2m]
            if args.shiftby:
                self.delta = args.shiftby
            print('Theoretical spectrum is shifted by %.2f nm' % self.delta)
            #self.x2ys[0] = [xtmp + self.delta for xtmp in self.x2ys[0]]
            self.xs[1] = [xtmp + self.delta for xtmp in self.xs[1]]
            if args.fs:
                self.xfs = [xtmp + self.delta for xtmp in self.xfs]

        # Plot data
        # self.make_figure()

    def make_figure(self):
        matplotlib.rcParams['axes.linewidth'] = 2
        # plt.figure(figsize=(3.33,3.33))
        # plt.gcf().subplots_adjust(left=0.15,bottom=0.15)

        #ax = plt.gca()
        fig, ax = plt.subplots()
        fig.subplots_adjust(right=0.85)
        lines = []

        if self.plottype == 'linechart':
            plt.plot(self.xs[0], self.ys[0], 'ro')
        if self.plottype == 'linechart_without_points':
            if self.colors == []:
                self.colors = [None for xx in self.xs]
            if self.labels == []:
                self.labels = [None for xx in self.xs]
            if self.linewidths == []:
                self.linewidths = [None for xx in self.xs]
            if self.markers == []:
                self.markers = [None for xx in self.xs]
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
            plt.scatter(self.xs[0], self.ys[0],
                        edgecolor='k', marker='.', zorder=10)

        if self.title != None: plt.title(self.title)
        plt.xlabel(self.xaxis_caption, fontsize=18)
        plt.ylabel(self.yaxis_caption, fontsize=18)

        zed = [tick.label.set_fontsize(14)
               for tick in ax.xaxis.get_major_ticks()]
        zed = [tick.label.set_fontsize(14)
               for tick in ax.yaxis.get_major_ticks()]

        if self.trendline == 'linear':
            # Calculate the trendline
            z = np.polyfit(self.xs[0], self.ys[0], 1)
            p = np.poly1d(z)
            plt.plot(self.xs[0] + [50000.0], p(self.xs[0] + [50000.0]), "r-")

            r_squared = (stats.correlation_coefficient(
                self.xs[0], self.ys[0])) ** 2
            plt.text(15000, 5000, r'$R^2=%.6f$' % r_squared, color='red')

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
            zed = [tick.set_fontsize(14)
                   for tick in ax3.yaxis.get_ticklabels()]
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

if __name__ == '__main__':
    args.parse(sys.argv[1:])

    thisplot = plot()
    thisplot.read_data()
    same_args = ['plottype', 'trendline', 'normalize',
                 'plot2Ys', 'plot3Ys', 'plotstart', 'plotend', 'savein']
    for arg in same_args:
        thisplot.__dict__[arg] = args.__dict__[arg]
    thisplot.make_figure()
