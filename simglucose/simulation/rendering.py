import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import logging
from datetime import timedelta

logger = logging.getLogger(__name__)


class Viewer(object):
    def __init__(self, start_time, patient_name, figsize=None):
        self.start_time = start_time
        self.patient_name = patient_name
        self.fig, self.axes, self.lines = self.initialize()
        self.update()
        
    def initialize(self):
        plt.ion()
        fig, axes = plt.subplots(4)

        axes[0].set_ylabel('BG (mg/dL)')
        axes[1].set_ylabel('CHO (g/min)')
        axes[2].set_ylabel('Insulin (U/min)')
        axes[3].set_ylabel('Risk Index')

        lineBG, = axes[0].plot([], [], label='BG')
        lineCGM, = axes[0].plot([], [], label='CGM')
        lineCHO, = axes[1].plot([], [], label='CHO')
        lineIns, = axes[2].plot([], [], label='Insulin')
        lineLBGI, = axes[3].plot([], [], label='Hypo Risk')
        lineHBGI, = axes[3].plot([], [], label='Hyper Risk')
        lineRI, = axes[3].plot([], [], label='Risk Index')

        lines = [lineBG, lineCGM, lineCHO, lineIns, lineLBGI, lineHBGI, lineRI]

        axes[0].set_ylim([70, 180])
        axes[1].set_ylim([-5, 30])
        axes[2].set_ylim([-0.05, 0.05])
        axes[3].set_ylim([0, 5])

        for ax in axes:
            ax.set_xlim(
                [self.start_time, self.start_time + timedelta(hours=3)])
            ax.legend()

        # Plot zone patches
        axes[0].axhspan(70, 180, alpha=0.3, color='limegreen', lw=0)
        axes[0].axhspan(50, 70, alpha=0.3, color='red', lw=0)
        axes[0].axhspan(0, 50, alpha=0.3, color='darkred', lw=0)
        axes[0].axhspan(180, 250, alpha=0.3, color='red', lw=0)
        axes[0].axhspan(250, 1000, alpha=0.3, color='darkred', lw=0)

        axes[0].tick_params(labelbottom=False)
        axes[1].tick_params(labelbottom=False)
        axes[2].tick_params(labelbottom=False)
        axes[3].xaxis.set_minor_locator(mdates.AutoDateLocator())
        axes[3].xaxis.set_minor_formatter(mdates.DateFormatter('%H:%M\n'))
        axes[3].xaxis.set_major_locator(mdates.DayLocator())
        axes[3].xaxis.set_major_formatter(mdates.DateFormatter('\n%b %d'))

        axes[0].set_title(self.patient_name)

        return fig, axes, lines

    def update(self):
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        
        try: __IPYTHON__
        except NameError:
            plt.pause(0.001)
        else:
            if __IPYTHON__:
                display(self.fig)
            else:
                plt.pause(0.001)


    def render(self, data):
        self.lines[0].set_xdata(data.index.values)
        self.lines[0].set_ydata(data['BG'].values)

        self.lines[1].set_xdata(data.index.values)
        self.lines[1].set_ydata(data['CGM'].values)

        self.axes[0].draw_artist(self.axes[0].patch)
        self.axes[0].draw_artist(self.lines[0])
        self.axes[0].draw_artist(self.lines[1])

        adjust_ylim(self.axes[0], min(min(data['BG']), min(data['CGM'])),
                    max(max(data['BG']), max(data['CGM'])))
        adjust_xlim(self.axes[0], data.index[-1])

        self.lines[2].set_xdata(data.index.values)
        self.lines[2].set_ydata(data['CHO'].values)

        self.axes[1].draw_artist(self.axes[1].patch)
        self.axes[1].draw_artist(self.lines[2])

        adjust_ylim(self.axes[1], min(data['CHO']), max(data['CHO']))
        adjust_xlim(self.axes[1], data.index[-1])

        self.lines[3].set_xdata(data.index.values)
        self.lines[3].set_ydata(data['insulin'].values)

        self.axes[2].draw_artist(self.axes[2].patch)
        self.axes[2].draw_artist(self.lines[3])
        adjust_ylim(self.axes[2], min(data['insulin']), max(data['insulin']))
        adjust_xlim(self.axes[2], data.index[-1])

        self.lines[4].set_xdata(data.index.values)
        self.lines[4].set_ydata(data['LBGI'].values)

        self.lines[5].set_xdata(data.index.values)
        self.lines[5].set_ydata(data['HBGI'].values)

        self.lines[6].set_xdata(data.index.values)
        self.lines[6].set_ydata(data['Risk'].values)

        self.axes[3].draw_artist(self.axes[3].patch)
        self.axes[3].draw_artist(self.lines[4])
        self.axes[3].draw_artist(self.lines[5])
        self.axes[3].draw_artist(self.lines[6])
        adjust_ylim(self.axes[3], min(data['Risk']), max(data['Risk']))
        adjust_xlim(self.axes[3], data.index[-1], xlabel=True)

        self.update()

    def close(self):
        plt.close(self.fig)

##############################################################################
# Viewer for plotting internal states
class StatesViewer(object):
    def __init__(self, start_time, patient_name, figsize=None):
        self.start_time = start_time
        self.patient_name = patient_name
        self.fig, self.axes, self.lines = self.initialize()
        self.update()
        
    def initialize(self):
        plt.ion()
        fig1, axes1 = plt.subplots(2)
        fig2, axes2 = plt.subplots(2)
        axes2 = np.append(axes2, axes2[0].twinx())

        axes1[0].set_ylabel('Glucose (Stomach)')
        axes1[1].set_ylabel('Glucose (Subsystem)')
        axes2[0].set_ylabel('Insulin (Subcut)')
        axes2[2].set_ylabel('Insulin (Subsystem)')
        axes2[1].set_ylabel('Insulin Concentration')

        lineQsto1, = axes1[0].plot([], [], label='$Q_{sto1}$')
        lineQsto2, = axes1[0].plot([], [], label='$Q_{sto2}$')
        lineQgut,  = axes1[0].plot([], [], label='$Q_{gut}$')

        lineGp, = axes1[1].plot([], [], label='$G_p$')
        lineGt, = axes1[1].plot([], [], label='$G_t$')
        lineGs, = axes1[1].plot([], [], label='$G_s$')
        
        lineIsc1, = axes2[0].plot([], [], label='$I_{sc1}$')
        lineIsc2, = axes2[0].plot([], [], label='$I_{sc2}$')
        lineIp,   = axes2[2].plot([], [], label='$I_p$', color='C2')
        lineIl,   = axes2[2].plot([], [], label='$I_l$', color='C3')

        lineX,  = axes2[1].plot([], [], label='$X$')
        lineXL, = axes2[1].plot([], [], label='$X^L$')
        lineId, = axes2[1].plot([], [], label='$I\'$')

        lines = [lineQsto1, lineQsto2, lineQgut, 
                 lineGp, lineGt, lineGs, 
                 lineIsc1, lineIsc2, lineIp, lineIl, 
                 lineX, lineXL, lineId]

        handler,  label  = axes2[0].get_legend_handles_labels()
        handler1, label1 = axes2[2].get_legend_handles_labels()
        axes2[0].legend(handler+handler1, label+label1)
        axes2[1].legend()

        for ax in axes1:
            ax.set_xlim(
                [self.start_time, self.start_time + timedelta(hours=3)])
            ax.legend()
            
        for ax in axes2:
            ax.set_xlim(
                [self.start_time, self.start_time + timedelta(hours=3)])
        
        # Plot zone patches
        axes1[0].tick_params(labelbottom=False)
        axes1[1].xaxis.set_minor_locator(mdates.AutoDateLocator())
        axes1[1].xaxis.set_minor_formatter(mdates.DateFormatter('%H:%M\n'))
        axes1[1].xaxis.set_major_locator(mdates.DayLocator())
        axes1[1].xaxis.set_major_formatter(mdates.DateFormatter('\n%b %d'))
        axes2[0].tick_params(labelbottom=False)
        axes2[1].xaxis.set_minor_locator(mdates.AutoDateLocator())
        axes2[1].xaxis.set_minor_formatter(mdates.DateFormatter('%H:%M\n'))
        axes2[1].xaxis.set_major_locator(mdates.DayLocator())
        axes2[1].xaxis.set_major_formatter(mdates.DateFormatter('\n%b %d'))

        axes1[0].set_title(self.patient_name+' Glucose States')
        axes2[0].set_title(self.patient_name+' Insulin States')

        return [fig1, fig2], [axes1, axes2], lines

    def update(self):
        self.fig[0].canvas.draw()
        self.fig[0].canvas.flush_events()
        self.fig[1].canvas.draw()
        self.fig[1].canvas.flush_events()
        
        try: __IPYTHON__
        except NameError:
            plt.pause(0.001)
        else:
            if __IPYTHON__:
                display(self.fig[0])
                display(self.fig[1])
            else:
                plt.pause(0.001)

    def render(self, data):
        self.lines[0].set_xdata(data.index.values)
        self.lines[0].set_ydata(data['Qsto1'].values)

        self.lines[1].set_xdata(data.index.values)
        self.lines[1].set_ydata(data['Qsto2'].values)

        self.lines[2].set_xdata(data.index.values)
        self.lines[2].set_ydata(data['Qgut'].values)

        self.axes[0][0].draw_artist(self.axes[0][0].patch)
        self.axes[0][0].draw_artist(self.lines[0])
        self.axes[0][0].draw_artist(self.lines[1])
        self.axes[0][0].draw_artist(self.lines[2])

        adjust_ylim(self.axes[0][0], 
                    min( min(data['Qsto1']), min(data['Qsto2']), min(data['Qgut']) ),
                    max( max(data['Qsto1']), max(data['Qsto2']), max(data['Qgut']) ) )
        adjust_xlim(self.axes[0][0], data.index[-1])


        self.lines[3].set_xdata(data.index.values)
        self.lines[3].set_ydata(data['Gp'].values)

        self.lines[4].set_xdata(data.index.values)
        self.lines[4].set_ydata(data['Gt'].values)

        self.lines[5].set_xdata(data.index.values)
        self.lines[5].set_ydata(data['Gs'].values)
        
        self.axes[0][1].draw_artist(self.axes[0][1].patch)
        self.axes[0][1].draw_artist(self.lines[3])
        self.axes[0][1].draw_artist(self.lines[4])
        self.axes[0][1].draw_artist(self.lines[5])
        
        adjust_ylim(self.axes[0][1], 
                    min( min(data['Gp']), min(data['Gt']), min(data['Gs']) ),
                    max( max(data['Gp']), max(data['Gt']), max(data['Gs']) ) )
        adjust_xlim(self.axes[0][1], data.index[-1])

        self.lines[6].set_xdata(data.index.values)
        self.lines[6].set_ydata(data['Isc1'].values)

        self.lines[7].set_xdata(data.index.values)
        self.lines[7].set_ydata(data['Isc2'].values)

        self.lines[8].set_xdata(data.index.values)
        self.lines[8].set_ydata(data['Ip'].values)

        self.lines[9].set_xdata(data.index.values)
        self.lines[9].set_ydata(data['Il'].values)
        
        self.axes[1][0].draw_artist(self.axes[1][0].patch)
        self.axes[1][0].draw_artist(self.lines[6])
        self.axes[1][0].draw_artist(self.lines[7])
        self.axes[1][0].draw_artist(self.lines[8])
        self.axes[1][0].draw_artist(self.lines[9])
        
        adjust_ylim(self.axes[1][0], 
                    min(min(data['Isc1']),min(data['Isc2'])), 
                    max(max(data['Isc1']),max(data['Isc2'])))
        adjust_xlim(self.axes[1][0], data.index[-1])
        adjust_ylim(self.axes[1][2], 
                    min(min(data['Ip']),min(data['Il'])), 
                    max(max(data['Ip']),max(data['Il'])))
        adjust_xlim(self.axes[1][2], data.index[-1])

        self.lines[10].set_xdata(data.index.values)
        self.lines[10].set_ydata(data['X'].values)

        self.lines[11].set_xdata(data.index.values)
        self.lines[11].set_ydata(data['XL'].values)

        self.lines[12].set_xdata(data.index.values)
        self.lines[12].set_ydata(data['Id'].values)

        self.axes[1][1].draw_artist(self.axes[1][1].patch)
        self.axes[1][1].draw_artist(self.lines[10])
        self.axes[1][1].draw_artist(self.lines[11])
        self.axes[1][1].draw_artist(self.lines[12])
        adjust_ylim(self.axes[1][1], 
                    min(min(data['X']),min(data['XL']),min(data['Id'])), 
                    max(max(data['X']),max(data['XL']),max(data['Id'])))
        adjust_xlim(self.axes[1][1], data.index[-1], xlabel=True)

        self.update()

    def close(self):
        plt.close(self.fig[0])
        plt.close(self.fig[1])


def adjust_ylim(ax, ymin, ymax):
    ylim = ax.get_ylim()
    update = False

    if ymin < ylim[0]:
        y1 = ymin - 0.1 * abs(ymin)
        update = True
    else:
        y1 = ylim[0]

    if ymax > ylim[1]:
        y2 = ymax + 0.1 * abs(ymax)
        update = True
    else:
        y2 = ylim[1]

    if update:
        ax.set_ylim([y1, y2])
        for spine in ax.spines.values():
            ax.draw_artist(spine)
        ax.draw_artist(ax.yaxis)


def adjust_xlim(ax, timemax, xlabel=False):
    xlim = mdates.num2date(ax.get_xlim())
    update = False

    # remove timezone awareness to make them comparable
    timemax = timemax.replace(tzinfo=None)
    xlim[0] = xlim[0].replace(tzinfo=None)
    xlim[1] = xlim[1].replace(tzinfo=None)

    if timemax > xlim[1] - timedelta(minutes=30):
        xmax = xlim[1] + timedelta(hours=20) # 6
        update = True

    if update:
        ax.set_xlim([xlim[0], xmax])
        for spine in ax.spines.values():
            ax.draw_artist(spine)
        ax.draw_artist(ax.xaxis)
        if xlabel:
            ax.xaxis.set_minor_locator(mdates.AutoDateLocator())
            ax.xaxis.set_minor_formatter(mdates.DateFormatter('%H:%M\n'))
            ax.xaxis.set_major_locator(mdates.DayLocator())
            ax.xaxis.set_major_formatter(mdates.DateFormatter('\n%b %d'))
