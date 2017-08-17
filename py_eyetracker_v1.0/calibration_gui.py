import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button
import argparse

from utils.gazetracker.calibrator import Calibrator
import utils.screen_mapping.map_function

plt.rcParams['toolbar'] = 'None'


def parsecli():
    parser = argparse.ArgumentParser(description="Calibration utility")
    parser.add_argument('-m', '--mapping-function', help='eye-vector to screen mapping function to use',
                        type=str, default="quadratic", choices=utils.screen_mapping.map_function.profiles.keys())
    parser.add_argument('-o', '--override-screensize', metavar='1366x768', help='specify the size of your screen',
                        type=str, default="None")
    return parser.parse_args()


cli = parsecli()
plt.switch_backend('TKAgg')
mng = plt.get_current_fig_manager()

if cli.override_screensize != "None":
    (screen_x, screen_y) = tuple(map(int,cli.override_screensize.split("x")))
else:
    screen_x =mng.window.winfo_screenwidth()
    screen_y =mng.window.winfo_screenheight()

print("Screen size: %dx%d" % (screen_x, screen_y))

calibrator = Calibrator(mapping=cli.mapping_function)

#full-window mode
#mng.window.state('zoomed')

#full-screen mode
mng.full_screen_toggle()

num_targets = 9
radius = 30
padding = 20
interval = 3
centers = [(radius + padding, screen_y - radius - padding),
           (screen_x/2, screen_y - radius - padding),
           (screen_x - radius - padding, screen_y - radius - padding),
           (radius + padding, screen_y/2),
           (screen_x/2, screen_y/2),
           (screen_x - radius - padding, screen_y/2),
           (radius + padding, radius + padding),
           (screen_x/2, radius + padding),
           (screen_x - radius - padding, radius + padding)]

index = np.random.permutation(num_targets)

gaze_target = plt.Circle(centers[index[0]], 20, color='r')


def start_calibration(event):
    axbutton.set_visible(False)
    ax.add_artist(gaze_target)
    for i in range(1,num_targets):
        print(i)
        calibrator.capture_point(interval * 1.5, 1, centers[index[i-1]])
        plt.pause(interval)
        gaze_target.center = centers[index[i]]
        fig.canvas.draw()
    plt.pause(interval)
    gaze_target.set_visible(False)
    calibrator.compute_mapping_parameters()
    calibrator.save_mapping_parameters()

fig = plt.gcf()
ax = fig.gca()

ax.set_position([0, 0, 1, 1])
ax.set_xlim(0,screen_x)
ax.set_ylim(0,screen_y)


axbutton = plt.axes([0.3, 0.2, 0.4, 0.2])
start = Button(axbutton, 'Start calibration')
start.on_clicked(start_calibration)

def quit_figure(event):
    if event.key == 'q':
        plt.close(event.canvas.figure)

cid = plt.gcf().canvas.mpl_connect('key_press_event', quit_figure)

plt.show()

