import argparse

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button

from utils.screen_mapping.calibrator import CaptureCalibrator
from utils.screen_mapping import mapper_implementations


def parsecli():
    parser = argparse.ArgumentParser(description="Calibration utility")
    parser.add_argument('-o', '--override-screensize', metavar='1366x768', help='specify the size of your screen',
                        type=str, default="None")
    parser.add_argument('-f', '--face-window', help='show face detection window', action='store_true')
    parser.add_argument('-t', '--test', help='run a second time to test accuracy', type=int, default=0) 
    parser.add_argument('-m', '--mapping-function', help='eye-vector to screen mapping function to use',
                        type=str, default="poly_quad", choices=mapper_implementations.keys())
    return parser.parse_args()


cli = parsecli()
calibrator = CaptureCalibrator(show_gui=cli.face_window)

plt.rcParams['toolbar'] = 'None'
plt.switch_backend('TKAgg')
mng = plt.get_current_fig_manager()

if cli.override_screensize != "None":
    (screen_x, screen_y) = tuple(map(int,cli.override_screensize.split("x")))
else:
    screen_x =mng.window.winfo_screenwidth()
    screen_y =mng.window.winfo_screenheight()

print("Screen size: %dx%d" % (screen_x, screen_y))


#full-window mode
#mng.window.state('zoomed')

#full-screen mode
mng.full_screen_toggle()

num_targets = 9
radius = 30
padding = 20
interval = 5
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
    for i in range(num_targets):
        print(i)
        gaze_target.center = centers[index[i]]
        fig.canvas.draw()
        calibrator.capture_point(interval * 1.2, 1, centers[index[i]])
        plt.pause(interval + 1)
    calibrator.save_mapping_parameters()
    if(cli.test):
        distance_from_screen = cli.test
        print(distance_from_screen)
        for i in range(num_targets):
            gaze_target.set_color('blue')
            gaze_target.center = centers[index[i]]
            fig.canvas.draw()
            calibrator.capture_point(interval * 1.2, 1, centers[index[i]])
            plt.pause(interval + 1)
        calibrator.evaluate_calibration(distance_from_screen,mapper_implementations[cli.mapping_function])
    gaze_target.set_visible(False)
            

    

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

