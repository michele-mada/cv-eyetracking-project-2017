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
    parser.add_argument('-t', '--test', help='specify distance in cm from screen to run test accuracy', type=int, default=0)
    parser.add_argument('-d', '--diagonal-inch', help='specify the size in inches of your screen diagonal ',
                        type=float, default="15.5")
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
           (screen_x - radius - padding, radius + padding)
           ]

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
        #convert distance from cm to pixels
        distance_from_screen = cli.test
        distance_inch = distance_from_screen/2.54
        diagonal_pixel = np.sqrt(screen_x**2 + screen_y**2)
        diagonal_inch = cli.diagonal_inch
        ppi = diagonal_pixel/diagonal_inch
        distance_pixel = distance_inch*ppi
        print(distance_from_screen)
        for i in range(num_targets):
            gaze_target.set_color('blue')
            gaze_target.center = centers[index[i]]
            fig.canvas.draw()
            calibrator.capture_point(interval * 1.2, 1, centers[index[i]])
            plt.pause(interval + 1)
        fuzzy_points = calibrator.evaluate_calibration(distance_pixel,mapper_implementations["fuzzy"])
        quad_points = calibrator.evaluate_calibration(distance_pixel,mapper_implementations["poly_quad"])
        lin_points = calibrator.evaluate_calibration(distance_pixel,mapper_implementations["poly_lin"])
        neural_points = calibrator.evaluate_calibration(distance_pixel,mapper_implementations["neural"])
    gaze_target.set_visible(False)

    for i in range(num_targets):
        estimatedf = plt.Circle(fuzzy_points[i], 10, color='r')
        ax.add_artist(estimatedf)
        estimatedq = plt.Circle(quad_points[i], 10, color='blue')
        ax.add_artist(estimatedq)
        estimatedl = plt.Circle(lin_points[i], 10, color='g')
        ax.add_artist(estimatedl)
        estimatedn = plt.Circle(neural_points[i], 10, color='k')
        ax.add_artist(estimatedn)
            

    

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

