import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button

from utils.gazetracker.calibrator import Calibrator

plt.rcParams['toolbar'] = 'None'

plt.switch_backend('TKAgg')
mng = plt.get_current_fig_manager()
screen_x =mng.window.winfo_screenwidth()
print(screen_x)
screen_y =mng.window.winfo_screenheight()
print(screen_y)

screen_y = 768

calibrator = Calibrator()

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
        calibrator.capture_point(interval, centers[index[i-1]])
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

