from matplotlib import patches as patches


def draw_routine(ax, picture, right_eye, left_eye, not_eyes):
    """
    Draw the main window
    :param ax: matplotlib ax to draw on 
    :param picture: 
    :param right_eye: 
    :param left_eye: 
    :param not_eyes: list of Rect
    :return: 
    """
    ax.imshow(picture, cmap="gray")
    if right_eye is not None and left_eye is not None:
        # draw eye regions (found using the haar cascade)
        ax.add_patch(patches.Rectangle(
            (right_eye.area.x, right_eye.area.y),
            right_eye.area.width,
            right_eye.area.height,
            linewidth=1, edgecolor='r', facecolor='none'
        ))
        ax.add_patch(patches.Rectangle(
            (left_eye.area.x, left_eye.area.y),
            left_eye.area.width,
            left_eye.area.height,
            linewidth=1, edgecolor='r', facecolor='none'
        ))

        for rect in not_eyes:
            ax.add_patch(patches.Rectangle(
                (rect.x, rect.y),
                rect.width,
                rect.height,
                linewidth=1, edgecolor='b', facecolor='none'
            ))

        # draw pupil spots (red)
        ax.plot(right_eye.pupil[0], right_eye.pupil[1], "r+")
        ax.plot(left_eye.pupil[0], left_eye.pupil[1], "r+")
        # draw inner corners (green)
        ax.plot(right_eye.inner_corner[0], right_eye.inner_corner[1], "g+")
        ax.plot(left_eye.inner_corner[0], left_eye.inner_corner[1], "g+")
        # draw outer corners (blue)
        ax.plot(right_eye.outer_corner[0], right_eye.outer_corner[1], "b+")
        ax.plot(left_eye.outer_corner[0], left_eye.outer_corner[1], "b+")