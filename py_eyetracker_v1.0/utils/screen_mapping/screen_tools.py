import matplotlib.pyplot as plt


def get_screen_size(override_screensize=None):
    if override_screensize:
        screen_size = override_screensize
    else:
        mng = plt.get_current_fig_manager()
        screen_size = (
            mng.window.winfo_screenwidth(),
            mng.window.winfo_screenheight(),
        )
        plt.close()
    return screen_size
