

class EyeFeaturesExtractor:

    def __init__(self):
        self.debug_mode = False
        self.debug_axes = None

    def create_debug_figure(self):
        return None, None

    def setup_debug_parameters(self, debug_mode, debug_axes):
        self.debug_mode = debug_mode
        self.debug_axes = debug_axes

    def clean_debug_axes(self):
        if self.debug_mode:
            for ax in self.debug_axes:
                ax.clear()

    def detect_eye_features(self, eye_image, eye_object):
        pass