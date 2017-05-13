from threading import Thread
import cv2


class WebcamVideoStream:
    def __init__(self, src=0, contrast=None, saturation=None, debug=False):
        # initialize the video camera stream and read the first frame
        # from the stream
        self.stream = cv2.VideoCapture(src)

        self.saved_contrast = None
        if contrast is not None:
            self.saved_contrast = self.stream.get(cv2.CAP_PROP_CONTRAST)
            self.stream.set(cv2.CAP_PROP_CONTRAST, contrast)
            if debug:
                print("setting camera contrast to", contrast)
        elif debug:
            print("camera contrast is", self.stream.get(cv2.CAP_PROP_CONTRAST))

        self.saved_saturation = None
        if saturation is not None:
            self.saved_saturation = self.stream.get(cv2.CAP_PROP_SATURATION)
            self.stream.set(cv2.CAP_PROP_SATURATION, saturation)
            if debug:
                print("setting camera saturation to", saturation)
        elif debug:
            print("camera saturation is", self.stream.get(cv2.CAP_PROP_SATURATION))
        (self.grabbed, self.frame) = self.stream.read()

        # initialize the variable used to indicate if the thread should
        # be stopped
        self.stopped = False

    def start(self):
        # start the thread to read frames from the video stream
        Thread(target=self.update, args=()).start()
        return self

    def update(self):
        # keep looping infinitely until the thread is stopped
        while True:
            # if the thread indicator variable is set, stop the thread
            if self.stopped:
                return

            # otherwise, read the next frame from the stream
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
        # return the frame most recently read
        return self.frame

    def stop(self):
        # indicate that the thread should be stopped
        self.stopped = True
        if self.saved_contrast is not None:
            self.stream.set(cv2.CAP_PROP_CONTRAST, self.saved_contrast)
        if self.saved_saturation is not None:
            self.stream.set(cv2.CAP_PROP_SATURATION, self.saved_saturation)