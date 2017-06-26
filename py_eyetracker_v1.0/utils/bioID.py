from math import sqrt
import cv2
import os
import csv

from classes import Point


class Face:

    def __init__(self, filepath, right_eye, left_eye):
        assert(isinstance(right_eye, Point))
        assert (isinstance(left_eye, Point))
        self.filepath = filepath
        self.right_eye = right_eye
        self.left_eye = left_eye

    @property
    def eye_center_distance(self):
        return sqrt((self.right_eye.x - self.left_eye.x) ** 2 + (self.right_eye.y - self.left_eye.y) ** 2)

    def load_cv2(self):
        image_cv2 = cv2.imread(self.filepath)
        return cv2.cvtColor(image_cv2, cv2.COLOR_RGB2GRAY)


class BioIDFaceDatabase:
    # TODO: add also face landmarks

    img_format = ".pgm"
    eye_data_format = ".eye"

    def __init__(self, folder):
        self.faces = []
        self.load_from_dir(folder)

    def load_from_dir(self, folder):
        for filename in os.listdir(folder):
            if filename.endswith(self.img_format):
                basename = filename[:-len(self.img_format)]
                self.load_data_item(os.path.join(folder,basename))

    def load_data_item(self, basename):
        right_eye, left_eye = self.load_eye_data(basename + self.eye_data_format)
        newface = Face(basename + self.img_format, right_eye, left_eye)
        self.faces.append(newface)

    def load_eye_data(self, eyefilename):
        with open(eyefilename, "r") as fp:
            eyereader = csv.reader(fp, delimiter='\t')
            rows = list(eyereader)
            (lx, ly, rx, ry) = rows[1]
            return Point(int(rx), int(ry)), Point(int(lx), int(ly))
