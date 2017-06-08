from collections import namedtuple


Rect = namedtuple("Rect", ["x", "y", "width", "height"])
Point = namedtuple("Point", ["x", "y"])


class Eye:

    def __init__(self, area, is_right):
        self.area = Rect(*area)
        self.is_right = is_right
        self.pupil_relative = Point(0,0)
        self.inner_corner_relative = Point(0, 0)
        self.outer_corner_relative = Point(0, 0)

    @property
    def pupil(self):
        return Point(self.area.x + self.pupil_relative.x, self.area.y + self.pupil_relative.y)

    @property
    def inner_corner(self):
        return Point(self.area.x + self.inner_corner_relative.x, self.area.y + self.inner_corner_relative.y)

    @property
    def outer_corner(self):
        return Point(self.area.x + self.outer_corner_relative.x, self.area.y + self.outer_corner_relative.y)

    @property
    def absolute_area_center(self):
        return Point(self.relative_area_center.x + self.area.x, self.relative_area_center.y + self.area.y)

    @property
    def relative_area_center(self):
        return Point(self.area.width / 2, self.area.height / 2)

    def set_leftmost_corner(self, point):
        if self.is_right:
            self.outer_corner_relative = point
        else:
            self.inner_corner_relative = point

    def set_rightmost_corner(self, point):
        if self.is_right:
            self.inner_corner_relative = point
        else:
            self.outer_corner_relative = point

    def __str__(self):
        return "Eye(%s, area=%s, pupil=%s, inner_corner=%s, outer_corner=%s" % (
            "Right" if self.is_right else "Left",
            str(self.area),
            str(self.pupil),
            str(self.inner_corner),
            str(self.outer_corner),
        )