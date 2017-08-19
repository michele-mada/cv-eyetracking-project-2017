from classes import Point


class MapperInterface:


    def before_training(self, observations):
        pass

    def train_from_data(self, observations, is_left=False):
        pass

    def map_point(self, eyevector):
        return Point(0,0)
