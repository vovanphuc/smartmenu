import math
import enum
HORIZONTAL_DISTANCE_THRESHOLD = 0.9
VERTICAL_DISTANCE_THRESHOLD = 0.075
DISTANCE_OF_SUB_NAME_THRESHOLD = 0.015
DISTANCE_OF_PRICE_PAIR_THRESHOLD = 0.2
DISTANCE_OF_GROUPS_THRESHOLD = 0.025
DISTANCE_OF_FAR_GROUPS_THRESHOLD = 0.036


class Direction(enum.Enum):
    no_direction = 100
    top = 0
    right = 1
    bot = 2
    left = 3

    def __eq__(self, other):
        """Overrides the default implementation"""
        try:
            res = self.value == other.value
        except:
            res = False
        finally:
            return res

    def get(value):
        if value == 0:
            return Direction.top
        elif value == 1:
            return Direction.right
        elif value == 2:
            return Direction.bot
        elif value == 3:
            return Direction.left
        else:
            return Direction.no_direction

    def get_direction_by_radians(radians):
        if radians >= math.pi*(2-1/48) or radians <= math.pi/48:
            return Direction.right
        elif radians >= math.pi*(1-1/48) and radians <= math.pi*(1+1/48):
            return Direction.left
        elif radians > math.pi*(1/2-1/6) and radians < math.pi*(1/2+1/6):
            return Direction.bot
        elif radians > math.pi*(3/2-1/6) and radians < math.pi*(3/2+1/6):
            return Direction.top
        else:
            return Direction.no_direction

    def normalize_vector(vector):
        norm = math.sqrt(vector[0] ** 2 + vector[1] ** 2)
        return [vector[0] / norm, vector[1] / norm]

    def get_angle(vector, need_negative=False):
        vector = Direction.normalize_vector(vector)
        angle = math.atan2(vector[1], vector[0])
        if need_negative == False and angle < 0:
            angle += math.pi * 2
        return angle

    def is_vertical(dir):
        return dir in [Direction.top, Direction.bot]

    def is_horizontal(dir):
        return dir in [Direction.right, Direction.left]

    def do_satify(info):
        if Direction.is_vertical(info["direction"]):
            return info["distance"] < VERTICAL_DISTANCE_THRESHOLD
        elif Direction.is_horizontal(info["direction"]):
            return info["distance"] < HORIZONTAL_DISTANCE_THRESHOLD
        return False

    def is_distance_of_sub_name(distance):
        return distance < DISTANCE_OF_SUB_NAME_THRESHOLD

    def is_distance_of_price_pair(distance):
        return distance < DISTANCE_OF_PRICE_PAIR_THRESHOLD

    def is_distance_of_groups(distance):
        return distance > DISTANCE_OF_GROUPS_THRESHOLD

    def is_distance_of_far_groups(distance):
        return distance > DISTANCE_OF_FAR_GROUPS_THRESHOLD

Direction.Directions = [Direction.top, Direction.right, Direction.bot, Direction.left]