from direction import Direction
import menu_class
from menu_class import Class
import math
MAX_RANGE = 1

class Vertice:
    def __init__(self, cls, bbox, text):
        self.cls = cls
        self.text = __class__.normalize(text)
        self.bbox = []
        for idx in range(4):
            self.bbox.append(((bbox[idx][0] + bbox[(idx + 1) % 4][0]) / 2, \
                            (bbox[idx][1] + bbox[(idx + 1) % 4][1]) / 2))
        self.bbox_angle = Direction.get_angle([self.bbox[1][0] - self.bbox[3][0], \
                                               self.bbox[1][1] - self.bbox[3][1]])

    def normalize(text):
        text = text.strip(" ,.-|*:-")
        return text

    def extract_specified_price(self):
        return menu_class.extract_specified_price(self.text)

    def is_abnormal_bbox_angle(self):
        if not (self.bbox_angle < 0.15 or self.bbox_angle > math.pi*2-0.15):
            print("[LOGMODE]", "abnormal bbox angle:", self.text, ", angle:", self.bbox_angle)
            return True

    def set_no_interest(self):
        self.cls = Class.no_interest

    def classify_node(self):
        if self.is_abnormal_bbox_angle():
            self.cls = Class.no_interest

    def is_vertical_line_with_other(self, other):
        return abs(self.bbox[-1][0] - other.bbox[-1][0]) < 0.01

    def get_center(self):
        x, y = 0, 0
        for idx in range(4):
            x += self.bbox[idx][0]
            y += self.bbox[idx][1]
        return [x/4, y/4]

    def get_bbox_size(self):
        return self.bbox[-2][1] - self.bbox[0][1]

    def get_points_distance(p1, p2):
        return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

    def get_distance(self, ano):
        return Vertice.get_points_distance(self.bbox[-1], ano.bbox[-1])

    def get_virtual_edge_info():
        return {"direction": Direction.no_direction, "angle": 1000, "distance": MAX_RANGE}

    def get_edge_info(self, ano):
        if self == ano:
            return __class__.get_virtual_edge_info()
        vector = [ano.bbox[-1][0]-self.bbox[-1][0], ano.bbox[-1][1]-self.bbox[-1][1]]
        radians = Direction.get_angle(vector)
        direction = Direction.get_direction_by_radians(radians)

        return {"direction": direction, "angle": radians, "distance": self.get_distance(ano)}

    def convert_pixel2ratio(bbox, size):
        new_bbox = []
        for p in bbox:
             new_bbox.append([p[0] / size[0], p[1] / size[1]])
        return new_bbox

    def get_n_nearest_vertices(vertice, vertice_lst, n_horizontal_min_values=10, n_vertical_min_values=1):
        info_lst = []
        for ano_vertice in vertice_lst:
            info_lst.append(vertice.get_edge_info(ano_vertice))

        res = []
        res_info = []
        for dir in Direction.Directions:
            select = []
            select_info = []
            # print(f"---{dir}---")
            for idx in sorted(range(len(info_lst)), key=lambda k: info_lst[k]["distance"]):
                # print("distance", distance_lst[idx], Direction.do_satify(distance_lst[idx], direction_lst[idx]))
                # print("direction", direction_lst[idx]["direction"], direction_lst[idx]["direction"] != dir)
                if not Direction.do_satify(info_lst[idx]) \
                    or info_lst[idx]["direction"] != dir:
                    continue
                select.append(idx)
                select_info.append((info_lst[idx]))
                # print(idx, distance_lst[idx], direction_lst[idx])
                if (Direction.is_horizontal(dir) and len(select) == n_horizontal_min_values) \
                    or (Direction.is_vertical(dir) and len(select) == n_vertical_min_values):
                    break
            res += select
            res_info += select_info
        return res, res_info
