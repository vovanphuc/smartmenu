import math
import cv2
import re
from graph.graph import ClassifiedGraph
from graph.menu_class import Class
from graph.direction import Direction

MEDIAN_THRESHOLD = 5.5


class RuleBasedPostProcessing:
    def get_img_size_helper(img):
        return img.shape[1], img.shape[0]

    def get_img_size(img_path):
        img = cv2.imread(img_path)
        return __class__.get_img_size_helper(img)

    def get_median_bbox_size(bbox_lst):
        size_lst = [bbox[-1][1] - bbox[0][1] for bbox in bbox_lst]
        size_lst = sorted(size_lst)
        while len(size_lst) > 0:
            mean = sum(size_lst) / len(size_lst)
            if mean - MEDIAN_THRESHOLD > size_lst[0] or mean + MEDIAN_THRESHOLD < size_lst[-1]:
                if mean - size_lst[0] > size_lst[-1] - mean:
                    size_lst = size_lst[1:]
                else:
                    size_lst = size_lst[:-1]
            else:
                break
        return (size_lst[0], size_lst[-1])
    
    def normalize_bboxs_by_right_points(bboxs):
        for bbox in bboxs:
            bbox[1][1] = bbox[0][1]
            bbox[2][1] = bbox[-1][1]
        return bboxs

    def is_vector_mean_valid(vector_lst, vector_mean):
        if len(vector_lst) < 4:
            return False

        count = 0
        dir_mean = Direction.get_angle(vector_mean, need_negative=True)
        for vec in vector_lst:
            dir_vec = Direction.get_angle(vec, need_negative=True)
            dif = min(abs(dir_vec - dir_mean), abs(dir_vec - dir_mean - 2 * math.pi))
            if dif < 0.005:
                count += 1

        return count / len(vector_lst) > 0.4

    def normalize_bboxs(bboxs):
        vector_lst = []
        for bbox in bboxs:
            vector_lst.append([bbox[0][0] - bbox[1][0], bbox[0][1] - bbox[1][1]])
            vector_lst.append([bbox[3][0] - bbox[2][0], bbox[3][1] - bbox[2][1]])
        vector_lst.sort(key=lambda k: Direction.get_angle(k, need_negative=True))
        vector_lst = vector_lst[math.ceil(len(vector_lst)*1/9): math.floor(len(vector_lst)*8/9)]
        vector_x = sum(v[0] for v in vector_lst) / len(vector_lst)
        vector_y = sum(v[1] for v in vector_lst) / len(vector_lst)

        if vector_x != 0 \
            and __class__.is_vector_mean_valid(vector_lst, [vector_x, vector_y]):
            for bbox in bboxs:
                if bbox[1][0] == bbox[0][0]:
                    bbox[0][1] = bbox[1][1]
                else:
                    new_y_01 = vector_y / -vector_x * bbox[1][0] + bbox[1][1]
                    bbox[0][1] = new_y_01
                    bbox[1][1] = new_y_01

                if bbox[2][0] == bbox[3][0]:
                    bbox[3][1] = bbox[2][1]
                else:
                    new_y_23 = vector_y / -vector_x * bbox[2][0] + bbox[2][1]
                    bbox[2][1] = new_y_23
                    bbox[3][1] = new_y_23

        return __class__.normalize_bboxs_by_right_points(bboxs)

    def classify_bbox_text_helper(bbox, median_bbox_size):
        size = bbox[-1][1] - bbox[0][1]
        if size <  median_bbox_size[0] * 0.7 or size > median_bbox_size[1] * 1.3:
            return Class.no_interest

    def classify_text(text_lst, bbox_lst, pred_lst=None):
        if not pred_lst or not isinstance(pred_lst, list):
            pred_lst = []

        if not bbox_lst:
            return text_lst, bbox_lst, pred_lst

        new_text_lst = []
        new_bbox_lst = []
        median_bbox_size = __class__.get_median_bbox_size(bbox_lst)
        for text, bbox in zip(text_lst, bbox_lst):
            text_pred = Class.classify_text(text)
            print("classify", text, text_pred)

            if Class.is_confusion_or_pair_class(text_pred):
                splited_text_list, splited_bbox_list = Class.split_confusion_or_pair_class(text_pred, text, bbox)
                if len(splited_text_list) <= 1:
                    text_pred = Class.price
                else:
                    if text_pred == Class.price_pair_type_pair:
                        new_sub_text_lst = splited_text_list
                        new_sub_bbox_lst = splited_bbox_list
                        pred_lst += [Class.price_m_type1, Class.price_l_type1, Class.price_m_type2, Class.price_l_type2]
                    else:
                        new_sub_text_lst, new_sub_bbox_lst, pred_lst = __class__.classify_text(splited_text_list, splited_bbox_list, pred_lst)
                    
                    new_text_lst += new_sub_text_lst
                    new_bbox_lst += new_sub_bbox_lst
                    continue

            new_text_lst.append(text)
            new_bbox_lst.append(bbox)
            bbox_pred = __class__.classify_bbox_text_helper(bbox, median_bbox_size)
            if bbox_pred and text_pred == Class.food_name:
                print("[LOGMODE] skip bbox by bbox size:", text)
                pred_lst.append(bbox_pred)
            else:
                pred_lst.append(text_pred)
        return new_text_lst, new_bbox_lst, pred_lst

    def normalize_name(text):
        text = text.upper().lstrip("()").rstrip("(")
        if text and text[-1] == ")" and text.find("(") == -1:
            text = text[:-1]

        text = text.replace("( ", "(") \
                    .replace(" )", ")") \
                    .replace(",", ", ") \
                    .replace(" ,", ",") \
                    .replace(" .", ".")
        text = re.sub("\s+", " ", text)
        text = re.sub("^\d{1,2}\.\s?", "", text)
        return text.strip(" ,.+->=|*:")

    def normalize_price(text):
        text = text.upper()
        if text != "NOT GIVEN":
            if text.find("/") != -1:
                text = text[:text.find("/")]
            text = re.sub("[^0-9]", "", text)
            if len(text) < 4:
                text += "000"
        return text

    def normalize_output(df, is_testing):
        for idx, row in df.iterrows():
            df.at[idx, "VietnameseName"] = __class__.normalize_name(row["VietnameseName"])
            if not is_testing:
                df.at[idx, "Price"] = __class__.normalize_price(row["Price"])
        return df

    def classify_on_graph(classes, bboxs, size, texts, img_name, is_testing=False):
        G = ClassifiedGraph()
        G.graph.update(image_name=img_name)
        vertices = G.build(classes, bboxs, size, texts)
        print("-----------",  G.graph.get("template"))

        G.classify_node()
        G.classify_graph()
        df = G.generate_results()
        df = __class__.normalize_output(df, is_testing)
        
        print(df)
        print("-----------------")

        # for idx, row in df.iterrows():
        #     print(row["VietnameseName"], "---", row["Price"])

        # # view
        # import matplotlib.pyplot as plt
        # import networkx as nx
        # plt.gca().invert_yaxis()
        # nx.draw(G, {idx: (vertices[idx].bbox[-1]) for idx in range(len(vertices))}, arrows=True)
        # plt.show()

        return df

if __name__ == "__main__":
    text = "DIMSUM 1 Phần 30k/6 viên"
    print(text, RuleBasedPostProcessing.classify_text_helper(text))
