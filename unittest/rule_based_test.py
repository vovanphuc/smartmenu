import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest
from rule_based import RuleBasedPostProcessing as pp
import graph.menu_class
import json
import pandas as pd

COUNT = 0
TRUE = 0


class RuleBasedTest(unittest.TestCase):
    def test_does_contain_pattern_time(self):
        must_match = ["12H30", "10H", "TỪ 6H ĐẾN 13H", "22:30-23:59", "8H"]
        for text in must_match:
            text = graph.menu_class.Class.normalize(text)
            self.assertTrue(graph.menu_class.does_contain_pattern(text, graph.menu_class.TIME_PATTERN), msg=f"Fail at {text}")

        must_not_match = ["12", "125K", "45$", "8", "85 NGÀN"]
        for text in must_not_match:
            text = graph.menu_class.Class.normalize(text)
            self.assertFalse(graph.menu_class.does_contain_pattern(text, graph.menu_class.TIME_PATTERN), msg=f"Fail at {text}")

    def test_does_contain_pattern_price(self):
        must_match = ["12", "125K", "12 K", "45$", "8", "85 NGÀN", "45K ĐĨA", "120.000 CON"]
        for text in must_match:
            text = graph.menu_class.Class.normalize(text)
            self.assertTrue(graph.menu_class.does_contain_price_pattern(text), msg=f"Fail at {text}")

        must_not_match = ["12H30", "10H", "TỪ 6H ĐẾN 13H", "22:30-23:59", "8H", ")0961 187274"]
        for text in must_not_match:
            text = graph.menu_class.Class.normalize(text)
            self.assertFalse(graph.menu_class.does_contain_price_pattern(text), msg=f"Fail at {text}")

    def test_does_contain_pattern_combo_price(self):
        must_match = ["COMBO ĐỒNG GIÁ 125K"]
        for text in must_match:
            text = graph.menu_class.Class.normalize(text)
            self.assertTrue(graph.menu_class.does_contain_combo_price_pattern(text), msg=f"Fail at {text}")

        must_not_match = []
        for text in must_not_match:
            text = graph.menu_class.Class.normalize(text)
            self.assertFalse(graph.menu_class.does_contain_combo_price_pattern(text), msg=f"Fail at {text}")

    def test_stable_cases(self):
        data_lst = []
        img_lst = []
        with open("unittest/testcases.txt", "r", encoding="utf-8") as f:
            while True:
                line = f.readline()
                if not line:
                    break
                img_lst.append(line.split("\t")[0])
                data_lst.append(json.loads(line.split("\t")[1]))

        text_lst: list = []
        bbox_lst: list = []
        for data in data_lst:
            text_obj = []
            bbox_obj = []
            for obj in data:
                text = obj["transcription"]
                bbox = obj["points"]
                text_obj.append(text)
                bbox_obj.append(bbox)
            text_lst.append(text_obj)
            bbox_lst.append(bbox_obj)

        checked_lst = [0, 2, 3, 4, 5, 6, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 47, 48, 49, 51, 53, 52]
        for idx, (_img, _texts, _bboxs) in enumerate(zip(img_lst, text_lst, bbox_lst)):
            if idx not in checked_lst:
                continue
            
            size = pp.get_img_size(os.path.join("images_full_qai", _img))
            _bboxs = pp.normalize_bboxs(_bboxs)
            _texts, _bboxs, _preds = pp.classify_text(_texts, _bboxs)
            df = pp.classify_on_graph(_preds, _bboxs, size, _texts, _img, is_testing=True)

            label_path = os.path.join("unittest/ground_truth", os.path.basename(_img) + ".csv")
            label = pd.read_csv(label_path, sep=',', encoding="utf-8", dtype=str)
            
            df = df[["VietnameseName", "Price"]]
            
            if not df.equals(label):
                if len(df) != len(label):
                    print("different len!")

                for idx in range(len(df)):
                    if df.loc[idx, "VietnameseName"] != label.loc[idx, "VietnameseName"]:
                        print(df.loc[idx, "VietnameseName"], "---", label.loc[idx, "VietnameseName"])
                    if df.loc[idx, "Price"] != label.loc[idx, "Price"]:
                        print(df.loc[idx, "Price"], "---", label.loc[idx, "Price"])

            self.assertTrue(df.equals(label), f"Fail on {_img}")

    # def test_classification(self):
    #     data_lst = []
    #     img_lst = []
    #     with open("unittest/testcases.txt", "r", encoding="utf-8") as f:
    #         while True:
    #             line = f.readline()
    #             if not line:
    #                 break
    #             img_lst.append(line.split("\t")[0])
    #             data_lst.append(json.loads(line.split("\t")[1]))

    #     text_lst: list = []
    #     bbox_lst: list = []
    #     for data in data_lst:
    #         text_obj = []
    #         bbox_obj = []
    #         for obj in data:
    #             text = obj["transcription"]
    #             bbox = obj["points"]
    #             text_obj.append(text)
    #             bbox_obj.append(bbox)
    #         text_lst.append(text_obj)
    #         bbox_lst.append(bbox_obj)

    #     count = 27
    #     for _img, _texts, _bboxs in zip(img_lst, text_lst, bbox_lst):
    #         if count:
    #             count -= 1
    #             continue
    #         size = pp.get_img_size(os.path.join("images_full_qai", _img))
    #         _bboxs = pp.normalize_bboxs(_bboxs)
    #         print("processing image:", _img, size)
    #         _texts, _bboxs, _preds = pp.classify_text(_texts, _bboxs)

    #         df = pp.classify_on_graph(_preds, _bboxs, size, _texts, _img, is_testing=True)

    #         df.to_csv(os.path.join("unittest/ground_truth/", os.path.basename(_img) + ".csv"), sep=',', encoding="utf-8", index=False)
    #         break
        
    #     # print("True vs count", TRUE, COUNT)

if __name__ == "__main__":
    unittest.main(verbosity=2)
    