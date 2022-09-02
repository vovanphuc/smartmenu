import networkx as nx
import math
import pandas as pd
import unidecode
import os, sys
sys.path.append(os.path.dirname(__file__))
from direction import Direction
from vertice import Vertice
from pair import Pair
from template import Template
from menu_class import Class

PREFIX_HEADER_MATCH_THRESHOLD = 2


class ClassifiedGraph(nx.DiGraph):
    def get_vertice(self, node):
        return self.nodes[node]["vertice"]

    def add_food_name_matching(self, node, is_on_the_right=False):
        if not self.is_in_matching_name_nodes(node, is_on_the_right):
            matching_list = self.get_food_name_matching_list(is_on_the_right)
            matching_list.append(node)
            self.set_food_name_matching_list(matching_list, is_on_the_right)

    def get_food_name_matching_list(self, is_on_the_right):
        if is_on_the_right:
            matching_list = self.graph.get("matching_name_nodes_on_the_right", [])
        else:
            matching_list = self.graph.get("matching_name_nodes", [])
        return matching_list

    def set_food_name_matching_list(self, matching_list, is_on_the_right):
        if is_on_the_right:
            self.graph.update(matching_name_nodes_on_the_right=matching_list)
        else:
            self.graph.update(matching_name_nodes=matching_list)

    def is_in_matching_name_nodes(self, node, is_on_the_right="ALL"):
        if is_on_the_right == "ALL":
            return (node in self.get_food_name_matching_list(is_on_the_right=False) \
                   or node in self.get_food_name_matching_list(is_on_the_right=True))
        return node in self.get_food_name_matching_list(is_on_the_right)

    def get_top_node(self, node):
        for _, ano in self.out_edges(node):
            if self.edges[node, ano]["direction"] == Direction.top:
                return ano
        return None

    def get_bot_node(self, node):
        for _, ano in self.out_edges(node):
            if self.edges[node, ano]["direction"] == Direction.bot:
                return ano
        return None

    def is_lay_out_interest_box(self, node):
        vertice = self.get_vertice(node)
        is_lay_out = vertice.bbox[0][1] > 0.8 or vertice.bbox[0][1] < 0.2
        if is_lay_out:
            print("[LOGMODE] lay out interest_box", vertice.text)
        return is_lay_out

    def is_group_header(self, node):
        vertice = self.get_vertice(node)
        for _, ano in self.out_edges(node):
            if not self.edges[node, ano]["direction"] == Direction.bot:
                continue
            
            for out_ano, _ in self.in_edges(ano):
                if self.edges[out_ano, ano].get("pair", None) == Pair.price_and_name:
                    if vertice.get_bbox_size() > 1.2 * self.get_vertice(ano).get_bbox_size():
                        print("[LOGMODE] ignoring group header type 1", self.get_vertice(node).text)
                        return True
                    elif vertice.get_bbox_size() > 1.12 * self.get_vertice(ano).get_bbox_size() \
                        and self.get_top_node(node) \
                        and Direction.is_distance_of_groups(self.edges[node, self.get_top_node(node)]["distance"]):
                            print("[LOGMODE] ignoring group header type 2", self.get_vertice(node).text)
                            return True
                    elif self.get_top_node(node) \
                        and Direction.is_distance_of_far_groups(self.edges[node, self.get_top_node(node)]["distance"]):
                            print("[LOGMODE] ignoring group header type 3", self.get_vertice(node).text)
                            return True
                    elif self.get_top_node(node) == None:
                        print("[LOGMODE] ignoring group header type 4", self.get_vertice(node).text)
                        return True
        return False

    def is_prefix_header(self, node):
        norm_prefix = unidecode.unidecode(self.get_vertice(node).text)
        matched_count = 0
        for other in self.nodes:
            if other == node:
                continue

            norm_other = unidecode.unidecode(self.get_vertice(other).text)
            if norm_other.startswith(norm_prefix) and norm_other != norm_prefix:
                matched_count += 1
                if matched_count >= PREFIX_HEADER_MATCH_THRESHOLD:
                    print("[LOGMODE] prefix header", self.get_vertice(node).text)
                    return True
        return False

    def is_below_name_price_pair(self, node):
        for _, ano in self.out_edges(node):
            if not self.edges[node, ano]["direction"] == Direction.top \
                or not Direction.is_distance_of_sub_name(self.edges[node, ano]["distance"] - self.get_vertice(node).get_bbox_size()):
                continue
            
            for out_ano, _ in self.in_edges(ano):
                if not self.edges[out_ano, ano].get("pair", None) == Pair.price_and_name:
                    continue

                if self.graph.get("template") == Template.template3:
                    print("[LOGMODE] ignoring info food type 1", self.get_vertice(node).text)
                    return True
                elif self.get_bot_node(node) == None \
                    or Direction.is_distance_of_far_groups(self.edges[node, self.get_bot_node(node)]["distance"] - self.get_vertice(node).get_bbox_size()):
                    print("[LOGMODE] ignoring info food type 2", self.get_vertice(node).text)
                    return True
        return False

    def classify_node(self):
        for node in list(self.nodes):
            vertice = self.get_vertice(node)
            vertice.classify_node()
            
    def build(self, classes, bboxs, size, texts):
        vertices = []
        for idx, (cls, bbox, text) in enumerate(zip(classes, bboxs, texts)):
            vertices.append(Vertice(cls, Vertice.convert_pixel2ratio(bbox, size), text))
            self.add_node(idx, vertice=vertices[idx])

        for idx in range(len(vertices)):
            dst_lst, info_lst = Vertice.get_n_nearest_vertices(vertices[idx], vertices)
            print("edge", idx, vertices[idx].text, dst_lst)
            if not dst_lst:
                print("[LOGMODE] node has no connections:", vertices[idx].text)
            for dst, info in zip(dst_lst, info_lst):
                self.add_edge(idx, dst, **info)

        self.graph.update(template=Template.classify_graph_template(classes))

        return vertices # for debug

    def reclassify_template(self):
        old_template = self.graph.get("template")
        classes = []
        for node in self.nodes:
            classes.append(self.get_vertice(node).cls)
        self.graph.update(template=Template.classify_graph_template(classes))
        if not self.graph.get("template") == old_template:
            print(f"[LOGMODE] change template from {old_template} to {self.graph.get('template')}")

    def get_node_by_pair(self, node, pair, direction=None):
        for _, ano in self.out_edges(node):
            edge = self.edges[node, ano]
            if edge.get("pair", None) == pair \
                and (direction == None or direction == edge.get("direction", None)):
                return ano
        return None

    def generate_results_for_size_first(self, df, idx):
        for src, dst in self.edges:
            if "pair" in self.edges[src, dst]:
                if self.edges[src, dst]["pair"] == Pair.size_and_price:
                    size_node = src
                    price_node = dst
                    name_node = self.get_node_by_pair(price_node, Pair.price_and_name)
                    if name_node is None:
                        print("[LOGMODE] Can not find name node to pair:", self.get_vertice(size_node).text, "---", self.get_vertice(size_node).text)
                        continue
                    self.edges[price_node, name_node]["pair"] = Pair.price_and_name_but_used_on_pairing_size

                    df.at[idx, "ImageName"] = self.graph["image_name"]
                    df.at[idx, "VietnameseName"] = self.get_vertice(name_node).text + " " \
                                                 + self.get_vertice(size_node).text
                    df.at[idx, "Price"] = self.get_vertice(price_node).text
                    idx += 1
        return df, idx

    def generate_results_for_predicted_size(self, df, idx):
        for src, dst in self.edges:
            if self.edges[src, dst].get("pair", None) == Pair.price_size_L_and_price_size_M \
                and Direction.is_distance_of_price_pair(self.edges[src, dst]["distance"]):
                flag = False
                for _, src_out in self.out_edges(src):
                    if not self.edges[src, src_out].get("direction", None) == Direction.left \
                        or not self.get_vertice(src_out).cls == Class.food_name:
                        continue
                    name_on_the_right = src_out
                    if name_on_the_right is not None \
                        and self.edges[src, name_on_the_right].get("distance") < self.edges[src, dst].get("distance"):
                        flag = True
                        break
                if flag:
                    continue

                price_node_size_L = src
                price_node_size_M = dst
                name_node_size_L = self.get_node_by_pair(price_node_size_L, Pair.price_and_name)
                if name_node_size_L is None:
                    name_node_size_L = self.get_node_by_pair(price_node_size_L, Pair.price_and_name_but_used_on_pairing_size)
                name_node_size_M = self.get_node_by_pair(price_node_size_M, Pair.price_and_name)
                if name_node_size_M is None:
                    name_node_size_M = self.get_node_by_pair(price_node_size_M, Pair.price_and_name_but_used_on_pairing_size)

                if not name_node_size_L and not name_node_size_M:
                    print("[LOGMODE] Can not find name node to pair predicted size nodes:", self.get_vertice(price_node_size_L).text, "---", self.get_vertice(price_node_size_M).text)
                    continue

                if not name_node_size_M:
                    name_node_size_M = name_node_size_L
                elif not name_node_size_L:
                    name_node_size_L = name_node_size_M

                if not self.has_edge(price_node_size_L, name_node_size_L) \
                    or self.edges[price_node_size_L, name_node_size_L].get("pair", None) == Pair.price_and_name:
                    if not self.has_edge(price_node_size_L, name_node_size_L):
                        self.add_edge(price_node_size_L, name_node_size_L, **self.get_vertice(price_node_size_L).get_edge_info(self.get_vertice(name_node_size_L)))
                    self.edges[price_node_size_L, name_node_size_L]["pair"] = Pair.price_and_name_but_used_on_pairing_predicted_size
                    df.at[idx, "ImageName"] = self.graph["image_name"]
                    df.at[idx, "VietnameseName"] = self.get_vertice(name_node_size_L).text + " L"
                    df.at[idx, "Price"] = self.get_vertice(price_node_size_L).text
                    idx += 1
                if not self.has_edge(price_node_size_M, name_node_size_M) \
                    or self.edges[price_node_size_M, name_node_size_M].get("pair", None) == Pair.price_and_name:
                    if not self.has_edge(price_node_size_M, name_node_size_M):
                        self.add_edge(price_node_size_M, name_node_size_M, **self.get_vertice(price_node_size_M).get_edge_info(self.get_vertice(name_node_size_M)))
                    self.edges[price_node_size_M, name_node_size_M]["pair"] = Pair.price_and_name_but_used_on_pairing_predicted_size
                    df.at[idx, "ImageName"] = self.graph["image_name"]
                    df.at[idx, "VietnameseName"] = self.get_vertice(name_node_size_M).text + " M"
                    df.at[idx, "Price"] = self.get_vertice(price_node_size_M).text
                    idx += 1
        return df, idx

    def generate_results(self):
        df = pd.DataFrame()
        idx = 0

        # process size first
        df, idx = self.generate_results_for_size_first(df, idx)
        # process predicted size
        df, idx = self.generate_results_for_predicted_size(df, idx)

        # process other
        for src, dst in self.edges:
            if "pair" in self.edges[src, dst]:
                if self.edges[src, dst]["pair"] == Pair.price_and_name:
                    df.at[idx, "ImageName"] = self.graph["image_name"]
                    df.at[idx, "VietnameseName"] = self.get_vertice(dst).text
                    df.at[idx, "Price"] = self.get_vertice(src).text
                    idx += 1
                    # print(texts[dst], "---", texts[src])
                elif self.edges[src, dst]["pair"] == Pair.name_and_not_offered_price:
                    df.at[idx, "ImageName"] = self.graph["image_name"]
                    df.at[idx, "VietnameseName"] = self.get_vertice(src).text
                    df.at[idx, "Price"] = "NOT GIVEN"
                    idx += 1
                    # print(texts[src], "---", "NOT GIVEN")
                elif self.edges[src, dst]["pair"] == Pair.ignored_name:
                    pass
                elif self.edges[src, dst]["pair"] == Pair.name_specified_price:
                    vertice = self.get_vertice(src)
                    df.at[idx, "ImageName"] = self.graph["image_name"]
                    df.at[idx, "VietnameseName"] = vertice.text
                    df.at[idx, "Price"] = vertice.specied_price
                    idx += 1

        return df

    def solve_price_node_pair_name(self, node):
        if not self.out_edges(node):
            return False

        min_dis_name_price = None
        min_dis_node = None
        for _, ano in self.out_edges(node):
            ano_vertice = self.get_vertice(ano)
            edge = self.edges[node, ano]
            if ano_vertice.cls == Class.food_name and \
                edge["direction"] == Direction.left:
                edge["pair"] = Pair.tmp_price_and_name
                scalar = edge["distance"] * 1000 ** abs(edge["angle"] - math.pi)
                if min_dis_node is None or scalar < min_dis_name_price:
                    min_dis_name_price = scalar
                    min_dis_node = ano
        if min_dis_node is not None:
           self.edges[node, min_dis_node]["pair"] = Pair.price_and_name
           self.add_food_name_matching(min_dis_node)
           self.graph.update(price_and_name=self.graph.get("price_and_name", 0) + 1)
           return True
        return False

    def solve_price_node_pair_name_on_the_right(self, node):
        if not self.out_edges(node):
            return False

        min_dis_name_price = None
        min_dis_node = None
        for _, ano in self.out_edges(node):
            ano_vertice = self.get_vertice(ano)
            edge = self.edges[node, ano]
            if ano_vertice.cls == Class.food_name and \
                edge["direction"] == Direction.right:
                edge["pair"] = Pair.tmp_price_and_name_on_the_right
                scalar = edge["distance"] * 1000 ** abs(edge["angle"])
                if min_dis_node is None or scalar < min_dis_name_price:
                    min_dis_name_price = scalar
                    min_dis_node = ano
        if min_dis_node is not None:
           self.edges[node, min_dis_node]["pair"] = Pair.price_and_name_on_the_right
           self.add_food_name_matching(min_dis_node, is_on_the_right=True)
           self.graph.update(price_and_name_on_the_right=self.graph.get("price_and_name_on_the_right", 0) + 1)
           return True
        return False

    def solve_price_node_pair_price(self, node):
        if not self.out_edges(node):
            return False

        min_dis_name_price = None
        min_dis_node = None
        for _, ano in self.out_edges(node):
            ano_vertice = self.get_vertice(ano)
            edge = self.edges[node, ano]

            if ano_vertice.cls == Class.price \
                and edge["direction"] == Direction.left:
                scalar = edge["distance"] * 1000 ** abs(edge["angle"] - math.pi)
                if min_dis_node is None or scalar < min_dis_name_price:
                    min_dis_name_price = scalar
                    min_dis_node = ano
        if min_dis_node is not None:
           print("[LOGMODE] price_size_L_and_price_size_M", self.get_vertice(node).text, "---", self.get_vertice(min_dis_node).text)     
           self.edges[node, min_dis_node]["pair"] = Pair.price_size_L_and_price_size_M
           return True
        return False

    def solve_price_node(self, node):
        is_matched = self.solve_price_node_pair_name(node)
        is_matched = self.solve_price_node_pair_name_on_the_right(node) or is_matched
        is_matched = self.solve_price_node_pair_price(node) or is_matched 
        # if not is_matched:
        #     vertice = self.get_vertice(node)
        #     print("[LOGMODE] can not pair this price", vertice.text)
        #     vertice.set_no_interest()

    def solve_food_name_node(self, node):
        if not self.out_edges(node):
            return
        
        min_len_name_price = None
        min_len_node = None
        # print("-----------")
        # print(texts[node])
        for _, ano in self.out_edges(node):
            ano_vertice = self.get_vertice(ano)
            edge = self.edges[node, ano]
            if ano_vertice.cls == Class.price and \
                (edge["direction"] == Direction.right \
                or edge["direction"] == Direction.left):
                edge["pair"] = Pair.tmp_name_and_price
                # print(texts[ano])
                # print("old", edge["distance"], "min_len_name_price", min_len_name_price)
                # print("old", edge["angle"])
                scalar = edge["distance"] * 1000 ** abs(edge["angle"] - math.pi if edge["direction"] == Direction.left else edge["angle"])
                if min_len_node is None or scalar < min_len_name_price:
                    min_len_name_price = scalar
                    min_len_node = ano
        if min_len_node is not None:
           self.edges[node, min_len_node]["pair"] = Pair.name_and_price
           self.graph.update(name_and_price= self.graph.get("name_and_price", 0) + 1)
        else:
            self.add_edge(node, node, **Vertice.get_virtual_edge_info())
            self.edges[node, node]["pair"] = Pair.tmp_name_and_not_offered_price
            # self.get_vertice(node).cls = Class.no_interest

    def trace_size_for_successor_price(self, size_node, successor):
        if not self.out_edges(successor):
            return
        
        size_vertice = self.get_vertice(size_node)
        for _, ano in self.out_edges(successor):
            ano_vertice = self.get_vertice(ano)
            edge = self.edges[successor, ano]
            if ano_vertice.cls == Class.price and \
                edge["direction"] == Direction.bot:
                if not self.has_edge(size_node, ano):
                    edge_info = size_vertice.get_edge_info(ano_vertice)
                    if not edge_info["direction"] == Direction.bot:
                        print("[LOGMODE] skip trace size for successor price by direction not is bot:", \
                            size_vertice.text, "---", ano_vertice.text)
                        continue
                    self.add_edge(size_node, ano, **edge_info)
                self.edges[size_node, ano]["pair"] = Pair.size_and_price
                print("[LOGMODE] trace size for successor price:", \
                            size_vertice.text, "---", ano_vertice.text)
                self.trace_size_for_successor_price(size_node=size_node, successor=ano)

    def solve_size_node(self, node):
        vertice = self.get_vertice(node)
        if vertice.text in ["SIZES", "SIZE S"]:
            vertice.text = "S"
        elif vertice.text in ["SIZEM", "SIZE M"]:
            vertice.text = "M"
        elif vertice.text in ["SIZEL", "SIZE L"]:
            vertice.text = "L"

        if not self.out_edges(node):
            return
            
        for _, ano in list(self.out_edges(node)):
            ano_vertice = self.get_vertice(ano)
            edge = self.edges[node, ano]
            if ano_vertice.cls == Class.price and \
                edge["direction"] == Direction.bot:
                edge["pair"] = Pair.size_and_price
                self.trace_size_for_successor_price(size_node=node, successor=ano)

    def decide_price_node_pair_name_on_the_right(self):
        pairs = self.graph.get("price_and_name_on_the_right", 0)
        if pairs < 3:
            return

        for src, dst in self.edges:
            edge = self.edges[src, dst]
            if not edge.get("pair", None) == Pair.price_and_name_on_the_right:
                continue

            left_name = self.get_node_by_pair(src, Pair.price_and_name, Direction.left)
            if left_name is not None:
                continue
            else:
                edge["pair"] = Pair.price_and_name

    def solve_specified_price_node(self, node):
        key, value = self.get_vertice(node).extract_specified_price()
        specified_price_lst = self.graph.get("specified_price_lst", [])
        specified_price_lst.append([key, value])
        self.graph.update(specified_price_lst=specified_price_lst)


    def try_to_solve_tmp_name_not_offered_price(self):
        waiting_lst = []

        if self.is_specified_price():
            self.pair_specified_price()

        for src, dst in self.edges:
            text = self.get_vertice(src).text
            edge = self.edges[src, dst]
            if edge.get("pair", None) == Pair.tmp_name_and_not_offered_price:
                print("tmp_name_not_offered_price:", text)
                if "TRÀ, CAFE" in text \
                    or "TRÀ, CAFFE" in text \
                    or "TRÀ, CÀ PHÊ" in text \
                    or "NƯỚC" in text \
                    or "SOFT DRINK" in text \
                    or "TRÁI CÂY" in text \
                    or "SMOTHIE" in text \
                    or "MOCTAIL" in text \
                    or "COCKTAIL" in text:
                    print("[LOGMODE] ignore by containing general name:", text)
                elif self.is_below_name_price_pair(src) \
                    or self.is_group_header(src) \
                    or self.is_prefix_header(src):
                    self.edges[src, dst]["pair"] = Pair.ignored_name
                elif self.graph.get("template") == Template.template1:
                    if self.is_lay_out_interest_box(src):
                        self.edges[src, dst]["pair"] = Pair.ignored_name
                    else:
                        waiting_lst.append([src, dst])
                elif self.graph.get("template") == Template.template3:
                    waiting_lst.append([src, dst])
                elif self.graph.get("template") == Template.template4:
                    waiting_lst.append([src, dst]) 
                else:
                    print("[LOGMODE] waiting to process:", text)

        if len(waiting_lst) >= 2:
            for src, dst in waiting_lst:
                if self.is_inline_with_food_name_nodes(src):
                    self.edges[src, dst]["pair"] = Pair.name_and_not_offered_price
                    print("[LOGMODE] inline_with_food_name_nodes", self.nodes[src]["vertice"].text)
                else:
                    print("[LOGMODE] ignoring tmp_name_and_not_offered_price by not inline with other food name nodes:", self.get_vertice(src).text)
        else:
            for src, _ in waiting_lst:
                print("[LOGMODE] ignoring tmp_name_and_not_offered_price by not in group:", self.get_vertice(src).text)

    def is_inline_with_food_name_nodes(self, node):
        inline_count = 0
        total_food_name = 0
        node_vtc = self.get_vertice(node)
        for other in self.nodes:
            if node == other:
                continue
            other_vtc = self.get_vertice(other)
            if other_vtc.cls == Class.food_name:
                total_food_name += 1
                if node_vtc.is_vertical_line_with_other(other_vtc):
                    inline_count += 1
        if inline_count >= total_food_name / 3:
            return True
        return False

    def is_specified_price(self):
        if self.graph.get("specified_price_lst", None):
            return True

    def pair_specified_price(self):
        specified_price_lst = self.graph.get("specified_price_lst")
        for src, dst in self.edges:
            edge = self.edges[src, dst]
            if not edge.get("pair", None) == Pair.tmp_name_and_not_offered_price:
                continue
            for key, value in specified_price_lst:
                if key in Class.normalize(self.get_vertice(src).text.upper()):
                    print("[LOGMODE] pair name with specified price", self.get_vertice(src).text, "---", key, ":", value)
                    edge["pair"] = Pair.name_specified_price
                    self.get_vertice(src).specied_price = value

    def check_name_and_price(self):
         for src, dst in self.edges:
            edge = self.edges[src, dst]
            if edge.get("pair", None) == Pair.name_and_price and \
                not self.is_in_matching_name_nodes(src):
                print("[LOGMODE] name_and_price have not been processed yet!",
                        self.get_vertice(src).text, "---", 
                        self.get_vertice(dst).text)
                edge["pair"] = Pair.tmp_name_and_not_offered_price

    def classify_graph(self):
        for node in list(self.nodes):
            vertice = self.get_vertice(node)
            if vertice.cls == Class.no_interest:
                continue
            elif Class.is_price_pair_type(vertice.cls):
                self.solve_price_node(node)
            elif vertice.cls == Class.price:
                self.solve_price_node(node)
            elif vertice.cls == Class.food_name:
                self.solve_food_name_node(node)
            elif vertice.cls == Class.size:
                self.solve_size_node(node)
            # elif vertice.cls == Class.confusion_name_price:
            #     self.solve_confusion_name_price_node(node)
            # elif vertice.cls == Class.confusion_price_size:
            #     self.solve_confusion_price_size_node(node)
            # elif vertice.cls == Class.confusion_name_size:
            #     self.solve_confusion_name_size_node(node)
            elif vertice.cls == Class.specified_price:
                self.solve_specified_price_node(node)

        self.decide_price_node_pair_name_on_the_right()
        self.check_name_and_price()
        # re-classify template
        self.reclassify_template()
        self.try_to_solve_tmp_name_not_offered_price()


if __name__ == "__main__":
    a = Vertice(0, [(0, 2), (2, 2), (2, 0), (0, 0)])
    b = Vertice(0, [(-10, 2), (-12, 2), (-12, 0), (-10, 0)])
    dis = a.get_distance(b)
    dir = a.get_edge_info(b)
    if dis != 2.8284271247461903 or dir != {'direction': Direction.left, 'angle': 0.7853981633974483}:
        print("wrong!")
    print("dis", dis, "dir", dir)
