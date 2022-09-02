import enum
from menu_class import Class


class Template(enum.Enum):
    template1 = 0 # template has only name and price, food name and price are as similar as amount, eg. image 009.jpeg
    template2 = 1 # name, price and size. food name and price are as similar as amount, eg. image 005.jpegs
    template3 = 2 # name, price and size. food name amount is larger than price, eg. image 001.jpegs
    template4 = 3

    def __eq__(self, other):
        """Overrides the default implementation"""
        try:
            res = self.value == other.value
        except:
            res = False
        finally:
            return res

    def classify_graph_template(classes):
        price_num = sum([1 for x in classes if x == Class.price])
        name_num = sum([1 for x in classes if x == Class.food_name])
        size_num = sum([1 for x in classes if x == Class.size])

        if price_num == 0 and name_num == 0:
            return Template.template4 
        elif (price_num == 0 and name_num > 3) \
            or (price_num * 1.7 < name_num):
            return Template.template3
        elif size_num == 0 \
            and min(price_num, name_num) / max(price_num, name_num) > 0.55:
            return Template.template1
        elif size_num > 0:
            return Template.template2
        else:
            return Template.template4