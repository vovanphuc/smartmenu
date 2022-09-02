import enum


class Pair(enum.Enum):
    name_and_price = 0
    tmp_name_and_price = 1
    price_and_name = 2
    tmp_price_and_name = 3
    name_and_size = 4
    tmp_name_and_size = 5
    size_and_name = 6
    tmp_size_and_name = 7
    price_and_size = 8
    tmp_price_and_size = 9
    size_and_price = 10
    tmp_size_and_price = 11
    name_and_not_offered_price = 12
    tmp_name_and_not_offered_price = 13
    ignored_name = 14
    tmp_ignored_name = 15
    name_specified_price = 16
    price_and_name_but_used_on_pairing_size = 17
    price_size_L_and_price_size_M = 18
    price_and_name_but_used_on_pairing_predicted_size = 19
    price_and_name_on_the_right = 20
    tmp_price_and_name_on_the_right = 21

    def __eq__(self, other):
        """Overrides the default implementation"""
        try:
            res = self.value == other.value
        except:
            res = False
        finally:
            return res