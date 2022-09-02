import enum
import re
import unidecode
TIME_PATTERN = r"((\d{1,3})[:HG])(\d{1,3}|OO)?"
PRICE_PATTERN = r"(\d{1,3}[\.,\s]?\d{3}|\d{1,3})[\s\.]?(K|\$|VND)?\s?\/?\s?(DIA|CON|CON|PHAN|NGAN|VIEN|TO|THO|CAY|KG|LON|CHAI|CAI|THANH)?$"
PRICE_PATTERN_TYPE_2 = r"(\d{1,3}[\.,\s]?\d{3}|\d{1,3})[\s\.]?(K|\$|VND)?\s?\/\s?.{0,5}$"
SHORT_PRICE_PATTERN = r"(\d{1,3}[\.,\s]?\d{3}|\d{1,3})[\s\.]?(K|\$|VND)\s?\/?\s?.{0,5}$"
SHORT_PRICE_PATTERN_TYPE_2 = r"(\d{1,3}[\.,\s]\d{3})\s?\/?\s?.{0,5}$"
SHORT_PRICE_PATTERN_TYPE_3 = r"(\d{1,3}[\.,\s]?\d{3}|\d{1,3})[\s\.]?(K|\$|VND)?\s?\/?\s?.{0,5}$"
COMBO_PRICE_PATTERN = r"(C[O0]MB[O0]).{0,10}(D[O0]NG G[I1][4A]).{0,5}\d{1,3}"
PRICE_PAIR_PATTERN = r"^\d{2}[\s\.]?(K|\$|VND)?[\|\/ -\.]*\d{2}[\s\.]?(K|\$|VND)?$"
PRICE_PAIR_PATTERN_TYPE_2 = r"^\d{1,3}[\s\.]?(K|\$|VND)[\|\/ -\.]*\d{1,3}[\s\.]?(K|\$|VND)$"
PRICE_TRIPLETS_PATTERN = r"^\d{2}[\s\.]?(K|\$|VND)?[\|\/ -\.]*\d{2}[\s\.]?(K|\$|VND)?[\|\/ -\.]*\d{2}[\s\.]?(K|\$|VND)?$"
PRICE_TRIPLETS_PATTERN_TYPE_2 = r"^\d{1,3}[\s\.]?(K|\$|VND)[\|\/ -\.]*\d{1,3}[\s\.]?(K|\$|VND)[\|\/ -\.]*\d{1,3}[\s\.]?(K|\$|VND)$"
PRICE_PAIR_TYPE_PAIR_PATTERN = r"^\d{2}[\s\.]?(K|\$|VND)?[\|\/ -\.]*\d{2}[\s\.]?(K|\$|VND)?[\|\/ -\.]*\d{2}[\s\.]?(K|\$|VND)?[\|\/ -\.]*\d{2}[\s\.]?(K|\$|VND)?$"
PRICE_PAIR_TYPE_PAIR_PATTERN_TYPE_2 = r"^\d{1,3}[\s\.]?(K|\$|VND)[\|\/ -\.]*\d{1,3}[\s\.]?(K|\$|VND)[\|\/ -\.]*\d{1,3}[\s\.]?(K|\$|VND)[\|\/ -\.]*\d{1,3}[\s\.]?(K|\$|VND)$"
SIZE_PRICE_PATTERN = r"[SML][:\. ]?(\d{1,3}[\.,\s]?\d{3}|\d{1,3})\s?(K|\$|VND)?$"
PHONE_PATTERN = r"\d{8,12}"
SPECIFIED_PRICE_PATTERN = r".{0,20}(D[O0]NG G[I1][4A]).{0,5}\d{1,3}"
YEAR_PATTERN = r".*(\D|^)(18|19|20)\d{2}$"
DOUBLE_ZEROS_PATTERN = r".*0{2}$"
DESCRIPTION_PATTERN = r"^\(.+\)$"
SIZE_LIST = ["S", "M", "L", "SIZES", "SIZEM", "SIZEL", "SIZE S", "SIZE M", "SIZE L"]

EXTRACT_SPECIFIED_PRICE_PATTERN = r"(.*)(D[O0]NG G[I1][4A])(.*)$"
EXTRACT_PRICE_PAIR_PATTERN = r"^(\d{2}).*(\d{2}).*$"
EXTRACT_PRICE_PAIR_PATTERN_TYPE_2 = r"^(\d{1,3}K|\d{1,3}\$|\d{1,3}VND)[\|\/ -\.]*(\d{1,3}K|\d{1,3}\$|\d{1,3}VND)$"
EXTRACT_PRICE_TRIPLETS_PATTERN = r"^(\d{2})[\|\/ -\.]*(\d{2})[\|\/ -\.]*(\d{2})[\$K]?$"
EXTRACT_PRICE_TRIPLETS_PATTERN_TYPE_2 = r"^(\d{1,3}K|\d{1,3}\$|\d{1,3}VND)[\|\/ -\.]*(\d{1,3}K|\d{1,3}\$|\d{1,3}VND)[\|\/ -\.]*(\d{1,3}K|\d{1,3}\$|\d{1,3}VND)$"
EXTRACT_PRICE_PAIR_TYPE_PAIR_PATTERN = r"^(\d{2})[\|\/ -\.]*(\d{2})[\|\/ -\.]*(\d{2})[\|\/ -\.]*(\d{2})$"
EXTRACT_PRICE_PAIR_TYPE_PAIR_PATTERN_TYPE_2 = r"^(\d{1,3}K|\d{1,3}\$|\d{1,3}VND)[\|\/ -\.]*(\d{1,3}K|\d{1,3}\$|\d{1,3}VND)[\|\/ -\.]*(\d{1,3}K|\d{1,3}\$|\d{1,3}VND)[\|\/ -\.]*(\d{1,3}K|\d{1,3}\$|\d{1,3}VND)$"
# EXTRACT_CONFUSION_PRICE_SIZE_PATTERN = ""
# EXTRACT_CONFUSION_NAME_SIZE_PATTERN = ""


def does_contain_pattern(text, pattern):
    m = re.findall(pattern, text)
    return len(m) > 0

def does_contain_price_pattern(text):
    check_pattern = (does_contain_pattern(text, PRICE_PATTERN) \
            or does_contain_pattern(text, PRICE_PATTERN_TYPE_2)) \
        and not does_contain_pattern(text, TIME_PATTERN) \
        and not does_contain_pattern(text, PHONE_PATTERN)
    check_valid = True
    for ch in text:
        if ch in ['1', '2', '3', '4', '5', '6', '7', '8', '9']:
            break
        elif ch == '0':
            check_valid = False
            break
        else:
            continue
    return check_pattern and check_valid

def does_contain_combo_price_pattern(text):
    return does_contain_pattern(text, COMBO_PRICE_PATTERN)

def does_contain_specied_price_pattern(text):
    return does_contain_pattern(text, SPECIFIED_PRICE_PATTERN)

def extract_specified_price(text, pattern=EXTRACT_SPECIFIED_PRICE_PATTERN):
    text = Class.strip_text(text)
    norm_text = Class.strip_and_remove_accents(text)
    m = re.search(pattern, norm_text)
    if m:
        found = m.groups()
        return list(filter(None, [Class.strip_text(text[text.find(found[0]):text.find(found[0]) + len(found[0])]), \
            Class.strip_text(text[text.find(found[2]):text.find(found[2]) + len(found[2])])]))
    return None 

def extract_pattern_info(text, pattern):
    text = Class.strip_text(text)
    norm_text = Class.strip_and_remove_accents(text)
    m = re.search(pattern, norm_text)
    if m:
        found = m.groups()
        return list(filter(None, [Class.strip_text(text[text.find(t):text.find(t) + len(t)]) for t in found]))
    return None

def extract_confusion_name_price(text):
    text = Class.strip_text(text)
    norm_text = Class.strip_and_remove_accents(text)
    for pos in range(len(norm_text)):
        if len(norm_text) - pos >= 15:
            continue
        if does_contain_pattern(norm_text[pos:], "^" + PRICE_PATTERN) \
        or does_contain_pattern(norm_text[pos:], "^" + PRICE_PATTERN_TYPE_2):
            return list(filter(None, [Class.strip_text(text[:pos]), Class.strip_text(text[pos:])]))
    return []

def find_point_between_2_point(pos, points):
    return [round(points[0][0] + (points[1][0] - points[0][0]) * pos),
            round(points[0][1] + (points[1][1] - points[0][1]) * pos)]

def split_bbox_by_info(text, info_lst, bbox):
    sub_bbox_lst = []
    for info in info_lst:
        start_pos = text.rfind(info)
        end_pos = start_pos + len(info)
        start_pos = start_pos / len(text)
        end_pos = end_pos / len(text)
        sub_bbox = []
        sub_bbox.append(find_point_between_2_point(start_pos, bbox[:2]))
        sub_bbox.append(find_point_between_2_point(end_pos, bbox[:2]))
        sub_bbox.append(find_point_between_2_point(end_pos, bbox[:1:-1]))
        sub_bbox.append(find_point_between_2_point(start_pos, bbox[:1:-1]))
        sub_bbox_lst.append(sub_bbox)
    return sub_bbox_lst

class Class(enum.Enum):
    no_interest = 0
    food_name = 1
    price = 2
    size = 3
    confusion_name_price = 4
    confusion_price_size = 5
    confusion_name_size = 6
    specified_price = 7
    price_pair = 8
    price_triplets = 9
    price_pair_type_pair = 10
    price_m_type1 = 11
    price_l_type1 = 12
    price_m_type2 = 13
    price_l_type2 = 14

    def __eq__(self, other):
        """Overrides the default implementation"""
        try:
            res = self.value == other.value
        except:
            res = False
        finally:
            return res

    def is_price_pair_type(cls):
        if cls == __class__.price_m_type1 \
            or cls == __class__.price_l_type1 \
            or cls == __class__.price_m_type2 \
            or cls == __class__.price_l_type2:
            return True
        return False

    def is_short_food_name(text):
        return text in SHORT_FOOD_NAME 
    
    def load_short_food_name():
        with open("./graph/data/short_food_name.txt", "r", encoding="utf-8") as f:
            name = [name.strip() for name in f.readlines()]
        return name

    def strip_text(text):
        return text.strip(" ,.+->=|*:").upper()

    def strip_and_remove_accents(text):
        text = __class__.strip_text(text)
        text = unidecode.unidecode(text)
        return text

    def normalize(text):
        text = __class__.strip_text(text)
        text = re.sub("[-]", " ", text)
        text = re.sub(" +", " ", text)
        text = unidecode.unidecode(text)
        return text

    def is_specied_price(text):
        return does_contain_specied_price_pattern(text)

    def is_confusion_or_pair_class(cls):
        if cls == __class__.confusion_name_price:
            return True
        # elif cls == __class__.confusion_price_size:
        #     return True
        # elif cls == __class__.confusion_name_size:
        #     return True
        elif cls == __class__.price_pair:
            return True
        elif cls == __class__.price_triplets:
            return True
        elif cls == __class__.price_pair_type_pair:
            return True
        return False

    def split_confusion_or_pair_class(cls, text, bbox):
        if cls == __class__.confusion_name_price:
            info_lst = extract_confusion_name_price(text)
            sub_bbox_lst = split_bbox_by_info(text.upper(), info_lst, bbox)
            return info_lst, sub_bbox_lst

        pattern = None
        # if cls == __class__.confusion_price_size:
        #     pattern = EXTRACT_CONFUSION_PRICE_SIZE_PATTERN
        # elif cls == __class__.confusion_name_size:
        #     pattern = EXTRACT_CONFUSION_NAME_SIZE_PATTERN
        # el
        if cls == __class__.price_pair:
            pattern = EXTRACT_PRICE_PAIR_PATTERN_TYPE_2
        elif cls == __class__.price_triplets:
            pattern = EXTRACT_PRICE_TRIPLETS_PATTERN_TYPE_2
        elif cls == __class__.price_pair_type_pair:
            pattern = EXTRACT_PRICE_PAIR_TYPE_PAIR_PATTERN_TYPE_2
        
        info_lst = []
        sub_bbox_lst = []
            
        if pattern:
            try:
                info_lst = extract_pattern_info(text, pattern)
                print("extract info", text, "---", info_lst)
                sub_bbox_lst = split_bbox_by_info(text.upper(), info_lst, bbox)
            except:
                info_lst = []
                sub_bbox_lst = []
                if cls == __class__.price_pair:
                    pattern = EXTRACT_PRICE_PAIR_PATTERN
                elif cls == __class__.price_triplets:
                    pattern = EXTRACT_PRICE_TRIPLETS_PATTERN
                elif cls == __class__.price_pair_type_pair:
                    pattern = EXTRACT_PRICE_PAIR_TYPE_PAIR_PATTERN
                try:
                    info_lst = extract_pattern_info(text, pattern)
                    sub_bbox_lst = split_bbox_by_info(text.upper(), info_lst, bbox)
                except:
                    return [], []

        return info_lst, sub_bbox_lst

    def classify_text(text):
        text = __class__.normalize(text)
        word_count = len(text.split())
        alphabet_count = sum(c.isalpha() for c in text)
        digit_count = sum(c.isdigit() for c in text)
        
        if "THEO THOI GIA" in text \
            or does_contain_pattern(text, DESCRIPTION_PATTERN):
            print("[LOGMODE] ignoring by DESCRIPTION_PATTERN", text)
            return __class__.no_interest
        elif text in SIZE_LIST:
            return __class__.size
        elif (does_contain_pattern(text, PRICE_PAIR_PATTERN) \
            or does_contain_pattern(text, PRICE_PAIR_PATTERN_TYPE_2)) \
            and not does_contain_pattern(text, DOUBLE_ZEROS_PATTERN):
            return __class__.price_pair
        elif (does_contain_pattern(text, PRICE_TRIPLETS_PATTERN) \
            or does_contain_pattern(text, PRICE_TRIPLETS_PATTERN_TYPE_2)) \
            and not does_contain_pattern(text, DOUBLE_ZEROS_PATTERN):
            return __class__.price_triplets
        elif (does_contain_pattern(text, PRICE_PAIR_TYPE_PAIR_PATTERN) \
            or does_contain_pattern(text, PRICE_PAIR_TYPE_PAIR_PATTERN_TYPE_2)) \
            and not does_contain_pattern(text, DOUBLE_ZEROS_PATTERN):
            return __class__.price_pair_type_pair
        elif does_contain_pattern(text, SIZE_PRICE_PATTERN):
            return __class__.confusion_price_size
        elif len(text) < 15 \
            and 5 * digit_count > alphabet_count \
            and (does_contain_combo_price_pattern(text) \
            or does_contain_price_pattern(text)):
            return __class__.price
        elif digit_count >= 1 and digit_count <= 6 \
            and word_count >= 2 and alphabet_count >= 6 \
            and does_contain_price_pattern(text) \
            and "PAGE" not in text:
            # process specied name
            if __class__.is_specied_price(text):
                return __class__.specified_price
            elif does_contain_pattern(text, YEAR_PATTERN):
                return __class__.food_name
            elif digit_count >= 4 \
                and (does_contain_pattern(text, SHORT_PRICE_PATTERN) \
                    or does_contain_pattern(text, SHORT_PRICE_PATTERN_TYPE_2) \
                    or (alphabet_count >= 20 and does_contain_pattern(text, SHORT_PRICE_PATTERN_TYPE_3))):
                return __class__.confusion_name_price
            else:
                return __class__.food_name
        elif alphabet_count <= 4 \
            and digit_count >= 4 \
            and (does_contain_pattern(text, SHORT_PRICE_PATTERN) \
                or does_contain_pattern(text, SHORT_PRICE_PATTERN_TYPE_2)):
            return __class__.price
        elif digit_count in [4, 8, 12] \
            and word_count >= 4 and alphabet_count >= 12 \
            and does_contain_pattern(text, YEAR_PATTERN):
            return __class__.food_name
        elif digit_count > 9 \
            or does_contain_pattern(text, TIME_PATTERN) \
            or len(text) >= 80:
            print("[LOGMODE] ignoring by TIME PATTERN", text)
            return __class__.no_interest
        elif digit_count <= 3 \
            and alphabet_count >= 2 \
            and (len(text) >= 4 or __class__.is_short_food_name(text)):
            return __class__.food_name
        else:
            print("[LOGMODE] not matching", text)
            return __class__.no_interest

SHORT_FOOD_NAME = Class.load_short_food_name()


if __name__ == "__main__":
    info = extract_confusion_name_price("phở bò 24k/ tô".upper())
    print(info)

    new_bboxs = split_bbox_by_info("Trứng cút lộn 14k/ 1chục".upper(), ['TRỨNG CÚT LỘN', '14K/ 1CHỤC'], [[347, 935], [1146, 935], [1143, 1016], [344, 1016]])
    print(new_bboxs)