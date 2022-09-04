# Quy Nhon AI Hackathon Challenge Smart Menu

mô hình detection và recognition được huấn luyện bằng [PPOCRv3](https://github.com/PaddlePaddle/PaddleOCR) với 1 vài thay đổi để đạt kết quả tốt cho bài toán, dữ liệu trong bài được sử dụng 100% là dữ liệu synthtext với corpus là chủ đề về ẩm thực. Ngoài ra để sửa các lỗi chính tả (chủ yếu là dấu câu) team còn sử dụng [SymSpell](https://github.com/wolfgarbe/SymSpell)

## DEMO:

### input:
<p align="center">
    <img src="images/098.jpeg">
</p>

### output:

```json
{
    "image_name": "098.jpeg",
    "infers": [
        {
            "food_name_en": "BLACK SUGAR BUBBLE MILK TEA L",
            "food_name_vi": "SỮA TƯƠI TRÂN CHÂU ĐƯỜNG ĐEN L",
            "food_price": "34000"
        },
        {
            "food_name_en": "TARO CUSTARD CHEESE MILK TEA L",
            "food_name_vi": "TRÀ SỮA KHOAI MÔN KEM CHEESE L",
            "food_price": "34000"
        },
        {
            "food_name_en": "",
            "food_name_vi": "TRÀ SỮA MATCHAR KEM CHEESE L",
            "food_price": "34000"
        },
        {
            "food_name_en": "",
            "food_name_vi": "TRÀ SỮA KHOAI LAN TÍM SỮA DỪA L",
            "food_price": "34000"
        },
        {
            "food_name_en": "MELON MILK TEA L",
            "food_name_vi": "TRÀ SỮA DƯA GANG L",
            "food_price": "34000"
        },
        {
            "food_name_en": "DURIAN MILK TEA L",
            "food_name_vi": "TRÀ SỮA SẦU RIÊNG L",
            "food_price": "34000"
        },
        {
            "food_name_en": "CHOCOLATE PEARL MILK TEA L",
            "food_name_vi": "TRÀ SỮA SOCOLA NGỌC TRAI L",
            "food_price": "30000"
        },
        {
            "food_name_en": "ROYAL BUBBLE MILK TEA L",
            "food_name_vi": "TRÀ SỮA TRÂN CHÂU HOÀNG GIA L",
            "food_price": "28000"
        },
        {
            "food_name_en": "OOLONG WATERMELON MILK TEA L",
            "food_name_vi": "TRÀ SỮA Ô LONG DƯA HẤU L",
            "food_price": "28000"
        },
        {
            "food_name_en": "OOLONG MINT MILK TEA L",
            "food_name_vi": "TRÀ SỮA Ô LONG BẠC HÀ L",
            "food_price": "28000"
        },
        {
            "food_name_en": "PEACH FLAVOUR MILK TEA L",
            "food_name_vi": "TRÀ SỮA HƯƠNG ĐÀO L",
            "food_price": "28000"
        },
        {
            "food_name_en": "BLACK SUGAR BUBBLE MILK TEA M",
            "food_name_vi": "SỮA TƯƠI TRÂN CHÂU ĐƯỜNG ĐEN M",
            "food_price": "29000"
        },
        {
            "food_name_en": "TARO CUSTARD CHEESE MILK TEA M",
            "food_name_vi": "TRÀ SỮA KHOAI MÔN KEM CHEESE M",
            "food_price": "29000"
        },
        {
            "food_name_en": "",
            "food_name_vi": "TRÀ SỮA MATCHAR KEM CHEESE M",
            "food_price": "29000"
        },
        {
            "food_name_en": "",
            "food_name_vi": "TRÀ SỮA KHOAI LAN TÍM SỮA DỪA M",
            "food_price": "29000"
        },
        {
            "food_name_en": "MELON MILK TEA M",
            "food_name_vi": "TRÀ SỮA DƯA GANG M",
            "food_price": "29000"
        },
        {
            "food_name_en": "DURIAN MILK TEA M",
            "food_name_vi": "TRÀ SỮA SẦU RIÊNG M",
            "food_price": "29000"
        },
        {
            "food_name_en": "CHOCOLATE PEARL MILK TEA M",
            "food_name_vi": "TRÀ SỮA SOCOLA NGỌC TRAI M",
            "food_price": "25000"
        },
        {
            "food_name_en": "ROYAL BUBBLE MILK TEA M",
            "food_name_vi": "TRÀ SỮA TRÂN CHÂU HOÀNG GIA M",
            "food_price": "23000"
        },
        {
            "food_name_en": "OOLONG WATERMELON MILK TEA M",
            "food_name_vi": "TRÀ SỮA Ô LONG DƯA HẤU M",
            "food_price": "23000"
        },
        {
            "food_name_en": "OOLONG MINT MILK TEA M",
            "food_name_vi": "TRÀ SỮA Ô LONG BẠC HÀ M",
            "food_price": "23000"
        },
        {
            "food_name_en": "PEACH FLAVOUR MILK TEA M",
            "food_name_vi": "TRÀ SỮA HƯƠNG ĐÀO M",
            "food_price": "23000"
        }
    ]
}
```
## RUN API
```
python api.py
```

## TEST API

```
python test_api.py
```

## Team:
   [Võ Văn Phúc](https://github.com/vovanphuc)

   [Nguyễn Văn Thiều](https://github.com/theluckygod)