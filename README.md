

## Mô tả dự án
Hệ thống phân loại phản hồi của LLM tiếng Việt thành 3 nhãn: no, intrinsic, extrinsic dựa trên context, prompt và response.

## Cấu trúc dự án
- src/preprocess.py: Tiền xử lý dữ liệu
- src/model.py: Định nghĩa mô hình PhoBERTClassifier
- src/train.py: Huấn luyện mô hình
- src/predict.py: Dự đoán nhãn cho dữ liệu mới
- src/utils.py: Hàm hỗ trợ
- requirements.txt: Thư viện cần thiết

## Hướng dẫn sử dụng
1. Cài đặt thư viện:
```bash
pip install -r requirements.txt
```
2. Chuẩn bị dữ liệu CSV với các cột: context, prompt, response, label
3. Huấn luyện mô hình:
```bash
python src/train.py <duong_dan_file_csv_train>
```
4. Dự đoán trên dữ liệu mới:
```bash
python src/predict.py <duong_dan_file_csv_test>
```
Kết quả sẽ lưu vào predictions.csv

## Lưu ý
- Chỉ sử dụng đúng bộ dữ liệu được cung cấp
- Không dùng API thương mại hoặc dữ liệu ngoài
- Mô hình sử dụng PhoBERT-base

## Đánh giá
- Macro-F1 và Accuracy trên 3 nhãn: no, intrinsic, extrinsic

## II. Evaluation
Mục tiêu đánh giá: Đo lường khả năng phân loại loại hallucination trong phản hồi của mô hình ở 3 nhãn: no, intrinsic, extrinsic.
Độ đo chính (xếp hạng): Macro-F1 trên ba nhãn (no, intrinsic, extrinsic). Macro-F1 đảm bảo mỗi lớp (label) được cân nhắc ngang nhau.
Độ đo phụ (dùng để giải quyết hòa điểm): Accuracy. Nếu hai đội có cùng Macro-F1 (đến 4 chữ số thập phân), đội có Accuracy cao hơn sẽ xếp trên.
Với mỗi lớp c ∈ {no, intrinsic, extrinsic} ta định nghĩa từ ma trận nhầm lẫn:
TP_c — true positives của lớp c
FP_c — false positives của lớp c
FN_c — false negatives của lớp c
Công thức
Đối với mỗi lớp c:


Macro-F1 (độ đo chính) là trung bình đơn của F1 theo từng lớp:


Accuracy (độ đo phụ — dùng để break-tie):


trong đó ∣D∣ là tổng số mẫu trong tập kiểm thử (test set).


## III. Terms
3.2. Mô hình ngôn ngữ & mô hình phụ trợ
Đội thi được phép sử dụng mô hình tiền huấn luyện (pre-trained) mã nguồn mở, bao gồm cả LLM, encoder-only, retriever, hoặc mô hình chuyên biệt khác.
Điều kiện:
Mô hình phải là public & reproducible (có checkpoint chính thức, commit/version rõ ràng, được công bố công khai).
Không sử dụng mô hình thương mại hoặc truy cập qua API đóng (GPT-4o, Gemini, Claude, v.v.).
Không fine-tune mô hình trên dữ liệu ngoài bộ chính thức.
Giới hạn với LLM:
Chỉ được sử dụng các mô hình LLM mã nguồn mở ≤ 7B parameters.
Các mô hình lớn hơn (13B, 70B, …) không được phép để đảm bảo công bằng tài nguyên tính toán.
Encoder-only (BERT, RoBERTa, PhoBERT, XLM-R, v.v.) hoặc retriever khác không giới hạn kích thước.



PhoBERT-base đã có sẵn file , hãy sử dung lại nó thông qua link path (E:\DAS2025\duan2\phobert-base) dẫn trong dự án


pip install -r requirements.txt
# train
python src/train.py data/vihallu-warmup.csv

# dự đoán
python src/predict.py data/vihallu-warmup.csv

# đánh giá kết quả dự đoán 
python src/evaluate.py predictions.csv

# xuất file csv chứa 2 column id, pred_label
python src/export_pred_label.py predictions.csv predictions2colum.csv
