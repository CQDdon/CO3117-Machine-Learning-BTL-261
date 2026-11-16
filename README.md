# BÀI TẬP LỚN MÔN HỌC MÁY (CO3117) - HK 241/261
## Đề tài: Ứng dụng SVM, Naive Bayes, Logistic Regression, và XGBoost vào bài toán phân loại review quần áo phụ nữ

Trường Đại học Bách Khoa - ĐHQG TP.HCM  
Khoa Khoa học và Kỹ thuật Máy tính  
**GVHD:** Th.S Võ Thanh Hùng  

---

## 📌 Giới thiệu đề tài
Dự án tập trung giải quyết bài toán phân loại quan điểm (Sentiment / Recommendation Classification) từ dữ liệu đánh giá sản phẩm thương mại điện tử dành cho phụ nữ ([Women's E-Commerce Clothing Reviews](https://www.kaggle.com/datasets/nicapotato/womens-ecommerce-clothing-reviews)).

Mục tiêu chính:
1. **Biểu diễn dữ liệu văn bản:** So sánh CountVectorizer (Bag-of-Words) và TfidfVectorizer.
2. **Lựa chọn đặc trưng (Feature Selection):** Khảo sát Chi-Square ($\chi^2$) và ANOVA F-Score với `SelectKBest`.
3. **Mô hình học máy áp dụng:**
   - Support Vector Machine (SVM) với các hàm Kernel (Linear, RBF, Poly, Sigmoid).
   - Hồi quy Logistic (Logistic Regression) xử lý mất cân bằng lớp.
   - Multinomial Naive Bayes.
   - Extreme Gradient Boosting (XGBoost) với histogram-based GPU acceleration.
4. **Kỹ thuật mở rộng nâng cao:**
   - Tinh chỉnh siêu tham số bằng `RandomizedSearchCV` kết hợp Stratified 5-Fold Cross-Validation.
   - Học kết hợp (Ensemble Learning): Bagging vs AdaBoost trên mô hình SVM.
   - Tích hợp siêu dữ liệu (XGBoost with Metadata: `Age`, `Division Name`, `Department Name`, `Class Name`).
5. **Đánh giá & Phân tích chuyên sâu:**
   - Xử lý dữ liệu mất cân bằng (82.2% Lớp 1 vs 17.8% Lớp 0) bằng các thước đo chuyên dụng: Macro-F1, F1 (Lớp 0), Recall (Lớp 0), ROC-AUC.
   - Phân tích Overfitting (Train vs Test Gap).
   - Phân tích các từ có sức ảnh hưởng lớn nhất (Feature Importance / Coefficients).
   - Phân tích lỗi (Error Analysis) trên các mẫu dự đoán sai với độ tự tin cao.

---

## 🚀 Hướng dẫn chạy trên Google Colab

### Cách 1: Mở trực tiếp từ GitHub trên Google Colab
1. Truy cập [Google Colab](https://colab.research.google.com/).
2. Chọn thẻ **GitHub** (File -> Open notebook -> GitHub).
3. Dán đường dẫn repository GitHub của bạn vào.
4. Chọn file notebook:
   - `ML_BTL_Unified_Womens_Clothing_Reviews.ipynb` (Notebook tổng hợp hoàn chỉnh).
   - Hoặc các file notebook thành phần: `Dang_WomensECommerceClothingReviews_SVM.ipynb`, `LogReg_NaiveBayes_XGBoost.ipynb`.
5. Đổi Runtime sang GPU nếu muốn tăng tốc XGBoost:
   - Menu **Runtime** -> **Change runtime type** -> Chọn **T4 GPU** -> **Save**.
6. Chạy lần lượt các cell (Notebook đã tích hợp sẵn lệnh tự động tải tập dữ liệu nếu chưa có).

---

## 📂 Cấu trúc Repository
```text
├── Dang_WomensECommerceClothingReviews_SVM.ipynb   # Notebook thí nghiệm SVM, Tokenization, Kernel & Ensemble
├── LogReg_NaiveBayes_XGBoost.ipynb                # Notebook thí nghiệm LogReg, Naive Bayes, XGBoost & Metadata
├── ML_BTL_Unified_Womens_Clothing_Reviews.ipynb     # Notebook TỔNG HỢP hoàn chỉnh chạy trên Google Colab
├── NHAP REPORT MACHINE LEARNING.pdf                # Báo cáo BTL (bản nháp cấu trúc)
├── ML-huong-dan-BTL.pdf                            # Đề cương & Hướng dẫn làm BTL CO3117
├── .gitignore                                      # File bỏ qua các thư mục tài liệu môn học không liên quan
└── README.md                                       # Tài liệu hướng dẫn đồ án
```
