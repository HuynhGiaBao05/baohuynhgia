import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import os

st.set_page_config(page_title="PCB Inventory Analysis", layout="wide")
st.title("📦 PCB Inventory — Phân tích & Dự báo")

# --- Cách 1: Nếu chạy trên máy cá nhân, file CSV cùng thư mục ---
CSV_FILENAME = "pcb_inventory_with_text.csv"

# --- Cách 2: Nếu trên hosting, dùng URL tải file --- 
# CSV_FILENAME = "https://raw.githubusercontent.com/username/repo/main/pcb_inventory_with_text.csv"

@st.cache_data(ttl=600)
def load_data(path_or_url):
    try:
        df = pd.read_csv(path_or_url)
        return df
    except Exception as e:
        st.error(f"Không tải được file CSV: {e}")
        return None

df = load_data(CSV_FILENAME)

if df is None:
    st.stop()

# Hiển thị 5 dòng đầu tiên
st.subheader("📌 5 dòng đầu tiên")
st.dataframe(df.head())

# Kiểu dữ liệu từng cột
st.subheader("🔍 Kiểu dữ liệu từng cột")
st.write(df.dtypes)

# Phân loại cột số và cột văn bản
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
text_cols = df.select_dtypes(include=['object']).columns.tolist()
st.write(f"🔢 Cột số: {numeric_cols}")
st.write(f"🔤 Cột văn bản: {text_cols}")

# Thống kê mô tả dữ liệu số
st.subheader("📈 Thống kê mô tả dữ liệu số")
st.write(df[numeric_cols].describe())

# Thống kê giá trị cột văn bản
for col in text_cols:
    st.subheader(f"📊 Thống kê giá trị cột '{col}'")
    st.write(df[col].value_counts())

# XỬ LÝ GIÁ TRỊ NULL
st.subheader("📌 Giá trị thiếu theo cột (trước xử lý)")
st.write(df.isnull().sum())

before_dropna = df.shape[0]
df = df.dropna()
after_dropna = df.shape[0]
st.success(f"✅ Đã xoá {before_dropna - after_dropna} dòng có giá trị thiếu")
st.write("📌 Giá trị thiếu sau xử lý")
st.write(df.isnull().sum())

# XỬ LÝ DÒNG TRÙNG LẶP
before_duplicates = df.shape[0]
df = df.drop_duplicates()
after_duplicates = df.shape[0]
st.success(f"✅ Đã xoá {before_duplicates - after_duplicates} dòng trùng lặp")

# Chuẩn hóa văn bản
if 'Shift' in df.columns:
    df['Shift'] = df['Shift'].astype(str).str.strip().str.title()
if 'Product_Type' in df.columns:
    df['Product_Type'] = df['Product_Type'].astype(str).str.strip().str.upper()

# Chuẩn hóa ngày tháng
if 'Week' in df.columns:
    df['Week'] = pd.to_datetime(df['Week'], errors='coerce')
    before_date_drop = df.shape[0]
    df = df.dropna(subset=['Week'])
    after_date_drop = df.shape[0]
    st.info(f"⏳ Đã loại {before_date_drop - after_date_drop} dòng do lỗi ngày tháng không parse được")
    df = df.set_index('Week')
else:
    st.warning("Không tìm thấy cột 'Week' để làm chỉ mục thời gian.")

# Kiểm tra kiểu dữ liệu cuối cùng
st.write("📅 Kiểu dữ liệu cuối cùng:")
st.write(df.dtypes)

st.subheader("✅ 5 dòng đầu sau chuẩn hóa")
st.dataframe(df.head())

# Tổng quan ca làm việc
st.subheader("🔤 Tần suất xuất hiện của từng ca làm việc")
if 'Shift' in df.columns:
    st.write(df['Shift'].value_counts())
else:
    st.info("Không có cột 'Shift'.")

# Tổng quan loại sản phẩm
st.subheader("🔤 Tần suất xuất hiện của từng loại sản phẩm")
if 'Product_Type' in df.columns:
    st.write(df['Product_Type'].value_counts())
else:
    st.info("Không có cột 'Product_Type'.")

# Thống kê tồn kho trung bình theo ca làm việc
if 'Shift' in df.columns and 'Inventory' in df.columns:
    st.subheader("📦 Trung bình tồn kho theo ca làm việc")
    st.write(df.groupby('Shift')['Inventory'].describe())
else:
    st.info("Thiếu cột 'Shift' hoặc 'Inventory' để thống kê.")

# Thống kê tồn kho trung bình theo loại sản phẩm
if 'Product_Type' in df.columns and 'Inventory' in df.columns:
    st.subheader("📦 Trung bình tồn kho theo loại sản phẩm")
    st.write(df.groupby('Product_Type')['Inventory'].describe())
else:
    st.info("Thiếu cột 'Product_Type' hoặc 'Inventory' để thống kê.")

# Một số thống kê tổng hợp theo ngày
if df.index.dtype.kind == 'M':  # Kiểm tra kiểu datetime index
    st.subheader("=== 1. 📦 Số lượng sản phẩm theo ngày (đếm số lần xuất hiện)")
    product_count_daily = df.groupby([df.index.date, 'Product_Type']).size().unstack(fill_value=0)
    st.dataframe(product_count_daily.tail(10))

    st.subheader("=== 2. 📥 Sản phẩm được thu thập theo thời gian (tổng số nhập)")
    if {'Import_Qty', 'Product_Type'}.issubset(df.columns):
        import_sum_by_product = df.groupby([df.index.date, 'Product_Type'])['Import_Qty'].sum().unstack(fill_value=0)
        st.dataframe(import_sum_by_product.tail(10))
    else:
        st.info("Thiếu cột 'Import_Qty' hoặc 'Product_Type' để thống kê.")

    st.subheader("=== 3. 🕒 10 dòng dữ liệu gần nhất")
    latest_10_rows = df.sort_index(ascending=False).head(10)
    st.dataframe(latest_10_rows)
else:
    st.info("Không có dữ liệu thời gian hợp lệ để thống kê theo ngày.")

# --- Vẽ biểu đồ ---
st.subheader("📊 Biểu đồ phân bố")

fig, axes = plt.subplots(1, 2, figsize=(12,4))

if 'Shift' in df.columns:
    sns.countplot(data=df.reset_index(), x='Shift', palette='Set2', ax=axes[0])
    axes[0].set_title("Phân bố theo ca sản xuất")
else:
    axes[0].text(0.5, 0.5, "Không có cột 'Shift'", ha='center', va='center')

if 'Product_Type' in df.columns:
    sns.countplot(data=df.reset_index(), x='Product_Type', palette='Set3', ax=axes[1])
    axes[1].set_title("Phân bố theo loại sản phẩm")
else:
    axes[1].text(0.5, 0.5, "Không có cột 'Product_Type'", ha='center', va='center')

plt.tight_layout()
st.pyplot(fig)

# Boxplot tồn kho theo ca sản xuất
if {'Shift', 'Inventory'}.issubset(df.columns):
    st.subheader("📦 Tồn kho theo ca sản xuất (Boxplot)")
    fig2, ax2 = plt.subplots(figsize=(6,5))
    sns.boxplot(data=df.reset_index(), x='Shift', y='Inventory', palette='coolwarm', ax=ax2)
    ax2.set_xlabel('Ca sản xuất')
    ax2.set_ylabel('Số lượng tồn kho')
    ax2.grid(True)
    st.pyplot(fig2)

# Biểu đồ thanh tổng hợp Import_Qty và Export_Qty theo loại sản phẩm
if {'Product_Type', 'Import_Qty', 'Export_Qty'}.issubset(df.columns):
    st.subheader("📊 Tổng Nhập - Xuất theo loại sản phẩm")
    grouped_sum = df.groupby('Product_Type')[['Import_Qty', 'Export_Qty']].sum().reset_index()
    fig3, ax3 = plt.subplots(figsize=(8,5))
    grouped_sum.plot(x='Product_Type', kind='bar', color=['skyblue', 'coral'], ax=ax3)
    ax3.set_ylabel('Số lượng')
    ax3.set_xticklabels(grouped_sum['Product_Type'], rotation=0)
    ax3.grid(axis='y')
    st.pyplot(fig3)

# Xu hướng tồn kho theo thời gian
if df.index.dtype.kind == 'M' and 'Inventory' in df.columns:
    st.subheader("📈 Xu hướng tồn kho PCB theo thời gian")
    fig4, ax4 = plt.subplots(figsize=(10,5))
    sns.lineplot(data=df, x=df.index, y='Inventory', marker='o', color='steelblue', ax=ax4)
    ax4.set_xlabel('Ngày')
    ax4.set_ylabel('Số lượng tồn kho')
    ax4.grid(True)
    st.pyplot(fig4)

# Ma trận tương quan
if {'Import_Qty', 'Export_Qty', 'Inventory'}.issubset(df.columns):
    st.subheader("📉 Ma trận tương quan giữa các biến số")
    fig5, ax5 = plt.subplots(figsize=(6,4))
    sns.heatmap(df[['Import_Qty', 'Export_Qty', 'Inventory']].corr(), annot=True, cmap='YlGnBu', fmt='.2f', ax=ax5)
    st.pyplot(fig5)

# --- Mô hình hồi quy tuyến tính ---
st.subheader("🤖 Dự báo tồn kho bằng Linear Regression")

# Chuẩn bị dữ liệu cho mô hình
if {'Inventory', 'Import_Qty', 'Export_Qty'}.issubset(df.columns):
    df_encoded = pd.get_dummies(df, columns=['Shift', 'Product_Type'])

    df_encoded = df_encoded.dropna(subset=['Import_Qty', 'Export_Qty', 'Inventory'])

    X = df_encoded.drop('Inventory', axis=1)
    y = df_encoded['Inventory']

    # Chỉ chọn cột số
    X = X.select_dtypes(include=[np.number])

    test_size = st.sidebar.slider("Tỉ lệ test cho mô hình", 0.05, 0.5, 0.2, 0.05)
    random_state = st.sidebar.number_input("Random state", value=42, step=1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    st.write(f"📉 RMSE: **{rmse:.4f}**")
    st.write(f"📈 R²: **{r2:.4f}**")

    compare_df = pd.DataFrame({'Thực tế': y_test.values, 'Dự báo': np.round(y_pred, 2)})
    st.dataframe(compare_df.head(10))

    if X_train.shape[1] <= 30:
        coef_df = pd.DataFrame({'Feature': X_train.columns, 'Coefficient': model.coef_}).sort_values(by='Coefficient', key=abs, ascending=False)
        st.subheader("Trọng số các đặc trưng (coefficients)")
        st.dataframe(coef_df)
else:
    st.info("Thiếu cột 'Inventory', 'Import_Qty' hoặc 'Export_Qty' để huấn luyện mô hình.")

# --- Kết thúc ---
st.sidebar.markdown("---")
st.sidebar.write("Chạy lệnh: `streamlit run streamlit_app.py`")
