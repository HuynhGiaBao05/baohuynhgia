# streamlit_app.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import os

st.set_page_config(page_title="PCB Inventory Analysis", layout="wide")
st.title("📦 PCB Inventory — Analysis & Prediction")

# ---------- Settings ----------
CSV_FILENAME = "pcb_inventory_with_text.csv"

# ---------- Load data ----------
@st.cache_data(ttl=300)
def load_data(path):
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path)
        return df
    except Exception as e:
        st.error(f"Không đọc được file CSV: {e}")
        return None

df = load_data(CSV_FILENAME)

if df is None:
    st.warning(f"Không tìm thấy file `{CSV_FILENAME}` trong thư mục làm việc. "
               "Hãy chắc chắn file đặt cùng thư mục với `streamlit_app.py`.")
    st.stop()

# ---------- Sidebar controls ----------
st.sidebar.header("Tùy chọn")
show_raw = st.sidebar.checkbox("Hiển thị dữ liệu thô (5 dòng)", value=True)
dropna_opt = st.sidebar.checkbox("Tự động drop các dòng có giá trị thiếu", value=True)
dropdup_opt = st.sidebar.checkbox("Tự động drop dòng trùng lặp", value=True)
test_size = st.sidebar.slider("Tỉ lệ test cho mô hình", 0.05, 0.5, 0.2, 0.05)
random_state = st.sidebar.number_input("random_state (mô hình)", value=42, step=1)

# ---------- Basic info / cleaning ----------
tab1, tab2, tab3, tab4 = st.tabs(["Tổng quan", "Tiền xử lý", "Biểu đồ", "Dự báo"])

with tab1:
    st.header("🔎 Tổng quan dữ liệu")
    st.write(f"**Source file:** `{CSV_FILENAME}` — kích thước ban đầu: {df.shape[0]} dòng × {df.shape[1]} cột")
    if show_raw:
        st.dataframe(df.head())

    st.subheader("🔍 Kiểu dữ liệu")
    st.write(df.dtypes)

    # detect numeric / text cols
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    text_cols = df.select_dtypes(include=['object']).columns.tolist()
    st.write("🔢 Cột số:", numeric_cols)
    st.write("🔤 Cột văn bản:", text_cols)

with tab2:
    st.header("🧹 Tiền xử lý và chuẩn hoá")
    df_clean = df.copy()

    # show null counts
    st.subheader("📌 Giá trị thiếu theo cột (trước xử lý)")
    st.write(df_clean.isnull().sum())

    # dropna option
    if dropna_opt:
        before = df_clean.shape[0]
        df_clean = df_clean.dropna()
        st.write(f"✅ Đã drop {before - df_clean.shape[0]} dòng có giá trị thiếu.")

    # drop duplicates
    if dropdup_opt:
        before = df_clean.shape[0]
        df_clean = df_clean.drop_duplicates()
        st.write(f"✅ Đã drop {before - df_clean.shape[0]} dòng trùng lặp.")

    # Normalize text columns if exist
    if 'Shift' in df_clean.columns:
        df_clean['Shift'] = df_clean['Shift'].astype(str).str.strip().str.title()
    if 'Product_Type' in df_clean.columns:
        df_clean['Product_Type'] = df_clean['Product_Type'].astype(str).str.strip().str.upper()

    # Parse Week column (tự động phát hiện tên 'Week' hoặc 'Date')
    date_col = None
    for candidate in ['Week', 'Date', 'Datetime', 'Time']:
        if candidate in df_clean.columns:
            date_col = candidate
            break

    if date_col:
        st.write(f"⏱️ Dùng cột thời gian: `{date_col}` — chuyển sang datetime")
        df_clean[date_col] = pd.to_datetime(df_clean[date_col], errors='coerce')
        before = df_clean.shape[0]
        df_clean = df_clean.dropna(subset=[date_col])
        st.write(f"✅ Đã loại {before - df_clean.shape[0]} dòng do lỗi ngày tháng (không parse được).")
        df_clean = df_clean.set_index(date_col)
    else:
        st.info("Không tìm thấy cột thời gian tiêu chuẩn ('Week', 'Date', 'Datetime', 'Time'). Nếu có cột thời gian khác, đổi tên thành 'Week' hoặc 'Date' để kích hoạt phân tích theo thời gian.")

    st.subheader("📌 Kiểu dữ liệu sau tiền xử lý")
    st.write(df_clean.dtypes)
    st.session_state['df_clean'] = df_clean  # lưu tạm để tab khác dùng
    st.success("Tiền xử lý hoàn tất — chuyển sang tab 'Biểu đồ' hoặc 'Dự báo'.")

with tab3:
    st.header("📊 Visualizations")
    df_vis = st.session_state.get('df_clean', df.copy())

    # show counts
    st.subheader("🔤 Tần suất ca / loại sản phẩm")
    cols = st.columns(2)
    if 'Shift' in df_vis.columns:
        fig1, ax1 = plt.subplots(figsize=(6, 3))
        sns.countplot(data=df_vis.reset_index(), x='Shift', order=df_vis['Shift'].value_counts().index, ax=ax1)
        ax1.set_title("Phân bố theo Shift")
        cols[0].pyplot(fig1)
    else:
        cols[0].info("Không có cột 'Shift' để vẽ.")

    if 'Product_Type' in df_vis.columns:
        fig2, ax2 = plt.subplots(figsize=(6, 3))
        sns.countplot(data=df_vis.reset_index(), x='Product_Type', order=df_vis['Product_Type'].value_counts().index, ax=ax2)
        ax2.set_title("Phân bố theo Product_Type")
        cols[1].pyplot(fig2)
    else:
        cols[1].info("Không có cột 'Product_Type' để vẽ.")

    # boxplot inventory by shift
    if 'Inventory' in df_vis.columns and 'Shift' in df_vis.columns:
        st.subheader("📦 Tồn kho theo ca (Boxplot)")
        fig3, ax3 = plt.subplots(figsize=(8, 4))
        sns.boxplot(data=df_vis.reset_index(), x='Shift', y='Inventory', ax=ax3)
        ax3.set_xlabel('Shift')
        ax3.set_ylabel('Inventory')
        st.pyplot(fig3)

    # stacked bar Import/Export by product type
    if {'Import_Qty', 'Export_Qty', 'Product_Type'}.issubset(df_vis.columns):
        st.subheader("📥/📤 Tổng Nhập - Xuất theo loại sản phẩm")
        grouped = df_vis.groupby('Product_Type')[['Import_Qty', 'Export_Qty']].sum().sort_values('Import_Qty', ascending=False)
        st.dataframe(grouped)
        fig4 = grouped.plot(kind='bar', figsize=(8,4)).get_figure()
        st.pyplot(fig4)

    # time series inventory
    if df_vis.index.dtype.kind == 'M' and 'Inventory' in df_vis.columns:
        st.subheader("📈 Xu hướng tồn kho theo thời gian")
        fig5, ax5 = plt.subplots(figsize=(10,4))
        sns.lineplot(x=df_vis.index, y='Inventory', data=df_vis, marker='o', ax=ax5)
        ax5.set_xlabel("Date")
        ax5.set_ylabel("Inventory")
        st.pyplot(fig5)
    else:
        st.info("Chưa có dữ liệu thời gian hợp lệ hoặc không có cột 'Inventory'.")

with tab4:
    st.header("🤖 Dự báo — Linear Regression")
    df_model = st.session_state.get('df_clean', df.copy())

    # check required columns
    req_cols = {'Inventory', 'Import_Qty', 'Export_Qty'}
    if not req_cols.issubset(df_model.columns):
        st.warning(f"Thiếu một trong các cột cần thiết cho mô hình: {req_cols}. Không thể huấn luyện.")
    else:
        # one-hot encode categorical if any
        cat_cols = [c for c in ['Shift', 'Product_Type'] if c in df_model.columns]
        df_encoded = pd.get_dummies(df_model.reset_index(), columns=cat_cols, drop_first=True)
        # ensure numeric only for X
        X = df_encoded.drop(columns=['Inventory'])
        y = df_encoded['Inventory']

        # align shapes (drop non-numeric columns if remain)
        X = X.select_dtypes(include=[np.number])

        # split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=int(random_state))

        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)

        st.subheader("Kết quả đánh giá mô hình")
        st.write(f"📉 RMSE: **{rmse:.4f}**")
        st.write(f"📈 R²: **{r2:.4f}**")

        # show a small comparison table
        compare_df = pd.DataFrame({
            'y_true': y_test.values,
            'y_pred': np.round(y_pred, 2)
        }).reset_index(drop=True)
        st.subheader("So sánh một số giá trị thực vs dự báo")
        st.dataframe(compare_df.head(10))

        # Optional: show feature coefficients if features small
        if X_train.shape[1] <= 30:
            coef_df = pd.DataFrame({
                'feature': X_train.columns,
                'coefficient': model.coef_
            }).sort_values(by='coefficient', key=abs, ascending=False)
            st.subheader("Trọng số (coefficients) của các input")
            st.dataframe(coef_df)

st.sidebar.markdown("---")
st.sidebar.write("Chạy: `streamlit run streamlit_app.py`")
