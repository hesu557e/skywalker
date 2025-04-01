import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from io import StringIO

st.set_page_config(page_title="Data Preprocessing UI", layout="wide")
st.title("Data Collection Platform")

# 上传原始txt文件
uploaded_file = st.file_uploader("Upload .txt file", type="txt")

# 参数输入区
st.subheader("Parameter Settings")
col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    start_time = st.number_input("Starting time (s)", min_value=0, value=600)
with col2:
    exposure_time = st.number_input("Exposure time (s)", min_value=0, value=60)
with col3:
    flushing_time = st.number_input("Flushing time (s)", min_value=0, value=120)
with col4:
    num_cycles = st.number_input("Number of cycles", min_value=1, value=10, step=1)
with col5:
    response_type = st.selectbox("Response Calculation Options", [
        "(R - R0) / R0 * 100 (Relative %)",
        "(R - R0) / R0 (Relative Ratio)",
        "R - R0 (Absolute Difference)"
    ])

# 显示通道图前的选项
st.subheader("Channel Selection")
channel_input = st.text_input("Channel Selection: All or please enter channel numbers separated by commas", value="All")
if channel_input.lower() == "all":
    selected_channels = [f"Channel_{i}" for i in range(1, 17)]
else:
    try:
        selected_indices = [int(x.strip()) for x in channel_input.split(',') if x.strip().isdigit() and 1 <= int(x.strip()) <= 16]
        selected_channels = [f"Channel_{i}" for i in selected_indices]
    except:
        st.error("Invalid channel input. Please enter numbers like: 1, 2, 5")
        st.stop()

# 显示是否只看 start_time 之后的图
st.subheader("Plotting Option")
only_after_start = st.checkbox("Only show data after start time?")

# 保存选项
st.subheader("Output Settings")
save_option = st.radio("Save as:", ["Sensor-wise", "Window-wise"])
filename_prefix = st.text_input("Filename prefix (optional)", value="")
output_dir = st.text_input("Storage location", value="output_data")
postprocess_option = st.radio("Post-process each response value before saving:", ["None", "Divide by 10", "Multiply by 250"])

# 清理路径中的非法字符
output_dir = output_dir.strip().strip('"').strip("'")

# 主处理流程
if uploaded_file and st.button("Process & Visualize"):
    # 读取数据
    stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
    raw_data = pd.read_csv(stringio, header=None, skiprows=3)

    # 拆分数据
    raw_data = raw_data[0].str.split(' ', expand=True)
    raw_data = raw_data.iloc[:-3, :-1]
    raw_data.columns = ['Time'] + [f"Channel_{i}" for i in range(1, 17)]

    # 转换数据类型
    raw_data['Time'] = pd.to_numeric(raw_data['Time'], errors='coerce') / 1000
    for col in raw_data.columns[1:]:
        raw_data[col] = pd.to_numeric(raw_data[col], errors='coerce')
    raw_data.dropna(inplace=True)

    # 根据是否勾选只显示 start_time 之后数据
    plot_data = raw_data[raw_data['Time'] > start_time] if only_after_start else raw_data

    # 显示通道图
    st.subheader("Selected Channel Overview")
    n = len(selected_channels)
    rows = (n + 3) // 4
    fig, axes = plt.subplots(rows, 4, figsize=(16, 4 * rows), constrained_layout=True)
    axes = axes.flatten()
    for i, col in enumerate(selected_channels):
        ax = axes[i]
        ax.plot(plot_data['Time'], plot_data[col])
        ax.set_title(col)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Response")
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
    st.pyplot(fig)

    # 创建保存文件夹
    os.makedirs(output_dir, exist_ok=True)

    # 保存数据
    if save_option == "Sensor-wise":
        baseline_row = raw_data[np.isclose(raw_data['Time'], start_time, atol=1)]
        if baseline_row.empty:
            st.error("Start time not found in data!")
            st.stop()

        baseline_values = baseline_row.iloc[0][selected_channels]
        response_data = raw_data[raw_data['Time'] > start_time].copy()

        for col in selected_channels:
            if response_type.startswith("(R - R0) / R0 * 100"):
                response_data[col] = ((response_data[col] - baseline_values[col]) / baseline_values[col]) * 100
            elif response_type.startswith("(R - R0) / R0"):
                response_data[col] = ((response_data[col] - baseline_values[col]) / baseline_values[col])
            else:
                response_data[col] = (response_data[col] - baseline_values[col])

            if postprocess_option == "Divide by 10":
                response_data[col] /= 10
            elif postprocess_option == "Multiply by 250":
                response_data[col] *= 250

            df = response_data[['Time', col]]
            df.to_csv(os.path.join(output_dir, f"{filename_prefix}{col}_response.csv"), index=False)

    else:
        window_length = exposure_time + flushing_time
        for cycle in range(int(num_cycles)):
            start = start_time + cycle * window_length
            end = start + window_length
            window_df = raw_data[(raw_data['Time'] >= start) & (raw_data['Time'] < end)].copy()

            baseline_row = window_df[np.isclose(window_df['Time'], start, atol=1)]
            if baseline_row.empty:
                continue
            baseline_values = baseline_row.iloc[0][selected_channels]

            for col in selected_channels:
                if response_type.startswith("(R - R0) / R0 * 100"):
                    window_df[col] = ((window_df[col] - baseline_values[col]) / baseline_values[col]) * 100
                elif response_type.startswith("(R - R0) / R0"):
                    window_df[col] = ((window_df[col] - baseline_values[col]) / baseline_values[col])
                else:
                    window_df[col] = (window_df[col] - baseline_values[col])

                if postprocess_option == "Divide by 10":
                    window_df[col] /= 10
                elif postprocess_option == "Multiply by 250":
                    window_df[col] *= 250

                sliced = window_df[['Time', col]]
                sliced.to_csv(
                    os.path.join(output_dir, f"{filename_prefix}{col}_Cycle{cycle+1}.csv"), index=False
                )

    st.success("Processing and saving complete!")
