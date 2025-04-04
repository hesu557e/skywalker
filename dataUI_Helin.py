import streamlit as st 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from io import StringIO
import zipfile
import shutil

st.set_page_config(page_title="Data Preprocessing UI", layout="wide")
st.title("Data Collection Platform")

uploaded_file = st.file_uploader("Upload .txt file", type="txt")

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
        "Raw",
        "(R - R0) / R0 * 100 (Relative %)",
        "(R - R0) / R0 (Relative Ratio)",
        "R - R0 (Absolute Difference)",
        "L2 Normalization",
        "(R0 - R) / R"
    ])

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

st.subheader("Plotting Option")
only_after_start = st.checkbox("Only show data after start time?")

st.subheader("Output Settings")
save_option = st.radio("Save as:", ["Sensor-wise", "Window-wise"])
filename_prefix = st.text_input("Filename prefix (optional)", value="")

output_mode = st.radio("Output Mode:", ["Save to directory", "Download as ZIP"])
if output_mode == "Save to directory":
    output_dir = st.text_input("Storage location", value="output_data")
else:
    output_dir = "temp_output"

postprocess_option = st.radio("Post-process each response value before saving:", ["None", "Divide by 10", "Multiply by 250"])
output_dir = output_dir.strip().strip('"').strip("'")

if uploaded_file and st.button("Process & Visualize"):
    stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
    raw_data = pd.read_csv(stringio, header=None, skiprows=3)
    raw_data = raw_data[0].str.split(' ', expand=True)
    raw_data = raw_data.iloc[:-3, :-1]
    raw_data.columns = ['Time'] + [f"Channel_{i}" for i in range(1, 17)]

    raw_data['Time'] = pd.to_numeric(raw_data['Time'], errors='coerce') / 1000
    raw_data.dropna(subset=['Time'], inplace=True)
    t0 = raw_data['Time'].iloc[0]
    raw_data['Time'] = raw_data['Time'] - t0

    for col in raw_data.columns[1:]:
        raw_data[col] = pd.to_numeric(raw_data[col], errors='coerce')
    raw_data.dropna(inplace=True)

    os.makedirs(output_dir, exist_ok=True)

    st.subheader("Selected Channel Overview")

    if save_option == "Sensor-wise":
        processed_data = raw_data[raw_data['Time'] >= start_time].copy()
        if response_type not in ["Raw", "L2 Normalization", "(R0 - R) / R"]:
            baseline_idx = (processed_data['Time'] - start_time).abs().idxmin()
            baseline_values = processed_data.loc[baseline_idx, selected_channels]
            for col in selected_channels:
                if response_type.startswith("(R - R0) / R0 * 100"):
                    processed_data[col] = ((processed_data[col] - baseline_values[col]) / baseline_values[col]) * 100
                elif response_type.startswith("(R - R0) / R0"):
                    processed_data[col] = ((processed_data[col] - baseline_values[col]) / baseline_values[col])
                elif response_type.startswith("R - R0"):
                    processed_data[col] = (processed_data[col] - baseline_values[col])

        if response_type == "L2 Normalization":
            for col in selected_channels:
                norm = np.linalg.norm(processed_data[col])
                processed_data[col] = processed_data[col] / norm if norm != 0 else processed_data[col]
        elif response_type == "(R0 - R) / R":
            baseline_idx = (processed_data['Time'] - start_time).abs().idxmin()
            baseline_values = processed_data.loc[baseline_idx, selected_channels]
            for col in selected_channels:
                processed_data[col] = (baseline_values[col] - processed_data[col]) / processed_data[col]

        if postprocess_option == "Divide by 10":
            for col in selected_channels:
                processed_data[col] /= 10
        elif postprocess_option == "Multiply by 250":
            for col in selected_channels:
                processed_data[col] *= 250

        plot_data = processed_data[processed_data['Time'] > start_time] if only_after_start else processed_data

        n = len(selected_channels)
        rows = (n + 3) // 4
        fig, axes = plt.subplots(rows, 4, figsize=(16, 4 * rows), constrained_layout=True)
        axes = axes.flatten()
        for i, col in enumerate(selected_channels):
            ax = axes[i]
            ax.plot(plot_data['Time'], plot_data[col])
            ax.set_title(col)
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Processed Response")
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])
        st.pyplot(fig)

        for col in selected_channels:
            df = processed_data[['Time', col]]
            df.to_csv(os.path.join(output_dir, f"{filename_prefix}{col}_response.csv"), index=False)

    elif save_option == "Window-wise":
        window_length = exposure_time + flushing_time
        norm_fig, norm_axes = plt.subplots(4, 4, figsize=(20, 16), constrained_layout=True)
        norm_axes = norm_axes.flatten()
        for idx, col in enumerate(selected_channels):
            if idx >= len(norm_axes):
                continue
            ax = norm_axes[idx]
            for cycle in range(int(num_cycles)):
                start = start_time + cycle * window_length
                end = start + window_length
                window_df = raw_data[(raw_data['Time'] >= start) & (raw_data['Time'] < end)].copy()
                if window_df.empty or col not in window_df.columns:
                    continue

                baseline_idx = (window_df['Time'] - start).abs().idxmin()
                baseline_val = window_df.loc[baseline_idx, col]

                if response_type not in ["Raw", "L2 Normalization", "(R0 - R) / R"]:
                    if response_type.startswith("(R - R0) / R0 * 100"):
                        window_df[col] = ((window_df[col] - baseline_val) / baseline_val) * 100
                    elif response_type.startswith("(R - R0) / R0"):
                        window_df[col] = ((window_df[col] - baseline_val) / baseline_val)
                    elif response_type.startswith("R - R0"):
                        window_df[col] = window_df[col] - baseline_val

                if response_type == "L2 Normalization":
                    norm = np.linalg.norm(window_df[col])
                    window_df[col] = window_df[col] / norm if norm != 0 else window_df[col]
                elif response_type == "(R0 - R) / R":
                    window_df[col] = (baseline_val - window_df[col]) / window_df[col]

                if postprocess_option == "Divide by 10":
                    window_df[col] /= 10
                elif postprocess_option == "Multiply by 250":
                    window_df[col] *= 250

                time = window_df['Time'].values - start
                response = window_df[col].values
                ax.plot(time, response, alpha=0.7)

                window_df['Time'] = window_df['Time'] - start
                sliced = window_df[['Time', col]]
                sliced.to_csv(
                    os.path.join(output_dir, f"{filename_prefix}{col}_Cycle{cycle+1}.csv"), index=False
                )
            ax.set_title(col)
            ax.set_xlabel("Time (s)")
            ax.set_xlim(0, exposure_time + flushing_time)
            ax.set_ylabel("Processed Response")
        for j in range(len(selected_channels), len(norm_axes)):
            norm_fig.delaxes(norm_axes[j])
        st.pyplot(norm_fig)

    st.success("Processing and saving complete!")

    if output_mode == "Download as ZIP":
        zip_filename = f"{filename_prefix}processed_data.zip"
        zip_path = os.path.join(".", zip_filename)
        with zipfile.ZipFile(zip_path, "w") as zipf:
            for root, _, files in os.walk(output_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    zipf.write(file_path, arcname=file)
        with open(zip_path, "rb") as f:
            st.download_button(
                label="Download ZIP file",
                data=f,
                file_name=zip_filename,
                mime="application/zip"
            )
        shutil.rmtree(output_dir)
        os.remove(zip_path)
