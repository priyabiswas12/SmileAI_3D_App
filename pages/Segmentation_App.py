import streamlit as st
import os
from predict.inference_seg import Inference

# Get corresponding image path for detected teeth
def get_image_path_for_label(label, jaw_type):
    image_directory = f"./ui_images/{jaw_type}_eg"  # Folder for images
    return os.path.join(image_directory, f"{label}.jpg")

# Define tooth orders and mappings
tooth_order_lower_right = [16, 14, 13, 12, 11, 10, 9, 8]  # Right Lower
tooth_order_lower_left = [7, 6, 5, 4, 3, 2, 1, 15]  # Left Lower
mapping_fdi_lower = {16:48, 14:47, 13:46, 12:45, 11:44, 10:43, 9:42, 8:41, 
                     7:31, 6:32, 5:33, 4:34, 3:35, 2:36, 1:37, 15:38}

tooth_order_upper_right = [15, 1, 2, 3, 4, 5, 6, 7]  # Right Upper
tooth_order_upper_left = [8, 9, 10, 11, 12, 13, 14, 16]  # Left Upper
mapping_fdi_upper = {15:18, 1:17, 2:16, 3:15, 4:14, 5:13, 6:12, 7:11, 
                     8:21, 9:22, 10:23, 11:24, 12:25, 13:26, 14:27, 16:28}

# Streamlit Title
st.title("3D STL Model Processor")

# Upload Sections
uploaded_file_upper = st.file_uploader("Upload an Upper Jaw STL file", type=["stl", "ply"], key="file2")
uploaded_file_lower = st.file_uploader("Upload a Lower Jaw STL file", type=["stl", "ply"], key="file1")


detected_teeth_lower = []
detected_teeth_upper = []

# Process uploaded Lower Jaw STL file
if uploaded_file_lower:
    file_path = os.path.join("./predict/uploads/lower", uploaded_file_lower.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file_lower.getbuffer())
    st.success(f"Lower Jaw File uploaded: {uploaded_file_lower.name}")

    # Run inference and get detected lower teeth
    p = Inference()
    p.preprocess_mesh(uploaded_file_lower.name,jaw_type= "lower")
    detected_teeth_lower = p.predict_labels_lower(uploaded_file_lower.name)
    if os.path.exists(file_path):
        os.remove(file_path)
    else:
        st.warning(f"File '{file_path}' not found.")

# Process uploaded Upper Jaw STL file
if uploaded_file_upper:
    file_path = os.path.join("./predict/uploads/upper", uploaded_file_upper.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file_upper.getbuffer())
    st.success(f"Upper Jaw File uploaded: {uploaded_file_upper.name}")

    # Run inference and get detected upper teeth
    p = Inference()
    p.preprocess_mesh(uploaded_file_upper.name,jaw_type= "upper")
    detected_teeth_upper = p.predict_labels_upper(uploaded_file_upper.name)
    if os.path.exists(file_path):
        os.remove(file_path)
    else:
        st.warning(f"File '{file_path}' not found.")

# --- Layout Section ---
st.markdown("---")  # Separator

# ---- UPPER JAW ---- #
if uploaded_file_upper:
    st.subheader("Upper Jaw")
    with st.container():
        col1, col2, col3 = st.columns([4, 0.1, 4])  # Thin vertical column
        with col1:
            st.markdown("###### Right Upper")
            upper_right_cols = st.columns(len(tooth_order_upper_right))
            for i, tooth_id in enumerate(tooth_order_upper_right):
                fdi_number = mapping_fdi_upper[tooth_id]
                image_path = get_image_path_for_label(tooth_id, "upper")
                with upper_right_cols[i]:
                    if tooth_id in detected_teeth_upper and os.path.exists(image_path):
                        #st.image(image_path, caption=f"{fdi_number}", use_container_width=True)
                        st.image(image_path, caption=f"{fdi_number}")
                    else:
                        st.markdown(f'<div style="border:1px solid black; display:flex; align-items:center; justify-content:center; width:40px; height:50px; text-align:center;"><b>{fdi_number}</b></div>', unsafe_allow_html=True)

        with col2:  # **Thin Vertical Divider (Extended)**
            st.markdown('<div style="width:1px; height:180px; background:black; margin:auto;"></div>', unsafe_allow_html=True)

        with col3:
            st.markdown("###### Left Upper")
            upper_left_cols = st.columns(len(tooth_order_upper_left))
            for i, tooth_id in enumerate(tooth_order_upper_left):
                fdi_number = mapping_fdi_upper[tooth_id]
                image_path = get_image_path_for_label(tooth_id, "upper")
                with upper_left_cols[i]:
                    if tooth_id in detected_teeth_upper and os.path.exists(image_path):
                        #st.image(image_path, caption=f"{fdi_number}", use_container_width=True)
                        st.image(image_path, caption=f"{fdi_number}")
                    else:
                        st.markdown(f'<div style="border:1px solid black; display:flex; align-items:center; justify-content:center; width:40px; height:50px; text-align:center;"><b>{fdi_number}</b></div>', unsafe_allow_html=True)

st.markdown("---")  # Separator

# ---- LOWER JAW ---- #
if uploaded_file_lower:
    st.subheader("Lower Jaw")
    with st.container():
        col1, col2, col3 = st.columns([4, 0.1, 4])  # Small fixed-width column for divider
        with col1:
            st.markdown("###### Right Lower")
            lower_right_cols = st.columns(len(tooth_order_lower_right))
            for i, tooth_id in enumerate(tooth_order_lower_right):
                fdi_number = mapping_fdi_lower[tooth_id]
                image_path = get_image_path_for_label(tooth_id, "lower")
                with lower_right_cols[i]:
                    if tooth_id in detected_teeth_lower and os.path.exists(image_path):
                        #st.image(image_path, caption=f"{fdi_number}", use_container_width=True)
                        st.image(image_path, caption=f"{fdi_number}")
                    else:
                        st.markdown(f'<div style="border:1px solid black; display:flex; align-items:center; justify-content:center; width:40px; height:50px; text-align:center;"><b>{fdi_number}</b></div>', unsafe_allow_html=True)

        with col2:  # **Thin Vertical Divider (Extended)**
            st.markdown('<div style="width:1px; height:180px; background:black; margin:auto;"></div>', unsafe_allow_html=True)

        with col3:
            st.markdown("###### Left Lower")
            lower_left_cols = st.columns(len(tooth_order_lower_left))
            for i, tooth_id in enumerate(tooth_order_lower_left):
                fdi_number = mapping_fdi_lower[tooth_id]
                image_path = get_image_path_for_label(tooth_id, "lower")
                with lower_left_cols[i]:
                    if tooth_id in detected_teeth_lower and os.path.exists(image_path):
                        #st.image(image_path, caption=f"{fdi_number}", use_container_width=True)
                        st.image(image_path, caption=f"{fdi_number}")
                    else:
                        st.markdown(f'<div style="border:1px solid black; display:flex; align-items:center; justify-content:center; width:40px; height:50px; text-align:center;"><b>{fdi_number}</b></div>', unsafe_allow_html=True)
