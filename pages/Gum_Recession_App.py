# gum_recession_app.py

import streamlit as st
import os
import tempfile
from vedo import load, screenshot, show
from predict.classify_recession import (
    load_model,
    predict_gum_recession,
    extract_teeth_with_gums,
    TOOTH_NAMES
)

st.set_page_config(page_title="Gum Recession Classifier", layout="wide")
st.title("🦷 Gum Recession Classifier")

st.markdown("This tool classifies **gum recession presence** on the six lower incisors (teeth 5–10) using a trained neural network.")

uploaded_name = st.text_input("Specify base filename (e.g., 1 LowerJawScan)")

if uploaded_name:
    base_name = uploaded_name.strip()
    vtp_path = os.path.join("predict/uploads/lower", base_name + ".vtp")
    ply_path = os.path.join("predict/uploads/lower/processed", base_name + ".ply")

    if not os.path.exists(vtp_path) or not os.path.exists(ply_path):
        st.error("❌ VTP or PLY file not found. Check the filename and location.")
    else:
        print(f"Using: `{vtp_path}` and `{ply_path}`")

        # Display mesh preview screenshot

    import open3d as o3d
    import tempfile
    import time
    import open3d as o3d

    st.subheader("🦷 Jaw Mesh Preview")
    col1, col2 = st.columns([2, 3])

    with col1:
        # Generate temp path for screenshot
        #tmp_img_path = os.path.join(tempfile.gettempdir(), "jaw_preview.png")
        img_folder = "predict/uploads/ply_screenshots_wholejaw"
        os.makedirs(img_folder, exist_ok=True)
        tmp_img_path = os.path.join(img_folder, "jaw_preview.png")

        # Load and prepare mesh
        mesh = o3d.io.read_triangle_mesh(ply_path)
        mesh.compute_vertex_normals()

        # Create Open3D offscreen visualizer
        vis = o3d.visualization.Visualizer()
        vis.create_window(visible=False, width=1024, height=768)
        vis.add_geometry(mesh)
        vis.poll_events()
        vis.update_renderer()

        # Let it settle before taking screenshot
        time.sleep(0.5)
        vis.capture_screen_image(tmp_img_path)
        vis.destroy_window()

        # Show status + screenshot
        print(f"📷 Saved screenshot at: {tmp_img_path}")
        if os.path.exists(tmp_img_path):
            st.image(tmp_img_path, caption="Full Lower Jaw Mesh", use_column_width=True)
        else:
            st.warning("⚠️ Screenshot could not be generated.")



    with col2:
        debug_folder = "predict/uploads/lower/debug"
        meshes = extract_teeth_with_gums(vtp_path, ply_path, save_to=debug_folder)

        model = load_model()
        results = predict_gum_recession(meshes, model)

        st.subheader("🧪 Results")
        for label, result in results.items():
            name = TOOTH_NAMES.get(label, f"Tooth {label}")
            st.write(f"{name}: {result}")


else:
    st.info("Please enter a base filename to continue.")





