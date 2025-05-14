# classify_recession.py (inside predict/utils)

import os
import numpy as np
import open3d as o3d
import tensorflow as tf
from vedo import load, write
import copy

INCISOR_LABELS = [ 6, 7, 8, 9]
TOOTH_NAMES = {
    #5: "Left Canine (5)",
    6: "Left Lateral (6)",
    7: "Left Central (7)",
    8: "Right Central (8)",
    9: "Right Lateral (9)",
    #10: "Right Canine (10)",
}

def load_model(model_path="predict/models/gumrecession_model_incisors.h5"):
    print(f"🔄 Loading model from: {model_path}")
    return tf.keras.models.load_model(model_path)

def get_toothface_gums(vtp_mesh, label_no):
    label = vtp_mesh.celldata['Label']
    faces = vtp_mesh.cells
    ind = np.where(label == label_no)[0].tolist()

    new_face = [faces[i] for i in ind]

    cutmesh = vtp_mesh.clone().threshold("Label", above=label_no - 0.1, below=label_no + 0.1, on='cells')
    if cutmesh.vertices.size == 0:
        return []

    xmax, ymax, zmax = cutmesh.vertices.max(axis=0)
    xmin, ymin, zmin = cutmesh.vertices.min(axis=0)

    m_xmax, m_ymax, m_zmax = vtp_mesh.vertices.max(axis=0)
    m_xmin, m_ymin, m_zmin = vtp_mesh.vertices.min(axis=0)

    x_b = (xmin, xmax)
    y_b = (m_ymin, m_ymax)
    z_b = (m_zmin, zmax)

    z_ids = vtp_mesh.find_cells_in_bounds(zbounds=z_b)
    y_ids = vtp_mesh.find_cells_in_bounds(ybounds=y_b)
    x_ids = vtp_mesh.find_cells_in_bounds(xbounds=x_b)

    inter = list(set(z_ids) & set(y_ids) & set(x_ids))
    new_face += [faces[i] for i in inter]
    return new_face

def extract_teeth_with_gums(vtp_path, ply_path, save_to="predict/uploads/lower/debug"):
    print(f"🔍 Loading VTP: {vtp_path}")
    print(f"🔍 Loading PLY: {ply_path}")

    pred_mesh = load(vtp_path)
    full_mesh = o3d.io.read_triangle_mesh(ply_path)

    meshes_by_label = {}
    os.makedirs(save_to, exist_ok=True)

    for label in INCISOR_LABELS:
        print(f"✂️ Extracting Tooth {label} with surrounding tissue...")
        tooth_faces = get_toothface_gums(pred_mesh, label)
        if not tooth_faces:
            meshes_by_label[label] = None
            print(f"⚠️ Tooth {label} not found")
            continue

        mesh_cropped = copy.deepcopy(full_mesh)
        mesh_cropped.triangles = o3d.utility.Vector3iVector(np.asarray(tooth_faces))
        debug_path = os.path.join(save_to, f"tooth_{label}.ply")
        o3d.io.write_triangle_mesh(debug_path, mesh_cropped)
        print(f"💾 Saved to: {debug_path}")
        meshes_by_label[label] = mesh_cropped
        pcd = o3d.geometry.PointCloud()
        pcd.points = mesh_cropped.vertices
        # Optional: transfer colors if available
        if mesh_cropped.has_vertex_colors():
            pcd.colors = mesh_cropped.vertex_colors
        meshes_by_label[label] = pcd

    return meshes_by_label

def preprocess_mesh_for_classification(mesh, num_points=5000):
    #pcd = o3d.geometry.PointCloud()
    #pcd.points = mesh.vertices if hasattr(mesh, 'vertices') else mesh.points
    pcd = mesh
    #print(mesh)

    if len(pcd.points) < num_points:
        print(f"⚠️ Not enough points in mesh: {len(pcd.points)}")
        return None

    #downpcd = pcd.uniform_down_sample(max(1, len(pcd.points) // num_points))
    downpcd = pcd.farthest_point_down_sample(5000)
    #if len(downpcd.points) > num_points:
        #downpcd = o3d.geometry.PointCloud()
        #downpcd.points = o3d.utility.Vector3dVector(np.asarray(pcd.points)[:num_points])

    colors=np.asarray(downpcd.colors)
    print(colors.shape)
    points = downpcd.points - np.mean(downpcd.points, axis=0)# Centralise points
    #points = np.asarray(downpcd.points)
    points = points / np.max(np.linalg.norm(points, axis=1)) # Nomralise points 
    points=np.asarray(points)
    print(points.shape)
    #points = points - points.mean(axis=0)
    #points = points / np.max(np.linalg.norm(points, axis=1))
    #colors = np.zeros_like(points)
    array = np.concatenate((colors, points), axis=1)
    return array

def predict_gum_recession(meshes_by_label, model):
    results = {}
    for label in INCISOR_LABELS:
        print(f"\n🦷 Predicting for tooth {label} ({TOOTH_NAMES[label]})...")
        mesh = meshes_by_label.get(label)
        if mesh is None:
            results[label] = "❌ Not found"
            continue

        print("🔍 Preprocessing tooth mesh...")
        array = preprocess_mesh_for_classification(mesh)
        if array is None or array.shape[0] < 500:
            results[label] = "⚠️ Insufficient points"
            continue

        prediction = model.predict(array[np.newaxis, :, :])[0][0]
        print(prediction)
        results[label] = "✅ Present" if prediction >= 0.7 else "🔴 Absent"
    return results