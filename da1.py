import streamlit as st
import cv2
import numpy as np
from PIL import Image
import pandas as pd

# Page configuration
st.set_page_config(page_title="Shape & Contour Analyzer", layout="wide")

# Title
st.title("🔷 Shape & Contour Analyzer")
st.markdown("Upload an image to detect geometric shapes, count objects, and calculate area & perimeter")

# Sidebar controls
st.sidebar.header("⚙️ Controls")
threshold_value = st.sidebar.slider("Threshold Value", 0, 255, 127)
min_area = st.sidebar.slider("Minimum Contour Area", 100, 5000, 500)
blur_kernel = st.sidebar.slider("Blur Kernel Size (odd)", 1, 15, 5, step=2)

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png', 'bmp'])

def classify_shape(contour):
    """Classify shape based on number of vertices"""
    peri = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.04 * peri, True)
    vertices = len(approx)
    
    if vertices == 3:
        return "Triangle", vertices
    elif vertices == 4:
        # Check if square or rectangle
        x, y, w, h = cv2.boundingRect(approx)
        aspect_ratio = float(w) / h
        if 0.95 <= aspect_ratio <= 1.05:
            return "Square", vertices
        else:
            return "Rectangle", vertices
    elif vertices == 5:
        return "Pentagon", vertices
    elif vertices == 6:
        return "Hexagon", vertices
    elif vertices > 6:
        return "Circle", vertices
    else:
        return "Unknown", vertices

def detect_shapes(image, thresh_val, min_area_val, blur_val):
    """Main shape detection function"""
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(gray, (blur_val, blur_val), 0)
    
    # Apply threshold
    _, thresh = cv2.threshold(blurred, thresh_val, 255, cv2.THRESH_BINARY_INV)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Create output image
    output = image.copy()
    
    shapes_data = []
    valid_contours = 0
    
    for i, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        
        # Filter by minimum area
        if area < min_area_val:
            continue
            
        valid_contours += 1
        
        # Calculate features
        perimeter = cv2.arcLength(contour, True)
        shape_name, vertices = classify_shape(contour)
        
        # Get center point
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            cx, cy = 0, 0
        
        # Draw contour and label
        cv2.drawContours(output, [contour], -1, (0, 255, 0), 3)
        cv2.putText(output, f"{shape_name}", (cx - 40, cy), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
        # Store data
        shapes_data.append({
            'Object': f"#{valid_contours}",
            'Shape': shape_name,
            'Vertices': vertices,
            'Area (px²)': round(area, 2),
            'Perimeter (px)': round(perimeter, 2),
            'Center': f"({cx}, {cy})"
        })
    
    return output, thresh, shapes_data, valid_contours

if uploaded_file is not None:
    # Read image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    # Process image
    output_image, threshold_image, shapes_data, object_count = detect_shapes(
        image, threshold_value, min_area, blur_kernel
    )
    
    # Display results
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📷 Original Image")
        st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), use_container_width=True)
        
    with col2:
        st.subheader("🎯 Detected Shapes")
        st.image(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB), use_container_width=True)
    
    # Metrics
    st.subheader("📊 Analysis Results")
    metric_col1, metric_col2, metric_col3 = st.columns(3)
    
    with metric_col1:
        st.metric("Total Objects Detected", object_count)
    
    with metric_col2:
        if shapes_data:
            avg_area = np.mean([s['Area (px²)'] for s in shapes_data])
            st.metric("Average Area", f"{avg_area:.2f} px²")
    
    with metric_col3:
        if shapes_data:
            avg_perimeter = np.mean([s['Perimeter (px)'] for s in shapes_data])
            st.metric("Average Perimeter", f"{avg_perimeter:.2f} px")
    
    # Data table
    if shapes_data:
        st.subheader("📋 Detailed Measurements")
        df = pd.DataFrame(shapes_data)
        st.dataframe(df, use_container_width=True)
        
        # Download option
        csv = df.to_csv(index=False)
        st.download_button(
            label="⬇️ Download Results as CSV",
            data=csv,
            file_name="shape_analysis_results.csv",
            mime="text/csv"
        )
    
    # Show threshold image in expander
    with st.expander("🔍 View Threshold Image"):
        st.image(threshold_image, caption="Binary Threshold", use_container_width=True)

else:
    st.info("👆 Please upload an image to begin shape detection")
    
    # Instructions
    st.markdown("""
    ### How to Use:
    1. Upload an image containing geometric shapes
    2. Adjust threshold and minimum area in the sidebar
    3. View detected shapes with measurements
    4. Download results as CSV
    
    ### Detected Shapes:
    - Triangles (3 vertices)
    - Squares & Rectangles (4 vertices)
    - Pentagons (5 vertices)
    - Hexagons (6 vertices)
    - Circles (7+ vertices)
    """)
