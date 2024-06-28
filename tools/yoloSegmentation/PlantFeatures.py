from pathlib import Path
import cv2
#import open3d as o3d
import matplotlib
import matplotlib.pyplot as plt
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors
import numpy as np

"""
Processing image: Broccoli/rgb_20240303_183217_1000_120_0.png
0: 384x640 2 crops, 779.8ms
Speed: 1.1ms preprocess, 779.8ms inference, 2.6ms postprocess per image at shape (1, 3, 384, 640)
"""

# Load the YOLO model once
model = YOLO("best.pt")

# Path to the image directory
image_path = Path("biggerrgb.png")
depth_path = Path("biggerdepth.tiff")
print(f"Checking directory: {image_path.resolve()}")

# Check if directory exists and has images
if not image_path.exists():
    print("Image directory does not exist.")
    exit()


print(f"Processing image: {image_path}")  # Print the image path
img = cv2.imread(str(image_path))
if img is None:
    print(f"Failed to load image {image_path}")
depth_img = cv2.imread(str(depth_path))
if depth_img is None:
    print(f"Failed to load depth image {depth_path}")

# Predict using the loaded model
results = model.predict(img)
annotator = Annotator(img, line_width=2)

# Camera parameters
fx = 1053.02
fy = 1052.72
cx = 894.2
cy = 560.436

# If masks are available in the results, annotate them
if results[0].masks is not None:
    clss = results[0].boxes.cls.cpu().tolist()
    masks = results[0].masks.xy
    boxes = results[0].boxes.xyxy.cpu().tolist()
    for mask, box, cls in zip(masks, boxes, clss):
        color = colors(int(cls),True)
        annotator.seg_bbox(mask=mask, mask_color=colors(int(cls), True))
        annotator.box_label(box=box, color=color)
        break
# Display the annotated image
cv2.imshow("Instance Segmentation", img)
print(mask.shape)
print(dir(mask))
print(type(mask))



"""
segmented_rgb = cv2.bitwise_and(rgb_image, rgb_imag, mask=mask)
segmented_depth = cv2.bitwise_and(depth_image, depth_image, mask=mask)


intrinsic = o3d.camera.PinholeCameraIntrinsic(img.shape[1], img.shape[0], fx, fy, cx, cy)

# Convert depth image to point cloud
depth_o3d = o3d.geometry.Image(segmented_depth.astype(np.float32))
rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
    o3d.geometry.Image(segmented_rgb),
    depth_o3d,
    depth_scale=1000.0,  # depends on your depth unit
    depth_trunc=1000.0,  # max depth value
    convert_rgb_to_intensity=False
)

pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
    rgbd_image,
    intrinsic
)

# Optionally visualize the point cloud
o3d.visualization.draw_geometries([pcd])
"""
mask = np.array(mask, dtype=np.int32)  # Ensure it's integer type
mask = mask.reshape((-1, 1, 2))        # Reshape to (n, 1, 2) for fillPoly

binary_mask = np.zeros(img.shape[:2], dtype=np.uint8)
cv2.fillPoly(binary_mask, [mask], 255)

# Apply the mask to isolate the region of interest
segmented_rgb = cv2.bitwise_and(img, img, mask=binary_mask)

# Load the original image
org_img = cv2.imread(str(image_path))

# Extract bounding box coordinates
x, y, w, h = cv2.boundingRect(mask)

# Create a black image of the same size as the original for background
black_background = np.zeros_like(org_img)

# Crop the mask to the bounding box and apply it to the cropped RGB region
bbox_mask = binary_mask[y:y+h, x:x+w]
plt.imshow(cv2.cvtColor(org_img, cv2.COLOR_BGR2RGB))
plt.show()
bbox_rgb = org_img[y:y+h, x:x+w]

# Place the segmented bbox back onto the black background
black_background[y:y+h, x:x+w] = bbox_rgb

# Show the results
plt.imshow(cv2.cvtColor(black_background, cv2.COLOR_BGR2RGB))
plt.title('Segmented BBox on Black Background')
plt.axis('off')
plt.show()

# Show the segmented images
plt.imshow(cv2.cvtColor(segmented_rgb, cv2.COLOR_BGR2RGB))
plt.title('Segmented RGB Image')
plt.axis('off')
plt.show()

cv2.imwrite('bigger_segmented_rgb.png', segmented_rgb)
#print("Saved")
cv2.imwrite('bigger_bbox_rgb.png', black_background)
print("Saved")

