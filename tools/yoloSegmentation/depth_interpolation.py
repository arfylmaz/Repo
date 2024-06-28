from pathlib import Path
import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# Path to the image directory

#image_path = Path("BoxImages/TakePhoto/rgb_20240603_145454_0_0_0.png")
depth_path = Path("depthplant.tiff")
print(f"Checking directory: {depth_path.resolve()}")

# Check if directory exists and has images
if not depth_path.exists():
    print("Image directory does not exist.")
    exit()


print(f"Processing image: {depth_path}")  # Print the image path
img = cv2.imread(str(depth_path))
if img is None:
    print(f"Failed to load image {depth_path}")

# Show the results
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title('Original Image')
plt.axis('off')
plt.show()


#cv2.imwrite('BoxImages/TakePhoto/cropped_20240603_145454_0_0_0.png', black_background)
print("Saved")