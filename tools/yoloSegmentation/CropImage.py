from pathlib import Path
import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# Path to the image directory
#
image_path = Path("BoxImages/TakePhoto/rgb_20240603_145454_0_0_0.png")
print(f"Checking directory: {image_path.resolve()}")

# Check if directory exists and has images
if not image_path.exists():
    print("Image directory does not exist.")
    exit()


print(f"Processing image: {image_path}")  # Print the image path
img = cv2.imread(str(image_path))
if img is None:
    print(f"Failed to load image {image_path}")

# Create a black image of the same size as the original for background
black_background = np.zeros_like(img)

x,y,h,w = 350, 180, 180, 250

bbox_rgb = img[y:y+h, x:x+w]

# Place the segmented bbox back onto the black background
black_background[y:y+h, x:x+w] = bbox_rgb

# Show the results
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title('Original Image')
plt.axis('off')
plt.show()

# Show the results
plt.imshow(cv2.cvtColor(black_background, cv2.COLOR_BGR2RGB))
plt.title('Cropped Image')
plt.axis('off')
plt.show()

#cv2.imwrite('BoxImages/TakePhoto/cropped_20240603_145454_0_0_0.png', black_background)
print("Saved")