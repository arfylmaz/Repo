import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Path to the image
image_path = Path("BoxImages/rgb_3.png")
print(f"Processing image: {image_path}")

# Read the image
img = cv2.imread(str(image_path))
if img is None:
    print(f"Failed to load image {image_path}")
    exit()

# Convert BGR to RGB for display
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_copy = img_rgb.copy()

points = []  # Store points

def draw_lines(event, x, y, flags, param):
    global img_copy, points

    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        cv2.circle(img_copy, (x, y), 5, (255, 0, 0), -1)

        if len(points) > 1:
            cv2.line(img_copy, points[-2], points[-1], (255, 0, 0), 2)
        cv2.imshow("Draw Lines", img_copy)
        print(f"Point selected: {(x, y)}")

        if len(points) == 4:
            cv2.line(img_copy, points[3], points[0], (255, 0, 0), 2)
            cv2.imshow("Draw Lines", img_copy)
            cv2.destroyAllWindows()

cv2.namedWindow("Draw Lines")
cv2.setMouseCallback("Draw Lines", draw_lines)

while True:
    cv2.imshow("Draw Lines", img_copy)
    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # Press 'ESC' to exit the loop
        break
    if len(points) == 4:  # Exit loop after 4 points are selected
        break

cv2.destroyAllWindows()

if len(points) == 4:
    print(f"Points selected: {points}")

    # Convert points to numpy array
    pts = np.array(points, dtype=np.int32)
    print(f"Points array: {pts}")

    # Create a mask for the quadrilateral
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [pts], (255))
    print(f"Mask shape: {mask.shape}")

    # Apply the mask to the original image
    black_background = np.zeros_like(img)
    black_background[mask == 255] = img[mask == 255]
    print(f"Mask applied successfully")

    # Show the results
    plt.figure(figsize=(12, 6))
    plt.subplot(121)
    plt.imshow(img_rgb)
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(122)
    plt.imshow(cv2.cvtColor(black_background, cv2.COLOR_BGR2RGB))
    plt.title('Cropped Image on Black Background')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

    # Save the cropped image
    cv2.imwrite('BoxImages/crgb_3.png', black_background)
    print("Saved")
else:
    print("Four points were not selected.")
