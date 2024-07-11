import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the depth image
depth_image_path = "depth_image_8.png"
depth_image = cv2.imread(depth_image_path, cv2.IMREAD_UNCHANGED)

# Convert the depth image to float32 type
depth_image = depth_image.astype(np.float32) # uint16인 데이터셋을 256.0으로나누기위함
depth_image /= 256.0

# Analyze the depth image
unique, counts = np.unique(depth_image, return_counts=True)
most_common_depth = unique[np.argmax(counts)]
min_depth = np.min(depth_image[depth_image > 0])  # Minimum non-zero depth
max_depth = np.max(depth_image)

# Displaying the analysis results
analysis_results = {
    "Unique depths": unique,
    "Counts of each depth": counts,
    "Most common depth": most_common_depth,
    "Minimum non-zero depth": min_depth,
    "Maximum depth": max_depth
}

# Visualizing the depth image
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.imshow(depth_image, cmap='plasma')
plt.title("Depth Image")
plt.colorbar(label='Depth')

plt.subplot(1, 2, 2)
plt.hist(depth_image[depth_image > 0].ravel(), bins=100, color='navy')
plt.title("Depth Histogram")
plt.xlabel("Depth")
plt.ylabel("Frequency")
plt.grid(True)
plt.show()
# Return the analysis results
print(analysis_results)
