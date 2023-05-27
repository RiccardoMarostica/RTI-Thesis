# import cv2 as cv
# import numpy as np

# image = np.zeros((DEFAULT_ASPECT_RATIO, DEFAULT_ASPECT_RATIO, 3), dtype=np.uint8)

# center_x = center_y = DEFAULT_ASPECT_RATIO // 2
# radius = DEFAULT_ASPECT_RATIO // 2

# point = np.array([[-0.82] [0.09] [0]], dtype=np.float32)

# cv.circle(image, (center_x, center_y), radius, (255, 255, 255), 2)
# cv.line(image, (0, center_y), (DEFAULT_ASPECT_RATIO, center_y), (255, 255, 255), 2)
# cv.line(image, (center_x, 0), (center_x, DEFAULT_ASPECT_RATIO), (255, 255, 255), 2)

# cv.imshow("Circle", image)
# cv.waitKey(0)
# cv.destroyAllWindows()

import cv2 as cv
import numpy as np

DEFAULT_ASPECT_RATIO = 400

# Define the image dimensions
image_width = image_height = DEFAULT_ASPECT_RATIO

# Create a blank image
image = np.zeros((image_height, image_width, 3), dtype=np.uint8)

# Calculate the center of the circle
center_x, center_y = image_width // 2, image_height // 2

print(center_x, center_y)

# Define the radius of the circle
radius = min(image_width, image_height) // 2

# Draw the circle border
cv.circle(image, (center_x, center_y), radius, (255, 255, 255), 1)
cv.line(image, (0, center_y), (image_height, center_y), (255, 255, 255), 1)
cv.line(image, (center_x, 0), (center_x, image_width), (255, 255, 255), 1)

cv.putText(image, "-1", (30, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1)
cv.putText(image, "1", (DEFAULT_ASPECT_RATIO - 30, DEFAULT_ASPECT_RATIO - 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1)

# point = np.array([[-0.82], [0.09], [1]])
point = np.array([[0], [1], [1]])
point = point[:2].tolist()
x = point[0][0]
y = point[1][0]
print(x, y)

# Convert the coordinates to normalized values between -1 and 1
# norm_x = (x / (image_width)) * 2 - 1
# norm_y = (y / (image_height)) * 2 - 1

x = ((x + 1) * image_width) / 2
y = ((y + 1) * image_height) / 2

print(x, y) 

# print(norm_x, norm_y)

# Calculate the distance from the center
distance = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
   
print(distance)
print(radius)
if distance <= radius:
    cv.circle(image, (int(x), int(y)), 2, (255, 255, 255), -1)
    cv.line(image, (center_x, center_y), (int(x), int(y)), (0, 255, 0), 2) 

# # Check if the distance is within the circle's radius
# if np.abs(distance - radius) <= 1:
    # Mark the coordinate on the image
    # cv.circle(image, (x, y), 1, (255, 255, 255), -1)

# Display the image
cv.imshow("Circle Visualization", image)
cv.waitKey(0)
cv.destroyAllWindows()