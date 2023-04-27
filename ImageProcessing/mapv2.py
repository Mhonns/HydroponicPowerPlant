# Image segmentation
#   Created by Korn Visaltanachoti

from PIL import Image
from collections import defaultdict
import math

# Load the image
img = Image.open("2023-03-31.png")

# Load the points
points = [(0,0)]

# Assign colors to the points
colors = [(255, 0, 0)]

# Map non-black pixels to the closest point and store the count of pixels connected to each point
point_counts = defaultdict(int)
mapped_img = Image.new("RGB", img.size)
for y in range(img.size[1]):
    for x in range(img.size[0]):
        pixel = img.getpixel((x, y))
        if pixel[0] > 0:  # Non-black pixel
            distances = [math.sqrt((x - p[0])**2 + (y - p[1])**2) for p in points]
            closest_point = distances.index(min(distances))
            point_counts[closest_point] += 1
            mapped_img.putpixel((x, y), colors[closest_point])

# Save the mapped image
mapped_img.save("output.png")

points.sort()

# Print the point counts
for i, count in sorted(point_counts.items()):
    print(f"Point {i}: {count} pixels")
