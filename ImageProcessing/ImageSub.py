# Image subtraction 
#   Created by Korn Visaltanachoti

from PIL import Image

# Open the two input images
image1 = Image.open("15.jpg")
image2 = Image.open("14.jpg")

# Ensure both images are the same size
if image1.size != image2.size:
    raise ValueError("Input images must be the same size")

# Create a new image for the output
output = Image.new(image1.mode, image1.size)

# Perform image subtraction
for x in range(image1.width):
    for y in range(image1.height):
        pixel1 = image1.getpixel((x, y))
        pixel2 = image2.getpixel((x, y))
        if pixel1 == pixel2:
            output.putpixel((x, y), (0, 0, 0))
        else:
            output.putpixel((x, y), pixel1)

# Save the output image
output.save("output.jpg")
