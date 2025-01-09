# Play Btn
# from PIL import Image, ImageDraw, ImageFilter

# # Create a high-resolution image
# size = 1000  # Increased size for higher quality
# image = Image.new("RGBA", (size, size), (255, 255, 255, 0))
# draw = ImageDraw.Draw(image)

# # Draw the white circle
# circle_radius = size // 2 - 50  # Radius with padding
# draw.ellipse(
#     [(50, 50), (size - 50, size - 50)], outline="white", width=30
# )

# # Define the play triangle points
# triangle_base = size // 3
# triangle_height = size // 2
# triangle_center = size // 2 + 50  # Adjusted to center triangle properly
# triangle = [
#     (triangle_center - triangle_base // 2, size // 3),  # Top point
#     (triangle_center - triangle_base // 2, size // 3 * 2),  # Bottom point
#     (triangle_center + triangle_base // 2, size // 2),  # Right point
# ]

# # Generate a filled triangle with rounded outside corners
# triangle_mask = Image.new("L", (size, size), 0)  # Mask for the rounded triangle
# mask_draw = ImageDraw.Draw(triangle_mask)

# # Draw the filled triangle
# mask_draw.polygon(triangle, fill=255)

# # Create rounded corners using a Gaussian blur
# # triangle_mask = triangle_mask.filter(ImageFilter.GaussianBlur(radius=20))

# # Apply the mask to the original image for a rounded, filled triangle
# image.paste("white", mask=triangle_mask)

# # Save the high-quality image
# output_path = "play_button_with_rounded_triangle.png"
# image.save(output_path)

# output_path

# Next/Prev Btns
from PIL import Image, ImageDraw

# Create a high-resolution image
size = 5000  # Increased size for higher quality
image = Image.new("RGBA", (size, size), (255, 255, 255, 0))
draw = ImageDraw.Draw(image)

# Define the play triangle points
triangle_base = size // 3
triangle_height = size // 2
triangle_center = size // 2 + 50  # Adjusted to center triangle properly
triangle = [
    (triangle_center - triangle_base // 2, size // 3),  # Top point
    (triangle_center - triangle_base // 2, size // 3 * 2),  # Bottom point
    (triangle_center + triangle_base // 2, size // 2),  # Right point
]

# Draw the filled triangle
draw.polygon(triangle, fill="white")

# Define the rectangle dimensions
rect_width = 100  # Specified width
rect_height_top = size // 3
rect_height_bottom = size // 3 * 2
rect_left = triangle_center + triangle_base // 2  # Start at the rightmost triangle point
rect_right = rect_left + rect_width

# Draw the rectangle
draw.rectangle(
    [(rect_left, rect_height_top), (rect_right, rect_height_bottom)],
    fill="white"
)
# Flip for prev btn. Comment out for right btn.
image = image.transpose(Image.FLIP_LEFT_RIGHT)

# Save the high-quality image
output_path = "prevBtn.png"
image.save(output_path)

output_path
