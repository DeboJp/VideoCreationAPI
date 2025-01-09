from moviepy import *

# Create a TextClip with the desired text
text_clip = TextClip(
    text="Hello World",
    color="white",
    bg_color="black",
    size=(640, 360),  # Resolution of the video
    font="Arial",     # Ensure the font is specified and available on your system
    font_size=70,
    duration=10       # Duration of the video
)

# Compose the video (optional for a single clip)
final_video = CompositeVideoClip([text_clip])

# Write the video file
final_video.write_videofile("hello_world.mp4", fps=24)

