# First working vers?
# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas  # Use Agg backend for rendering
# from moviepy.video.VideoClip import VideoClip
# from moviepy.audio.io.AudioFileClip import AudioFileClip
# from scipy.fft import fft

# # Configuration
# AUDIO_FILE = "audio.wav"          # Path to the audio file
# BACKGROUND_IMAGE = "background.jpg"  # Path to the background image
# OUTPUT_FILE = "output_video.mp4"  # Path to save the final video
# DURATION = 10                     # Duration of the video in seconds
# FPS = 30                          # Frames per second

# # Function to generate audio spectrum
# def audio_spectrum_frame(t, audio_data, sample_rate):
#     n = len(audio_data)
#     start = int(t * sample_rate) % n
#     end = min(start + 2048, n)
#     audio_chunk = audio_data[start:end]

#     # FFT of audio data
#     fft_data = np.abs(fft(audio_chunk))[:1024]
#     if np.max(fft_data) == 0:  # Prevent division by zero
#         return np.zeros(100)
#     spectrum = fft_data[:100]  # Reduce to 100 frequency bands
#     return spectrum / np.max(spectrum)  # Normalize for visualization

# # Visualization function
# def make_frame(t):
#     spectrum = audio_spectrum_frame(t, audio_data, sample_rate)

#     # Create a figure
#     fig, ax = plt.subplots(figsize=(16, 9))
#     canvas = FigureCanvas(fig)  # Use Agg canvas
#     ax.clear()
#     ax.imshow(background_img, extent=[0, 100, 0, 1], aspect='auto')  # Add background image
#     ax.bar(range(len(spectrum)), spectrum, color='cyan', alpha=0.7)  # Add spectrum bars
#     ax.text(10, 0.9, "Music Visualization", color='white', fontsize=20, fontweight='bold')  # Add text
#     ax.axis("off")  # Hide axes

#     # Convert to numpy array
#     canvas.draw()  # Render the figure onto the canvas
#     frame = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8)
#     frame = frame.reshape(canvas.get_width_height()[::-1] + (3,))
#     plt.close(fig)
#     return frame

# # Load the audio and prepare data
# audio_clip = AudioFileClip(AUDIO_FILE)
# sample_rate = int(audio_clip.fps)
# audio_data = audio_clip.to_soundarray(fps=sample_rate).mean(axis=1)  # Mono audio

# # Load the background image
# background_img = plt.imread(BACKGROUND_IMAGE)

# # Create the video clip
# video_clip = VideoClip(lambda t: make_frame(t), duration=min(DURATION, audio_clip.duration))
# video_clip = video_clip.with_audio(audio_clip.subclipped(0, DURATION))  # Set audio for the clip

# # Write the video file
# video_clip.write_videofile(OUTPUT_FILE, fps=FPS, codec="libx264", audio_codec="aac")

# print(f"Video saved to {OUTPUT_FILE}")




# Working version before testing visuals
# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
# from moviepy.video.VideoClip import VideoClip
# from moviepy.audio.io.AudioFileClip import AudioFileClip
# from scipy.fft import fft

# # Configuration
# AUDIO_FILE = "audio.wav"
# BACKGROUND_IMAGE = "bg2.jpg"
# OUTPUT_FILE = "output_video.mp4"
# DURATION = 5
# FPS = 10  # Lower FPS to save memory

# # Function to generate audio spectrum
# def audio_spectrum_frame(t, audio_data, sample_rate):
#     n = len(audio_data)
#     start = int(t * sample_rate) % n
#     end = min(start + 2048, n)
#     audio_chunk = audio_data[start:end]

#     fft_data = np.abs(fft(audio_chunk))[:1024]
#     if np.max(fft_data) == 0:
#         return np.zeros(100)
#     spectrum = fft_data[:100]
#     return spectrum / np.max(spectrum)

# # Visualization function
# def make_frame(t):
#     spectrum = audio_spectrum_frame(t, audio_data, sample_rate)

#     fig, ax = plt.subplots(figsize=(8, 4.5))  # Smaller resolution
#     canvas = FigureCanvas(fig)
#     ax.clear()
#     ax.imshow(background_img, extent=[0, 100, 0, 1], aspect='auto')
#     ax.bar(range(len(spectrum)), spectrum, color='cyan', alpha=0.7)
#     ax.text(10, 0.9, "Music Visualization", color='white', fontsize=20, fontweight='bold')
#     ax.axis("off")

#     canvas.draw()
#     frame = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8)
#     frame = frame.reshape(canvas.get_width_height()[::-1] + (3,))
#     plt.close(fig)  # Free memory
#     return frame

# # Load the audio and prepare data
# audio_clip = AudioFileClip(AUDIO_FILE)
# # .with_fps(22050)  # Lower sample rate
# sample_rate = int(audio_clip.fps)
# audio_data = audio_clip.to_soundarray(fps=sample_rate).mean(axis=1)

# # Load the background image
# background_img = plt.imread(BACKGROUND_IMAGE)

# # Create the video clip
# video_clip = VideoClip(lambda t: make_frame(t), duration=min(DURATION, audio_clip.duration))
# video_clip = video_clip.with_audio(audio_clip.subclipped(0, DURATION))

# # Write the video file
# video_clip.write_videofile(
#     OUTPUT_FILE, 
#     fps=FPS, 
#     codec="libx264", 
#     audio_codec="aac", 
#     preset="ultrafast", 
# )

# print(f"Video saved to {OUTPUT_FILE}")

# Attemp 1
# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
# from moviepy.video.VideoClip import VideoClip
# from moviepy.audio.io.AudioFileClip import AudioFileClip
# from scipy.fft import fft
# from PIL import Image

# # Configuration
# AUDIO_FILE = "audio.wav"
# BACKGROUND_IMAGE = "bg2.jpg"
# OUTPUT_FILE = "test.mp4"
# DURATION = 5
# FPS = 5

# # Preprocess background image to fit 1920x1080 resolution
# def preprocess_background(image_path, target_size=(1920, 1080)):
#     img = Image.open(image_path)
#     #ANTIALIAS NO LONGER SUPPORTED: need to use PIL.Image.LANCZOS or PIL.Image.Resampling.LANCZOS.
#     img = img.resize(target_size, Image.Resampling.LANCZOS)
#     return np.array(img)

# # Generate audio spectrum
# def audio_spectrum_frame(t, audio_data, sample_rate):
#     n = len(audio_data)
#     start = int(t * sample_rate) % n
#     end = min(start + 2048, n)
#     audio_chunk = audio_data[start:end]

#     fft_data = np.abs(fft(audio_chunk))[:1024]
#     if np.max(fft_data) == 0:
#         return np.zeros(200)  # Use 200 frequency bands
#     spectrum = fft_data[:200]
#     return spectrum / np.max(spectrum)

# # Visualization function
# def make_frame(t):
#     spectrum = audio_spectrum_frame(t, audio_data, sample_rate)

#     fig, ax = plt.subplots(figsize=(16, 9))
#     canvas = FigureCanvas(fig)
#     ax.clear()
    
#     # Add background image
#     ax.imshow(background_img, extent=[0, 200, 0, 1], aspect='auto')

#     # Add bar plot with customizations
#     colors = plt.cm.plasma(np.linspace(0, 1, len(spectrum)))  # Gradient color scheme
#     ax.bar(range(len(spectrum)), spectrum, color=colors, alpha=0.8, width=1.0)

#     # Add text
#     ax.text(100, 0.9, "Music Visualization", color='white', fontsize=30, fontweight='bold', ha='center')

#     # Remove axes
#     ax.axis("off")

#     canvas.draw()
#     frame = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8)
#     frame = frame.reshape(canvas.get_width_height()[::-1] + (3,))
#     plt.close(fig)
#     return frame

# # Load audio
# audio_clip = AudioFileClip(AUDIO_FILE)
# sample_rate = int(audio_clip.fps)
# audio_data = audio_clip.to_soundarray(fps=sample_rate).mean(axis=1)

# # Load and preprocess the background image
# background_img = preprocess_background(BACKGROUND_IMAGE)

# # Create the video clip
# video_clip = VideoClip(lambda t: make_frame(t), duration=min(DURATION, audio_clip.duration))
# video_clip = video_clip.with_audio(audio_clip.subclipped(0, DURATION))

# # Write the video file
# video_clip.write_videofile(
#     OUTPUT_FILE, 
#     fps=FPS, 
#     codec="libx264", 
#     audio_codec="aac", 
#     preset="ultrafast",
# )

# print(f"Video saved to {OUTPUT_FILE}")




## WORKING VERSION Touched Up
# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
# from moviepy.video.VideoClip import VideoClip
# from moviepy.audio.io.AudioFileClip import AudioFileClip
# from scipy.fft import fft
# from PIL import Image, ImageDraw, ImageFilter, ImageEnhance, ImageFont
# from scipy.ndimage import gaussian_filter1d

# # Configuration
# AUDIO_FILE = "audio.wav"
# BACKGROUND_IMAGE = "bg2.jpg"
# OUTPUT_FILE = "test2.mp4"
# DURATION = 60
# FPS = 23.967
# SONG_NAME = "Let Me Down Slowly"

# # Preprocess background image for "object-fit: cover"
# def preprocess_background(image_path, target_size=(1920, 1080), blur = False):
#     img = Image.open(image_path)
#     img_ratio = img.width / img.height
#     target_ratio = target_size[0] / target_size[1]

#     if img_ratio > target_ratio:
#         # Image is wider than target; crop sides
#         new_width = int(target_size[1] * img_ratio)
#         img = img.resize((new_width, target_size[1]), Image.Resampling.LANCZOS)
#         left = (img.width - target_size[0]) // 2
#         img = img.crop((left, 0, left + target_size[0], target_size[1]))
#     else:
#         # Image is taller than target; crop top and bottom
#         new_height = int(target_size[0] / img_ratio)
#         img = img.resize((target_size[0], new_height), Image.Resampling.LANCZOS)
#         top = (img.height - target_size[1]) // 2
#         img = img.crop((0, top, target_size[0], top + target_size[1]))
    
#     if blur:
#         img = img.filter(ImageFilter.GaussianBlur(radius=20))
#         # Darken the image
#         enhancer = ImageEnhance.Brightness(img)
#         img = enhancer.enhance(0.5)  

#     return np.array(img)

# # Generate audio spectrum
# def audio_spectrum_frame(t, audio_data, sample_rate):
#     n = len(audio_data)
#     start = int(t * sample_rate) % n
#     end = min(start + 2048, n)
#     audio_chunk = audio_data[start:end]

#     fft_data = np.abs(fft(audio_chunk))[:1024]
#     if np.max(fft_data) == 0:
#         return np.zeros(200)  # Use 200 frequency bands
    
#     # Normalizing Curve: to look smooth.[logarithmic scaling with smoothing filter to make wavy effect.]
#     spectrum = fft_data[:200]
#     spectrum = np.log1p(spectrum)  # Compress using log scaling
#     spectrum = spectrum.reshape(50, -1).mean(axis=1)
#     spectrum = gaussian_filter1d(spectrum, sigma=2)  # Smooth the result
#     spectrum /= np.max(spectrum)  # Normalize to 0-1

#     return spectrum

# # Visualization function
# def make_frame(t):
#     spectrum = audio_spectrum_frame(t, audio_data, sample_rate)

#     # Start with the background image
#     frame = Image.fromarray(background_img.copy())
#     draw = ImageDraw.Draw(frame)

#     # CREATING IMAGE ON SQUAUE W/ SHADOWS
#     # Create the square overlay (object-fit: cover)
#     overlay_size = 500  # Size of the square box
#     overlay_image = preprocess_background(BACKGROUND_IMAGE, target_size=(overlay_size, overlay_size))
#     # Add shadow effect
#     shadow = Image.new("RGBA", (overlay_size + 20, overlay_size + 20), (0, 0, 0, 0))
#     shadow_draw = ImageDraw.Draw(shadow)
#     shadow_draw.rectangle(
#         [(10, 10), (overlay_size + 10, overlay_size + 10)],
#         fill=(0, 0, 0, 150)  # Semi-transparent black for shadow
#     )
#     shadow = shadow.filter(ImageFilter.GaussianBlur(radius=15))  # Blur the shadow

#     # PASTING IMAGE ON SQUAUE W/ SHADOWS
#     # Paste shadow on the background
#     shadow_position = (700, 290)  # Centered on 1920x1080 (960, 540) minus half overlay size
#     frame.paste(shadow, shadow_position, shadow)
#     # Paste the square overlay on top of the shadow
#     overlay_image = Image.fromarray(overlay_image)
#     overlay_position = (710, 300)  # Centered and slightly offset by the shadow
#     frame.paste(overlay_image, overlay_position)

#     # Create a transparent figure for the barplot
#     fig, ax = plt.subplots(figsize=(16, 9))
#     fig.patch.set_alpha(0)  # Transparent figure background
#     ax.patch.set_alpha(0)   # Transparent axes background

#     # Add bar plot with customizations
#     # colors = plt.cm.plasma(np.linspace(0, 1, len(spectrum)))  # Gradient color scheme (usage color = colors)
#     ax.bar(range(len(spectrum)), spectrum, color='white', alpha=1, width=0.6)

#     # Remove axes
#     ax.axis("off")

#     # Render the plot to a NumPy array with RGBA channels
#     canvas = FigureCanvas(fig)
#     canvas.draw()
#     width, height = canvas.get_width_height()  # Get canvas dimensions
#     plot_frame = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8).reshape((height, width, 4))  # Ensure correct shape
#     plt.close(fig)

#     # Convert plot frame to PIL image
#     plot_image = Image.fromarray(plot_frame, mode="RGBA")

#     # Resize and paste the plot as an overlay
#     plot_image = plot_image.resize((920, 140), Image.Resampling.LANCZOS)  # Adjust height for the barplot
#     frame.paste(plot_image, (490, 710), plot_image)  # Position the barplot at the bottom with transparency

#     ## Buttons next,prev and Play
#     # Add PNG overlays for buttons
#     button_size = 100  # Size of each button
#     button_gap = 35   # Gap between buttons
#     opacity = 1.0

#     # Load and resize buttons
#     next_button = Image.open("res/nextBtn.png").resize((button_size, button_size), Image.Resampling.LANCZOS)
#     play_button = Image.open("res/playBtn.png").resize((button_size, button_size), Image.Resampling.LANCZOS)
#     prev_button = Image.open("res/prevBtn.png").resize((button_size, button_size), Image.Resampling.LANCZOS)
#     def apply_opacity(image, opacity):
#         alpha = image.getchannel("A")
#         alpha = alpha.point(lambda p: int(p * opacity))  # Scale alpha channel
#         image.putalpha(alpha)
#         return image

#     next_button = apply_opacity(next_button, opacity)
#     play_button = apply_opacity(play_button, opacity)
#     prev_button = apply_opacity(prev_button, opacity)
#     # Calculate positions for bottom center
#     total_width = 3 * button_size + 2 * button_gap  # Total width of all buttons + gaps
#     center_x = 960  # Center of the screen
#     y_position = 880  # Bottom of the screen with some padding
#     start_x = center_x - total_width // 2  # Starting x position for the first button
#     # Paste buttons
#     frame.paste(prev_button, (start_x, y_position), prev_button)
#     frame.paste(play_button, (start_x + button_size + button_gap, y_position), play_button)
#     frame.paste(next_button, (start_x + 2 * (button_size + button_gap), y_position), next_button)

#     # Add text overlay on the video
#         # Load fonts
#     now_playing_font = ImageFont.truetype("fonts/Staatliches-Regular.ttf", 60)  # Large "NOW PLAYING" text
#     song_name_font = ImageFont.truetype("fonts/PermanentMarker-Regular.ttf", 90)  # Song name
#     artist_font = ImageFont.truetype("fonts/Fondamento-Regular.ttf", 60)  # Artist name
#     # Add "NOW PLAYING" text
#     draw.text( (850, 100), "NOWPLAYING:", fill="white", anchor="mm", align="center", font=now_playing_font,)
#     draw.text((960, 173), SONG_NAME, fill="white",anchor="mm",align="center",font=song_name_font,)
#     draw.text( (960, 255),  "Artist: Tech Music", fill="white", anchor="mm", align="center", font=artist_font,)

#     # draw.text((960, 50), "@CopyRighted by Tech Music", fill="black", anchor="mm", align="center", size=60)

#     return np.array(frame)


# # Load audio
# audio_clip = AudioFileClip(AUDIO_FILE)
# sample_rate = int(audio_clip.fps)
# audio_data = audio_clip.to_soundarray(fps=sample_rate).mean(axis=1)

# # Load and preprocess the background image
# background_img = preprocess_background(BACKGROUND_IMAGE, blur=True)

# # Create the video clip
# video_clip = VideoClip(lambda t: make_frame(t), duration=min(DURATION, audio_clip.duration))
# video_clip = video_clip.with_audio(audio_clip.subclipped(0, DURATION))

# # Write the video file
# video_clip.write_videofile(
#     OUTPUT_FILE, 
#     fps=FPS, 
#     codec="libx264", 
#     audio_codec="aac", 
#     preset="ultrafast",
# )

# print(f"Video saved to {OUTPUT_FILE}")

