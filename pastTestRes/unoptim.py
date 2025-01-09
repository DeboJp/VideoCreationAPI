# ## Optimized Version for better performance.
# import os, psutil, time
# import threading
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
# OUTPUT_FILE = "test1.mp4"
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

# # For Resource Monitoring.
# def monitor_process(pid, interval, results, stop_event):
#     process = psutil.Process(pid)
#     total_cpu_time = 0
#     total_memory_used = 0
#     data_points = 0

#     try:
#         while not stop_event.is_set():  # Check if stop signal is received
#             if not process.is_running():
#                 break  # Exit if process is no longer running
            
#             # Fetch CPU and memory usage
#             cpu_usage = process.cpu_percent(interval=0)  # CPU in percent since last call
#             memory_usage = process.memory_info().rss / (1024 * 1024)  # Memory in MB
            
#             # Accumulate stats
#             total_cpu_time += cpu_usage * (interval / 100)  # Convert percent to actual time
#             total_memory_used += memory_usage
#             data_points += 1

#             time.sleep(interval)
#     except psutil.NoSuchProcess:
#         pass  # Process ended

#     # Calculate averages
#     avg_memory = total_memory_used / data_points if data_points else 0

#     # Store results in shared dictionary
#     results["cpu_time"] = total_cpu_time
#     results["avg_memory"] = avg_memory


# if __name__ == "__main__":

#     # Optimization check(start):
#     startTime = time.time()
#     pid = os.getpid()  # Current process ID
#     interval = 1  # Sampling interval in seconds
#     # monitoring background threads (Cpu multithreaded)
#     monitor_results = {}
#     stop_event = threading.Event()  # Event to signal the monitoring thread to stop
#     monitor_thread = threading.Thread(target=monitor_process, args=(pid, interval, monitor_results, stop_event))
#     monitor_thread.start()

#     # Load audio
#     audio_clip = AudioFileClip(AUDIO_FILE)
#     sample_rate = int(audio_clip.fps)
#     audio_data = audio_clip.to_soundarray(fps=sample_rate).mean(axis=1)

#     # Load and preprocess the background image
#     background_img = preprocess_background(BACKGROUND_IMAGE, blur=True)

#     # Create the video clip
#     video_clip = VideoClip(lambda t: make_frame(t), duration=min(DURATION, audio_clip.duration))
#     video_clip = video_clip.with_audio(audio_clip.subclipped(0, DURATION))

#     # Write the video file
#     video_clip.write_videofile(
#         OUTPUT_FILE, 
#         fps=FPS, 
#         codec="libx264", 
#         audio_codec="aac", 
#         preset="ultrafast",
#     )

#     print(f"Video saved to {OUTPUT_FILE}")
#     #Optimization checks(end/summary)
#     endTime = time.time()
#     stop_event.set()  # Notify the monitoring thread to exit
#     monitor_thread.join() #waits for threads to finish for consistency.
#     # Get summarized results
#     # Retrieve summarized results from the thread
#     total_cpu_used = monitor_results.get("cpu_time", 0)
#     avg_memory_used = monitor_results.get("avg_memory", 0)

#     print("\nResource Usage Summary:")
#     print(f"Total time used: {endTime-startTime:.2f} seconds, parameters Duration and FPS: {DURATION, FPS}")
#     print(f"Total CPU Time Used: {total_cpu_used:.2f} seconds")
#     print(f"Average Memory Usage: {avg_memory_used:.2f} MB")

## Optimized Version1.0 for better performance.
import os, psutil, time
import threading
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from moviepy.video.VideoClip import VideoClip
from moviepy.audio.io.AudioFileClip import AudioFileClip
from scipy.fft import fft
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance, ImageFont
from scipy.ndimage import gaussian_filter1d

# Configuration
AUDIO_FILE = "audio.wav"
BACKGROUND_IMAGE = "bg2.jpg"
OUTPUT_FILE = "test4.mp4"
DURATION = 60
FPS = 23.967
SONG_NAME = "Let Me Down Slowly"

# Preload Static Resources
now_playing_font = ImageFont.truetype("fonts/Staatliches-Regular.ttf", 60)
song_name_font = ImageFont.truetype("fonts/PermanentMarker-Regular.ttf", 90)
artist_font = ImageFont.truetype("fonts/Fondamento-Regular.ttf", 60)
button_size = 100
button_gap = 35
next_button = Image.open("res/nextBtn.png").resize((button_size, button_size), Image.Resampling.LANCZOS).convert("RGBA")
play_button = Image.open("res/playBtn.png").resize((button_size, button_size), Image.Resampling.LANCZOS).convert("RGBA")
prev_button = Image.open("res/prevBtn.png").resize((button_size, button_size), Image.Resampling.LANCZOS).convert("RGBA")
background_img = None
shadow = None
overlay_image = None


# Preprocess Background Image
def preprocess_background(image_path, target_size=(1920, 1080), blur=False):
    img = Image.open(image_path)
    img_ratio = img.width / img.height
    target_ratio = target_size[0] / target_size[1]

    if img_ratio > target_ratio:
        new_width = int(target_size[1] * img_ratio)
        img = img.resize((new_width, target_size[1]), Image.Resampling.LANCZOS)
        left = (img.width - target_size[0]) // 2
        img = img.crop((left, 0, left + target_size[0], target_size[1]))
    else:
        new_height = int(target_size[0] / img_ratio)
        img = img.resize((target_size[0], new_height), Image.Resampling.LANCZOS)
        top = (img.height - target_size[1]) // 2
        img = img.crop((0, top, target_size[0], top + target_size[1]))

    if blur:
        img = img.filter(ImageFilter.GaussianBlur(radius=20))
        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(0.5)

    return img


# Generate Audio Spectrum
def audio_spectrum_frame(t, audio_data, sample_rate):
    n = len(audio_data)
    start = int(t * sample_rate) % n
    end = min(start + 2048, n)
    audio_chunk = audio_data[start:end]

    fft_data = np.abs(fft(audio_chunk))[:1024]
    if np.max(fft_data) == 0:
        return np.zeros(200)

    spectrum = fft_data[:200]
    spectrum = np.log1p(spectrum)
    spectrum = spectrum.reshape(50, -1).mean(axis=1)
    spectrum = gaussian_filter1d(spectrum, sigma=2)
    spectrum /= np.max(spectrum)

    return spectrum


# Create Shadow Once
def create_shadow(overlay_size, blur_radius=15):
    shadow = Image.new("RGBA", (overlay_size + 20, overlay_size + 20), (0, 0, 0, 0))
    shadow_draw = ImageDraw.Draw(shadow)
    shadow_draw.rectangle(
        [(10, 10), (overlay_size + 10, overlay_size + 10)],
        fill=(0, 0, 0, 150)
    )
    return shadow.filter(ImageFilter.GaussianBlur(radius=blur_radius))


# Visualization Function
def make_frame(t):
    spectrum = audio_spectrum_frame(t, audio_data, sample_rate)

    # Start with preloaded background
    frame = background_img.copy()
    draw = ImageDraw.Draw(frame)

    # Add shadow and square overlay
    frame.paste(shadow, (700, 290), shadow)
    frame.paste(overlay_image, (710, 300))

    # Draw Bar Plot using Pillow
    bar_width = 7  # Width of each bar
    bar_gap = 6  # Space between bars
    num_bars = len(spectrum)  # Number of bars (matches spectrum size)
    bar_plot_width = num_bars * (bar_width + bar_gap) - bar_gap  # Total width of the bar plot
    bar_x_start = 960 - bar_plot_width // 2  # Center the bar plot horizontally
    bar_plot_height = 94  # Max height of the bars
    bar_plot_y_position = 833  # Y-coordinate of the bottom of the bars

    for i, bar_height in enumerate(spectrum * bar_plot_height):
        if bar_height < 1e-3:  # Skip drawing very small or zero bars
            continue

        x0 = bar_x_start + i * (bar_width + bar_gap)
        y0 = bar_plot_y_position - bar_height
        x1 = x0 + bar_width
        y1 = bar_plot_y_position

        draw.rectangle([x0, y0, x1, y1], fill="white")

    # Add Buttons
    total_width = 3 * button_size + 2 * button_gap
    center_x = 960
    y_position = 880
    start_x = center_x - total_width // 2
    frame.paste(prev_button, (start_x, y_position), prev_button)
    frame.paste(play_button, (start_x + button_size + button_gap, y_position), play_button)
    frame.paste(next_button, (start_x + 2 * (button_size + button_gap), y_position), next_button)

    # Add Text
    draw.text((850, 100), "NOWPLAYING:", fill="white", font=now_playing_font, anchor="mm", align="center")
    draw.text((960, 173), SONG_NAME, fill="white", font=song_name_font, anchor="mm", align="center")
    draw.text((960, 255), "Artist: Tech Music", fill="white", font=artist_font, anchor="mm", align="center")

    return np.array(frame)


# Monitor Resources
def monitor_process(pid, interval, results, stop_event):
    process = psutil.Process(pid)
    total_cpu_time = 0
    total_memory_used = 0
    data_points = 0

    try:
        while not stop_event.is_set():
            if not process.is_running():
                break

            cpu_usage = process.cpu_percent(interval=0)
            memory_usage = process.memory_info().rss / (1024 * 1024)

            total_cpu_time += cpu_usage * (interval / 100)
            total_memory_used += memory_usage
            data_points += 1

            time.sleep(interval)
    except psutil.NoSuchProcess:
        pass

    avg_memory = total_memory_used / data_points if data_points else 0
    results["cpu_time"] = total_cpu_time
    results["avg_memory"] = avg_memory


if __name__ == "__main__":
    # Start Resource Monitoring
    start_time = time.time()
    pid = os.getpid()
    interval = 1
    monitor_results = {}
    stop_event = threading.Event()
    monitor_thread = threading.Thread(target=monitor_process, args=(pid, interval, monitor_results, stop_event))
    monitor_thread.start()

    # Preload background and shadow
    background_img = preprocess_background(BACKGROUND_IMAGE, blur=True)
    shadow = create_shadow(500)
    overlay_image = preprocess_background(BACKGROUND_IMAGE, target_size=(500, 500))

    # Load audio
    audio_clip = AudioFileClip(AUDIO_FILE)
    sample_rate = int(audio_clip.fps)
    audio_data = audio_clip.to_soundarray(fps=sample_rate).mean(axis=1)

    # Create video clip
    video_clip = VideoClip(lambda t: make_frame(t), duration=min(DURATION, audio_clip.duration))
    video_clip = video_clip.with_audio(audio_clip.subclipped(0, DURATION))
    video_clip.write_videofile(OUTPUT_FILE, fps=FPS, codec="libx264", audio_codec="aac", preset="ultrafast")

    # Stop Monitoring and Summarize
    end_time = time.time()
    stop_event.set()
    monitor_thread.join()
    total_cpu_used = monitor_results.get("cpu_time", 0)
    avg_memory_used = monitor_results.get("avg_memory", 0)

    print("\nResource Usage Summary:")
    print(f"Total time used: {end_time-start_time:.2f} seconds, parameters Duration and FPS: {DURATION, FPS}")
    print(f"Total CPU Time Used: {total_cpu_used:.2f} seconds")
    print(f"Average Memory Usage: {avg_memory_used:.2f} MB")
    

# ## V3. Was Bad no point.
# import os, psutil, time
# import threading
# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
# from moviepy.video.VideoClip import VideoClip
# from moviepy.audio.io.AudioFileClip import AudioFileClip
# from scipy.fft import rfft, rfftfreq
# from PIL import Image, ImageDraw, ImageFilter, ImageEnhance, ImageFont
# from scipy.ndimage import gaussian_filter1d

# # Configuration
# AUDIO_FILE = "audio.wav"
# BACKGROUND_IMAGE = "bg2.jpg"
# OUTPUT_FILE = "test2.mp4"
# DURATION = 60
# FPS = 23.967
# SONG_NAME = "Let Me Down Slowly"
# FRAME_SIZE = (1920, 1080)
# AUDIO_CHUNK_SIZE = 2048
# FFT_BANDS = 50  # Target number of bands, adjust if needed
# SPECTRUM_SMOOTHING = 2

# # Preload Static Resources
# now_playing_font = ImageFont.truetype("fonts/Staatliches-Regular.ttf", 60)
# song_name_font = ImageFont.truetype("fonts/PermanentMarker-Regular.ttf", 90)
# artist_font = ImageFont.truetype("fonts/Fondamento-Regular.ttf", 60)
# button_size = 100
# button_gap = 35
# next_button = Image.open("res/nextBtn.png").resize(
#     (button_size, button_size), Image.Resampling.LANCZOS
# ).convert("RGBA")
# play_button = Image.open("res/playBtn.png").resize(
#     (button_size, button_size), Image.Resampling.LANCZOS
# ).convert("RGBA")
# prev_button = Image.open("res/prevBtn.png").resize(
#     (button_size, button_size), Image.Resampling.LANCZOS
# ).convert("RGBA")
# background_img = None
# shadow = None

# # Preprocess Background Image
# def preprocess_background(image_path, target_size=FRAME_SIZE, blur=False):
#     img = Image.open(image_path).convert("RGB")  # Convert to RGB if no alpha needed
#     img_ratio = img.width / img.height
#     target_ratio = target_size[0] / target_size[1]

#     if img_ratio > target_ratio:
#         new_width = int(target_size[1] * img_ratio)
#         img = img.resize((new_width, target_size[1]), Image.Resampling.LANCZOS)
#         left = (img.width - target_size[0]) // 2
#         img = img.crop((left, 0, left + target_size[0], target_size[1]))
#     else:
#         new_height = int(target_size[0] / img_ratio)
#         img = img.resize((target_size[0], new_height), Image.Resampling.LANCZOS)
#         top = (img.height - target_size[1]) // 2
#         img = img.crop((0, top, target_size[0], top + target_size[1]))

#     if blur:
#         img = img.filter(ImageFilter.GaussianBlur(radius=20))
#         enhancer = ImageEnhance.Brightness(img)
#         img = enhancer.enhance(0.5)

#     return img

# # Generate Audio Spectrum with Chunking, rfft, and Dynamic Reshape
# def audio_spectrum_frame(t, audio_data, sample_rate):
#     start = int(t * sample_rate)
#     end = min(start + AUDIO_CHUNK_SIZE, len(audio_data))
#     audio_chunk = audio_data[start:end]

#     if len(audio_chunk) < AUDIO_CHUNK_SIZE:
#         audio_chunk = np.pad(audio_chunk, (0, AUDIO_CHUNK_SIZE - len(audio_chunk)))

#     fft_data = np.abs(rfft(audio_chunk))

#     if np.max(fft_data) == 0:
#         return np.zeros(FFT_BANDS)

#     spectrum = np.log1p(fft_data)

#     # Dynamically calculate reshape parameters
#     bands = min(FFT_BANDS, len(spectrum))  # Ensure bands do not exceed spectrum size
#     reshape_factor = len(spectrum) // bands
#     truncated_spectrum = spectrum[: bands * reshape_factor]

#     # Reshape based on the actual size of the spectrum
#     spectrum = truncated_spectrum.reshape(bands, reshape_factor).mean(axis=1)

#     spectrum = gaussian_filter1d(spectrum, sigma=SPECTRUM_SMOOTHING)
#     spectrum /= np.max(spectrum)

#     return spectrum

# # Create Shadow Once
# def create_shadow(overlay_size, blur_radius=15):
#     shadow = Image.new("RGBA", (overlay_size + 20, overlay_size + 20), (0, 0, 0, 0))
#     shadow_draw = ImageDraw.Draw(shadow)
#     shadow_draw.rectangle(
#         [(10, 10), (overlay_size + 10, overlay_size + 10)], fill=(0, 0, 0, 150)
#     )
#     return shadow.filter(ImageFilter.GaussianBlur(radius=blur_radius))

# # Visualization Function with NumPy-based Bar Drawing
# def make_frame(t):
#     spectrum = audio_spectrum_frame(t, audio_data, sample_rate)

#     # In-place modification of background_img
#     frame_np = np.array(background_img)
#     frame = Image.fromarray(frame_np)
#     draw = ImageDraw.Draw(frame)

#     # Add shadow and square overlay
#     overlay_image = preprocess_background(BACKGROUND_IMAGE, target_size=(500, 500))
#     frame.paste(shadow, (700, 290), shadow)
#     frame.paste(overlay_image, (710, 300))

#     # Optimized Bar Plot with NumPy
#     bar_width = 7
#     bar_gap = 6
#     num_bars = len(spectrum)
#     bar_plot_width = num_bars * (bar_width + bar_gap) - bar_gap
#     bar_x_start = 960 - bar_plot_width // 2
#     bar_plot_height = 94
#     bar_plot_y_position = 833

#     # Create a NumPy array for the bars and draw directly
#     bars_array = np.zeros((bar_plot_height, bar_plot_width, 3), dtype=np.uint8)
#     for i, bar_height in enumerate(spectrum * bar_plot_height):
#         if bar_height > 0:
#             bar_height_int = int(bar_height)
#             bars_array[
#                 -bar_height_int:,
#                 i * (bar_width + bar_gap) : i * (bar_width + bar_gap) + bar_width,
#                 :,
#             ] = 255  # White bars

#     # Convert the NumPy array to a PIL Image and paste it
#     bars_image = Image.fromarray(bars_array, "RGB")
#     frame.paste(bars_image, (bar_x_start, bar_plot_y_position - bar_plot_height))

#     # Add Buttons
#     total_width = 3 * button_size + 2 * button_gap
#     center_x = 960
#     y_position = 880
#     start_x = center_x - total_width // 2
#     frame.paste(prev_button, (start_x, y_position), prev_button)
#     frame.paste(play_button, (start_x + button_size + button_gap, y_position), play_button)
#     frame.paste(
#         next_button, (start_x + 2 * (button_size + button_gap), y_position), next_button
#     )

#     # Add Text
#     draw.text(
#         (850, 100),
#         "NOWPLAYING:",
#         fill="white",
#         font=now_playing_font,
#         anchor="mm",
#         align="center",
#     )
#     draw.text(
#         (960, 173),
#         SONG_NAME,
#         fill="white",
#         font=song_name_font,
#         anchor="mm",
#         align="center",
#     )
#     draw.text(
#         (960, 255),
#         "Artist: Tech Music",
#         fill="white",
#         font=artist_font,
#         anchor="mm",
#         align="center",
#     )

#     # Convert PIL Image back to NumPy array for MoviePy
#     return np.array(frame)

# # Monitor Resources
# def monitor_process(pid, interval, results, stop_event):
#     process = psutil.Process(pid)
#     total_cpu_time = 0
#     total_memory_used = 0
#     data_points = 0

#     try:
#         while not stop_event.is_set():
#             if not process.is_running():
#                 break

#             cpu_usage = process.cpu_percent(interval=0)
#             memory_usage = process.memory_info().rss / (1024 * 1024)

#             total_cpu_time += cpu_usage * (interval / 100)
#             total_memory_used += memory_usage
#             data_points += 1

#             time.sleep(interval)
#     except psutil.NoSuchProcess:
#         pass

#     avg_memory = total_memory_used / data_points if data_points else 0
#     results["cpu_time"] = total_cpu_time
#     results["avg_memory"] = avg_memory

# if __name__ == "__main__":
#     # Start Resource Monitoring
#     start_time = time.time()
#     pid = os.getpid()
#     interval = 1
#     monitor_results = {}
#     stop_event = threading.Event()
#     monitor_thread = threading.Thread(
#         target=monitor_process, args=(pid, interval, monitor_results, stop_event)
#     )
#     monitor_thread.start()

#     # Preload background and shadow
#     background_img = preprocess_background(BACKGROUND_IMAGE, blur=True)
#     shadow = create_shadow(500)

#     # Load audio
#     audio_clip = AudioFileClip(AUDIO_FILE)
#     sample_rate = int(audio_clip.fps)
#     audio_data = audio_clip.to_soundarray(fps=sample_rate).mean(axis=1)

#     # Create video clip
#     video_clip = VideoClip(
#         lambda t: make_frame(t), duration=min(DURATION, audio_clip.duration)
#     )
#     video_clip = video_clip.with_audio(audio_clip.subclipped(0, DURATION))

#     # Use optimized codec settings, experiment with hardware acceleration if available
#     video_clip.write_videofile(
#         OUTPUT_FILE,
#         fps=FPS,
#         codec="libx264",  # Or "h264_nvenc", "hevc_nvenc", etc. if available
#         audio_codec="aac",
#         preset="ultrafast",  # Experiment with "superfast", "veryfast"
#         threads=12,  # Adjust based on your CPU cores
#     )

#     # Stop Monitoring and Summarize
#     end_time = time.time()
#     stop_event.set()
#     monitor_thread.join()
#     total_cpu_used = monitor_results.get("cpu_time", 0)
#     avg_memory_used = monitor_results.get("avg_memory", 0)

#     print("\nResource Usage Summary:")
#     print(
#         f"Total time used: {end_time-start_time:.2f} seconds, parameters Duration and FPS: {DURATION, FPS}"
#     )
#     print(f"Total CPU Time Used: {total_cpu_used:.2f} seconds")
#     print(f"Average Memory Usage: {avg_memory_used:.2f} MB")