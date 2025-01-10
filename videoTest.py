## Optimized Version2.0 for better performance.
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
AUDIO_FILE = "./Tests/audio2.mp4" #Will also be passed in by parameters from api request (for now the name will be in the local folder of the server.).
BACKGROUND_IMAGE = "bg2.jpg" #Will also be passed in by parameters from api request (for now the name will be in the local folder of the server.).
OUTPUT_FILE = "test6.mp4" # Need to be uploaded to a speicifed youtube drive.
DURATION = None # Later Filled In.
FPS = 1
SONG_NAME = "AI PIRATES AND BOOK BYTES\n" #Will Be later filled in by parameters from api Request

#Load fonts
fonts = {
    "now_playing_font": {
        "folder": os.path.join(".", "pastTestRes", "Fondamento,Permanent_Marker,Staatliches", "Staatliches"),
        "filename": "Staatliches-Regular.ttf",
        "size": 60,
    },
    "song_name_font": {
        "folder": os.path.join(".", "pastTestRes", "Fondamento,Permanent_Marker,Staatliches", "Permanent_Marker"),
        "filename": "PermanentMarker-Regular.ttf",
        "size": 90,
    },
    "artist_font": {
        "folder": os.path.join(".", "pastTestRes", "Fondamento,Permanent_Marker,Staatliches", "Fondamento"),
        "filename": "Fondamento-Regular.ttf",
        "size": 60,
    },
}

# Load fonts
font_objects = {}
for font_name, font_details in fonts.items():
    font_path = os.path.join(font_details["folder"], font_details["filename"])
    if os.path.exists(font_path):
        try:
            font_objects[font_name] = ImageFont.truetype(font_path, font_details["size"])
            print(f"{font_name} loaded successfully from {font_path}")
        except Exception as e:
            print(f"Error loading {font_name} from {font_path}: {e}")
    else:
        print(f"Font file not found: {font_path}")

# Preload Static Resources
now_playing_font = font_objects.get("now_playing_font")
song_name_font = font_objects.get("song_name_font")
artist_font = font_objects.get("artist_font")
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
    draw.text((960, 223), SONG_NAME, fill="white", font=song_name_font, anchor="mm", align="center") #173 before
    draw.text((960, 255), "Artist: TechNews", fill="white", font=artist_font, anchor="mm", align="center")

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
    DURATION = audio_clip.duration
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
