# from flask import Flask, request, jsonify

# app = Flask(__name__)

# @app.route('/', methods=['POST'])
# def hello_world():
#     # grab data
#     data = request.get_json()
    
#     # Check if the 'text' key exists and its value is "go for it"
#     if data and data.get('text') == "go for it":
#         return jsonify(message="Hello, World!"), 200
#     else:
#         return jsonify(error="Invalid input. Please send {'text': 'go for it'} in the request body."), 400

# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=3000)

from flask import Flask, request, jsonify, send_file
import requests 
import os
import numpy as np
from moviepy.video.VideoClip import VideoClip
from moviepy.audio.io.AudioFileClip import AudioFileClip
from scipy.fft import fft
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance, ImageFont
# from werkzeug.utils import secure_filename
from scipy.ndimage import gaussian_filter1d
import uuid


app = Flask(__name__)

# Configure upload folder
UPLOAD_FOLDER = './downloads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

OUTPUT_FOLDER = './output'
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER

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

@app.route('/process', methods=['POST'])
def process_audio_video():
    try:
        # Parse files and form data
        UUID = uuid.uuid4()
        audio_file_url = request.form.get('audio_file')
        background_image_url = request.form.get('background_image')
        song_name = request.form.get('song_name', 'Unknown Song')
        output_file = os.path.join(app.config['OUTPUT_FOLDER'], f"{song_name}_{UUID}.mp4")

        if not audio_file_url or not background_image_url:
            return jsonify({"error": "Both audio_file and background_image are required"}), 400

        # Definition to download audio and image files
        def download_file(url, filename):
            response = requests.get(url, stream=True)
            if response.status_code == 200:
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                with open(file_path, 'wb') as f:
                    for chunk in response.iter_content(1024):
                        f.write(chunk)
                return file_path
            else:
                raise Exception(f"Failed to download file from {url}")

        # Downloads files from url.
        try:
            audio_file_path = download_file(audio_file_url, f"{song_name}_{UUID}.mp3")
            background_image_path = download_file(background_image_url, f"{song_name}_{UUID}.jpg")
        except Exception as e:
            return jsonify({"error": str(e) + "... Something went wrong while downloading"}), 511
        # Checks Files exist.
        if not os.path.exists(audio_file_path) or not os.path.exists(background_image_path):
            return jsonify({"error": "one or more files were not found in dir."}), 512


        # Preload Background
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

        # Spectrum Frame Generator
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
            draw.text((960, 173), song_name, fill="white", font=song_name_font, anchor="mm", align="center")
            draw.text((960, 255), "Artist: Tech Music", fill="white", font=artist_font, anchor="mm", align="center")

            return np.array(frame)

        ## MAIN:
        # Load background and audio
        background_img = preprocess_background(background_image_path, blur=True)
        shadow = create_shadow(500)
        overlay_image = preprocess_background(background_image_path, target_size=(500, 500))

        audio_clip = AudioFileClip(audio_file_path)
        DURATION = audio_clip.duration
        sample_rate = int(audio_clip.fps)
        audio_data = audio_clip.to_soundarray(fps=sample_rate).mean(axis=1)

        # Create Video
        video_clip = VideoClip(lambda t: make_frame(t), duration=min(DURATION, audio_clip.duration))
        video_clip = video_clip.with_audio(audio_clip.subclipped(0, DURATION))
        video_clip.write_videofile(output_file, fps=23.967, codec="libx264", audio_codec="aac", preset="ultrafast")

        return send_file(output_file, as_attachment=True)

    except Exception as e:
        return jsonify({"error": str(e) + "Something went wrong in whole process"}), 540

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000)

# CURL USAGE:
# curl --location --request POST 'http://127.0.0.1:3000/process' \
# --header 'Content-Type: application/x-www-form-urlencoded' \
# --data-urlencode 'audio_file=https://cdn1.suno.ai/9ef53071-ef0e-4d23-b402-fc2f04dbd3bb.mp3' \
# --data-urlencode 'background_image=https://cdn1.suno.ai/image_large_9ef53071-ef0e-4d23-b402-fc2f04dbd3bb.jpeg' \
# --data-urlencode 'song_name=Hiya HEE Song' \
# --output result.mp4
# ngrok http http://127.0.0.1:3000 -> to expose server.