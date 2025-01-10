#!/usr/bin/python
import httplib2
import os
import random
import sys
import time
import json

from apiclient.discovery import build
from apiclient.errors import HttpError
from apiclient.http import MediaFileUpload
from oauth2client.client import flow_from_clientsecrets
from oauth2client.file import Storage
from oauth2client.tools import argparser, run_flow

from flask import Flask, request, jsonify
import requests 
import numpy as np
from moviepy.video.VideoClip import VideoClip
from moviepy.audio.io.AudioFileClip import AudioFileClip
from scipy.fft import fft
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance, ImageFont
from scipy.ndimage import gaussian_filter1d
import uuid
from datetime import date
from argparse import ArgumentParser


# -------------------------------------------------------------------------
# YouTube Upload Code (Mostly Untouched)
# -------------------------------------------------------------------------

httplib2.RETRIES = 1
MAX_RETRIES = 10
RETRIABLE_EXCEPTIONS = (httplib2.HttpLib2Error, IOError)
RETRIABLE_STATUS_CODES = [500, 502, 503, 504]
CLIENT_SECRETS_FILE = "client_secrets.json"
YOUTUBE_UPLOAD_SCOPE = "https://www.googleapis.com/auth/youtube.upload"
YOUTUBE_API_SERVICE_NAME = "youtube"
YOUTUBE_API_VERSION = "v3"
MISSING_CLIENT_SECRETS_MESSAGE = """
WARNING: Please configure OAuth 2.0

To make this sample run you will need to populate the client_secrets.json file
found at:

   %s

with information from the API Console
https://console.cloud.google.com/

For more information about the client_secrets.json file format, please visit:
https://developers.google.com/api-client-library/python/guide/aaa_client_secrets
""" % os.path.abspath(os.path.join(os.path.dirname(__file__),
                                   CLIENT_SECRETS_FILE))

VALID_PRIVACY_STATUSES = ("public", "private", "unlisted")


def get_authenticated_service(args):
    flow = flow_from_clientsecrets(
        CLIENT_SECRETS_FILE,
        scope=YOUTUBE_UPLOAD_SCOPE,
        message=MISSING_CLIENT_SECRETS_MESSAGE
    )

    storage = Storage("%s-oauth2.json" % sys.argv[0])
    credentials = storage.get()

    if credentials is None or credentials.invalid:
        credentials = run_flow(flow, storage, args)

    return build(YOUTUBE_API_SERVICE_NAME, YOUTUBE_API_VERSION,
                 http=credentials.authorize(httplib2.Http()))


def resumable_upload(insert_request):
    response = None
    error = None
    retry = 0
    while response is None:
        try:
            print("Uploading file...")
            status, response = insert_request.next_chunk()
            if response is not None:
                if 'id' in response:
                    print("Video id '%s' was successfully uploaded." % response['id'])
                else:
                    exit("The upload failed with an unexpected response: %s" % response)
        except HttpError as e:
            if e.resp.status in RETRIABLE_STATUS_CODES:
                error = "A retriable HTTP error %d occurred:\n%s" % (e.resp.status, e.content)
            else:
                raise
        except RETRIABLE_EXCEPTIONS as e:
            error = "A retriable error occurred: %s" % e

        if error is not None:
            print(error)
            retry += 1
            if retry > MAX_RETRIES:
                exit("No longer attempting to retry.")

            max_sleep = 2 ** retry
            sleep_seconds = random.random() * max_sleep
            print("Sleeping %f seconds and then retrying..." % sleep_seconds)
            time.sleep(sleep_seconds)


def initialize_upload(youtube, options):
    """
    The original function from your upload script.
    We'll call this after generating the video file in the Flask route.
    """
    tags = None
    if options.keywords:
        tags = options.keywords.split(",")

    body = dict(
        snippet=dict(
            title=options.title,
            description=options.description,
            tags=tags,
            categoryId=options.category
        ),
        status=dict(
            privacyStatus=options.privacyStatus,
            selfDeclaredMadeForKids=False
        )
    )

    insert_request = youtube.videos().insert(
        part=",".join(body.keys()),
        body=body,
        media_body=MediaFileUpload(options.file, chunksize=-1, resumable=True)
    )

    # Actually perform the upload
    resumable_upload(insert_request)


def create_mock_args(file_path):
    mock_argparser = ArgumentParser()
    mock_argparser.add_argument("--file", required=True, help="Video file to upload")
    mock_argparser.add_argument("--title", help="Video title", default="Test Title")
    mock_argparser.add_argument("--description", help="Video description", default="Test Description")
    mock_argparser.add_argument("--category", default="22", help="Numeric video category.")
    mock_argparser.add_argument("--keywords", help="Video keywords, comma separated", default="")
    mock_argparser.add_argument("--privacyStatus", choices=VALID_PRIVACY_STATUSES, default="public")
    return mock_argparser.parse_args(["--file", file_path])



# -------------------------------------------------------------------------
# Flask App / Video Generation Code
# -------------------------------------------------------------------------

app = Flask(__name__)

# Configure folders (same as original)
UPLOAD_FOLDER = './downloads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

OUTPUT_FOLDER = './output'
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER

DOWNLOAD_FOLDER = './downloads'
os.makedirs(DOWNLOAD_FOLDER, exist_ok=True)
app.config['DOWNLOAD_FOLDER'] = DOWNLOAD_FOLDER

# Load fonts
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

GOAPI_URL = "https://api.goapi.ai/api/v1/task"
GOAPI_KEY = ''  # Replace with your actual API key

@app.route('/process', methods=['POST'])
def process_and_upload_video():
    """
    This route:
      1) Generates the video from audio + background image (as before).
      2) Uploads the resulting video to YouTube.
      3) Returns JSON with status and possibly the YouTube video ID.
    """
    try:
        # Step 1: Parse form data
        UUID_value = uuid.uuid4()
        audio_file_url = request.form.get('audio_file')
        background_image_url = request.form.get('background_image')
        song_name = request.form.get('song_name', 'Unknown_Song')
        song_name = song_name.strip() # to remove leading/trailing spaces and newlines
        # Now we also want the video description from the request
        # while everything else is hard-coded or derived.
        video_description = request.form.get('description', 'Default Description')

        if not audio_file_url or not background_image_url:
            return jsonify({"error": "Both audio_file and background_image are required"}), 400

        # Where final video is saved before upload
        output_file = os.path.join(app.config['OUTPUT_FOLDER'], f"{song_name}_{UUID_value}.mp4")

        # Step 2: Download files

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
    
        try:
            audio_file_path = download_file(audio_file_url, f"{song_name}_{UUID_value}.mp3")
            background_image_path = download_file(background_image_url, f"{song_name}_{UUID_value}.jpg")
        except Exception as e:
            return jsonify({"error": str(e) + "... Something went wrong while downloading"}), 511

        if not os.path.exists(audio_file_path) or not os.path.exists(background_image_path):
            return jsonify({"error": "one or more files were not found in dir."}), 512

        # Step 3: Preprocessing functions
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

        def create_shadow(overlay_size, blur_radius=15):
            shadow_img = Image.new("RGBA", (overlay_size + 20, overlay_size + 20), (0, 0, 0, 0))
            shadow_draw = ImageDraw.Draw(shadow_img)
            shadow_draw.rectangle(
                [(10, 10), (overlay_size + 10, overlay_size + 10)],
                fill=(0, 0, 0, 150)
            )
            return shadow_img.filter(ImageFilter.GaussianBlur(radius=blur_radius))

        def audio_spectrum_frame(t, audio_data_array, sr):
            n = len(audio_data_array)
            start = int(t * sr) % n
            end = min(start + 2048, n)
            audio_chunk = audio_data_array[start:end]

            fft_data = np.abs(fft(audio_chunk))[:1024]
            if np.max(fft_data) == 0:
                return np.zeros(200)

            spectrum = fft_data[:200]
            spectrum = np.log1p(spectrum)
            spectrum = spectrum.reshape(50, -1).mean(axis=1)
            spectrum = gaussian_filter1d(spectrum, sigma=2)
            spectrum /= np.max(spectrum)

            return spectrum

        def make_frame(t):
            """
            The main function for building each frame of the video visualization.
            """
            spectrum_data = audio_spectrum_frame(t, audio_data, sample_rate)
            frame = background_img.copy()
            draw = ImageDraw.Draw(frame)

            # Square overlay
            frame.paste(shadow, (700, 290), shadow)
            frame.paste(overlay_image, (710, 300))

            # Spectrum bars
            bar_width = 7
            bar_gap = 6
            num_bars = len(spectrum_data)
            bar_plot_width = num_bars * (bar_width + bar_gap) - bar_gap
            bar_x_start = 960 - bar_plot_width // 2
            bar_plot_height = 94
            bar_plot_y_position = 833

            for i, bar_height in enumerate(spectrum_data * bar_plot_height):
                if bar_height < 1e-3:
                    continue
                x0 = bar_x_start + i * (bar_width + bar_gap)
                y0 = bar_plot_y_position - bar_height
                x1 = x0 + bar_width
                y1 = bar_plot_y_position
                draw.rectangle([x0, y0, x1, y1], fill="white")

            # Buttons
            total_width = 3 * button_size + 2 * button_gap
            center_x = 960
            y_position = 880
            start_x = center_x - total_width // 2

            frame.paste(prev_button, (start_x, y_position), prev_button)
            frame.paste(play_button, (start_x + button_size + button_gap, y_position), play_button)
            frame.paste(next_button, (start_x + 2 * (button_size + button_gap), y_position), next_button)

            # Text
            draw.text((850, 100), "NOWPLAYING:", fill="white", font=now_playing_font, anchor="mm", align="center")
            draw.text((960, 173), song_name, fill="white", font=song_name_font, anchor="mm", align="center")
            draw.text((960, 255), "Artist: Tech Music", fill="white", font=artist_font, anchor="mm", align="center")

            return np.array(frame)

        # Step 4: Generate the video
        global background_img, shadow, overlay_image
        background_img = preprocess_background(background_image_path, blur=True)
        shadow = create_shadow(500)
        overlay_image = preprocess_background(background_image_path, target_size=(500, 500))

        audio_clip = AudioFileClip(audio_file_path)
        DURATION = audio_clip.duration
        sample_rate = int(audio_clip.fps)
        audio_data = audio_clip.to_soundarray(fps=sample_rate).mean(axis=1)

        video_clip = VideoClip(lambda t: make_frame(t), duration=min(DURATION, audio_clip.duration))
        video_clip = video_clip.with_audio(audio_clip.subclipped(0, DURATION))
        video_clip.write_videofile(output_file, fps=23.967, codec="libx264", audio_codec="aac", preset="ultrafast")

        # ---------------------------------------------------------------------
        # Step 5: Upload the newly generated video to YouTube
        #         We adapt the original code by building an "args" object
        #         This approach uses the same logic as your original script.
        # ---------------------------------------------------------------------
        today = date.today()
        year = today.year
        month = today.strftime("%b") 
        day = today.day
        
        song_name_Title = f"{song_name} | {month} {day} {year} | Tech News Music"
        try:
            # We'll build an ArgParse-like object to replicate the original usage.
            # Hard-coding category 28, keywords, privacyStatus, etc.
            class UploadArgs:
                def __init__(self, file, title, description, category, keywords, privacyStatus):
                    self.file = file
                    self.title = title
                    self.description = description
                    self.category = category
                    self.keywords = keywords
                    self.privacyStatus = privacyStatus

            hashtags = "#tech, #music, #breakingnews, #news, #exciting, #parody\n"
            formatted_description = f"{hashtags}{video_description}"

            # Our "options" mirror the original script's approach
            # Hard-code category, keywords, privacy, etc. as requested
            upload_args = UploadArgs(
                file=output_file,
                title=song_name_Title,                       # from user request (song_name)
                description=formatted_description,         # from user request (description)
                category="28",                         # Hard-coded (Science & Technology)
                keywords="Tech,Music,Breaking News,Exciting,Parody",   # Hard-coded example
                privacyStatus="public"                 # Hard-coded
            )

            # Use the existing argparser to set up the storage filename
            # This is a quick trick to keep your code mostly the same:
            # The only real reason for this is so get_authenticated_service()
            # can store or read from "upload.py-oauth2.json"
            if not argparser._actions:  # quick check to avoid repeated definitions
                argparser.add_argument("--file", required=True, help="Video file to upload")
                argparser.add_argument("--title", help="Video title", default="Test Title")
                argparser.add_argument("--description", help="Video description", default="Test Description")
                argparser.add_argument("--category", default="22",
                                       help="Numeric video category.")
                argparser.add_argument("--keywords", help="Video keywords, comma separated", default="")
                argparser.add_argument("--privacyStatus", choices=VALID_PRIVACY_STATUSES,
                                       default=VALID_PRIVACY_STATUSES[0], help="Video privacy status.")

            # We'll do a minimal parse_args call just to produce an 'args'
            # object that "get_authenticated_service()" expects.
            # But we won't actually use those CLI args for real, we already
            # have a lot of it in 'upload_args'.
            tmp_args = create_mock_args(upload_args.file)

            # Actually get the youtube service, initialize and perform upload
            youtube = get_authenticated_service(tmp_args)
            initialize_upload(youtube, upload_args)
            print(f"Video {song_name}_{UUID_value} successfully uplaoded for {song_name_Title}.")
            # If we got here, it likely uploaded successfully.
            return jsonify({
                "status": "success",
                "song_name": song_name,
                "video_description": video_description,
                "note": "Video generation and upload completed successfully."
            }), 200

        except HttpError as e:
            return jsonify({
                "status": "failure",
                "error": f"HTTP error {e.resp.status} occurred:\n{e.content}"
            }), 500

    except Exception as e:
        return jsonify({"error": str(e) + "Something went wrong in the process."}), 540

@app.route('/generate_music', methods=['POST'])
def generate_music():
    """
    API endpoint to generate music using GoAPI.
    Accepts multipart/form-data with fields: title, lyrics, tags.
    Returns only the task_id if successful, otherwise an error message.
    """
    try:
        # Get data from the request
        title = request.form.get('title', '').strip()
        lyrics = request.form.get('lyrics', '').strip()
        tags = request.form.get('tags', '').strip()

        # Validate that required fields are present
        if not title or not lyrics or not tags:
            return jsonify({"error": "Missing required fields: title, lyrics, or tags"}), 400

        # Construct the payload for the GoAPI request
        payload = {
            "model": "music-s",
            "task_type": "generate_music_custom",
            "input": {
                "title": title,
                "prompt": lyrics,
                "tags": tags,
                "negative_tags": "metal"
            },
            "config": {
                "service_mode": "public"
            }
        }

        # Prepare headers for the GoAPI request
        headers = {
            'X-API-KEY': GOAPI_KEY,
            'Content-Type': 'application/json'
        }

        # Make the request to the GoAPI
        response = requests.post(GOAPI_URL, headers=headers, data=json.dumps(payload))

        # Check for errors in the GoAPI response
        response.raise_for_status()

        # Parse the GoAPI response
        response_data = response.json()

        # Check if the task was successful based on the status and return only the task_id
        if response_data.get("code") == 200 and response_data.get("message") == "success" and response_data.get("data",{}).get("status") == "pending":
          task_id = response_data.get("data", {}).get("task_id")
          if task_id:
            return jsonify({"task_id": task_id}), 200
          else:
            return jsonify({"error": "Task created but missing task_id"}), 500
        else:
            return jsonify({"error": "GoAPI request was not successful", "details": response_data.get("message")}), response_data.get("code")

    except requests.exceptions.RequestException as e:
        return jsonify({"error": "Error communicating with GoAPI", "details": str(e)}), 500
    except Exception as e:
        return jsonify({"error": "Internal server error", "details": str(e)}), 500

@app.route('/get_music_data/<task_id>', methods=['GET'])
def get_music_data(task_id):
    """
    API endpoint to retrieve music data from GoAPI using the task_id.
    Downloads audio and image files, and returns a JSON response
    with the file paths if successful, otherwise an error message.
    """
    def download_file(url, filepath):
        """
        Download a file from the given URL and save it to the specified filepath.
        """
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Raise an error for HTTP issues

        # Write the file to the specified filepath
        with open(filepath, 'wb') as f:
            for chunk in response.iter_content(chunk_size=1024):
                f.write(chunk)

    try:
        # Delay for 200 seconds
        time.sleep(200)

        # Prepare headers for the GoAPI request
        headers = {
            'X-API-KEY': GOAPI_KEY,
            'Content-Type': 'application/json'
        }

        # Fetch task data from GoAPI
        response = requests.get(f"{GOAPI_URL}/{task_id}", headers=headers)
        response.raise_for_status()  # Raise error for HTTP issues

        # Parse the response data
        response_data = response.json()

        # Check if the task status is "completed"
        if response_data.get("code") == 200 and response_data.get("data", {}).get("status") == "completed":
            clips_data = response_data.get("data", {}).get("output", {}).get("clips", {})
            result = {}

            # Directory to save downloaded files
            today = date.today()
            year = today.year
            month = today.strftime("%b") 
            day = today.day
            fetchfilename = f"{task_id}_{month}_{day}_{year}"
            task_download_dir = os.path.join(app.config['DOWNLOAD_FOLDER'], fetchfilename)
            os.makedirs(task_download_dir, exist_ok=True)

            # Extract and download audio and image files
            for i, (clip_id, clip_info) in enumerate(clips_data.items(), start=1):
                audio_url = clip_info.get("audio_url")
                image_url = clip_info.get("image_large_url")

                # Download and save audio file
                if audio_url:
                    audio_filepath = os.path.join(task_download_dir, f"clip{i}_audio.mp3")
                    try:
                        download_file(audio_url, audio_filepath)
                        result[f"clip{i}audiourl"] = audio_url
                        result[f"clip{i}audiopath"] = audio_filepath
                    except Exception as e:
                        app.logger.error(f"Error downloading audio for clip {i}: {e}")

                # Download and save image file
                if image_url:
                    image_filepath = os.path.join(task_download_dir, f"clip{i}_image.jpg")
                    try:
                        download_file(image_url, image_filepath)
                        result[f"clip{i}imageurl"] = image_url
                        result[f"clip{i}imagepath"] = image_filepath
                    except Exception as e:
                        app.logger.error(f"Error downloading image for clip {i}: {e}")

            return jsonify(result), 200
        elif response_data.get("data", {}).get("status") == "pending":
            return jsonify({"error": "Task is still pending. Try again later."}), 503
        else:
            return jsonify({"error": "Task failed or unexpected response from GoAPI.", "details": response_data.get("message")}), 500

    except requests.exceptions.RequestException as e:
        return jsonify({"error": "Failed to communicate with GoAPI.", "details": str(e)}), 500
    except Exception as e:
        return jsonify({"error": "Internal server error.", "details": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000)
