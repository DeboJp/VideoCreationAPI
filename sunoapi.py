from flask import Flask, request, jsonify
import requests
import json

app = Flask(__name__)

GOAPI_URL = "https://api.goapi.ai/api/v1/task"
GOAPI_KEY = ''

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

if __name__ == '__main__':
    app.run(debug=True)