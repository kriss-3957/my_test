


In the project below make minimum chnges to display the message, "Your content is being generated. Please check back later.", right before the prompt entered by the user starts to be processed ?


File Structure


project/
│   └── templates/
│       ├── login.html
│       └── home.html
│
├── models.py
├── app.py
├── requirements.txt
├── generated_content
├── app.db
└── db.py





Files and Their Contents

1. app/models.py


# models.py
import os
from PIL import Image
import torch
import numpy as np
import imageio
from diffusers import StableDiffusionXLPipeline, DiffusionPipeline, DPMSolverMultistepScheduler

# Load models
image_pipe = StableDiffusionXLPipeline.from_pretrained(
    "segmind/SSD-1B",
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16"
).to("cuda")

video_pipe = DiffusionPipeline.from_pretrained(
    "damo-vilab/text-to-video-ms-1.7b",
    torch_dtype=torch.float16,
    variant="fp16"
)
video_pipe.scheduler = DPMSolverMultistepScheduler.from_config(video_pipe.scheduler.config)
video_pipe.enable_model_cpu_offload()
video_pipe.enable_vae_slicing()

def generate_image(prompt, user_id):
    user_dir = os.path.join("generated_content", user_id)  # Updated path
    os.makedirs(user_dir, exist_ok=True)
    image_paths = []
    image_urls = []

    try:
        # Generate images
        with torch.no_grad():
            images = image_pipe(prompt, num_images_per_prompt=2, num_inference_steps=10).images

        for i, img in enumerate(images):
            img_path = os.path.join(user_dir, f"image_{i+1}.png")
            img.save(img_path)
            image_paths.append(img_path)
            image_urls.append(f"/generated_content/{user_id}/image_{i+1}.png")  # Updated URL path

    finally:
        # Free up GPU memory
        torch.cuda.empty_cache()

    return images, image_urls


def generate_video(prompt, user_id, num_videos=2):
    user_dir = os.path.join("generated_content", user_id)  # Updated path
    os.makedirs(user_dir, exist_ok=True)

    video_urls = []
    video_frames_list = []

    # Generate videos one by one (sequentially) for debugging
    for index in range(num_videos):
        video_path = os.path.join(user_dir, f"video_{index+1}.mp4")

        try:
            # Generate video frames for each video (sequential)
            with torch.no_grad():
                video_frames = video_pipe(prompt, num_inference_steps=3).frames

            rgb_frames = [np.array(Image.fromarray(frame).convert("RGB")) for frame in video_frames]

            # Save video
            with imageio.get_writer(video_path, fps=10) as writer:
                for frame in rgb_frames:
                    writer.append_data(frame)

            # Add the video URL
            video_urls.append(f"/generated_content/{user_id}/video_{index+1}.mp4")  # Updated URL path
            video_frames_list.append(video_frames)

        except Exception as e:
            print(f"Error generating video {index+1}: {e}")

        finally:
            # Free up GPU memory after each video is processed
            torch.cuda.empty_cache()

    return video_frames_list, video_urls





2. app/app.py




import os
import sqlite3
from flask import Flask, render_template, request, redirect, url_for, session, send_from_directory
from pyngrok import ngrok
from db import init_db, save_user_data
from models import generate_image, generate_video  # Assuming model functions are moved to models.py

app = Flask(__name__)
app.secret_key = os.urandom(24)

# Set up ngrok authentication (replace 'your_ngrok_token' with your actual ngrok auth token)
ngrok.set_auth_token("2q7utm7avEDPmWFRQM0yd7p8mFg_482jGCmDPSE1YxtzBDbJt")

# Set up ngrok tunnel
public_url = ngrok.connect(5000).public_url
print(f"Flask app is publicly accessible at: {public_url}")

# Serve files from the generated_content directory
@app.route('/generated_content/<user_id>/<filename>')
def serve_generated_content(user_id, filename):
    content_dir = os.path.join("generated_content", user_id)  # Directory where files are saved
    return send_from_directory(content_dir, filename)

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        user_id = request.form.get("user_id")
        if user_id:
            session["user_id"] = user_id
            return redirect(url_for("home"))
    return render_template("login.html")

@app.route("/", methods=["GET", "POST"])
def home():
    if "user_id" not in session:
        return redirect(url_for("login"))

    user_id = session["user_id"]
    user_dir = os.path.join("generated_content", user_id)
    os.makedirs(user_dir, exist_ok=True)

    image_urls = []
    video_urls = []
    status = None

    if request.method == "POST":
        prompt = request.form.get("prompt")
        if prompt:
            # Save 'Processing' status
            status = "Processing"
            save_user_data(user_id, prompt, status, [], [])

            try:
                # Generate images
                _, image_urls = generate_image(prompt, user_id)

                # Generate videos
                _, video_urls = generate_video(prompt, user_id, num_videos=5)  # Generate 5 videos

                # Update status to 'Completed'
                status = "Completed"
                save_user_data(user_id, prompt, status, image_urls, video_urls)

            except Exception as e:
                # Log error and set status to 'Error'
                print(f"Error: {e}")
                status = "Error"
                save_user_data(user_id, prompt, status, [], [])

    return render_template("home.html", user_id=user_id, image_urls=image_urls, video_urls=video_urls, status=status)

if __name__ == "__main__":
    init_db()
    app.run(port=5000)


3.  app/db.py


import sqlite3
from datetime import datetime
import json

def init_db():
    conn = sqlite3.connect('app.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS user_data (
                    user_id TEXT PRIMARY KEY,
                    prompt TEXT,
                    video_paths TEXT,
                    image_paths TEXT,
                    status TEXT,
                    generated_at TIMESTAMP)''')
    conn.commit()
    conn.close()

def save_user_data(user_id, prompt, status, image_paths=None, video_paths=None):
    conn = sqlite3.connect('app.db')
    c = conn.cursor()

    # Serialize lists to JSON strings
    image_paths_json = json.dumps(image_paths) if image_paths else ""
    video_paths_json = json.dumps(video_paths) if video_paths else ""

    # Check if the user_id exists
    c.execute('SELECT * FROM user_data WHERE user_id = ?', (user_id,))
    existing_data = c.fetchone()

    if existing_data:
        c.execute('''UPDATE user_data SET prompt = ?, video_paths = ?, image_paths = ?, status = ?, generated_at = ? WHERE user_id = ?''',
                  (prompt, video_paths_json, image_paths_json, status, datetime.now(), user_id))
    else:
        c.execute('''INSERT INTO user_data (user_id, prompt, video_paths, image_paths, status, generated_at) VALUES (?, ?, ?, ?, ?, ?)''',
                  (user_id, prompt, video_paths_json, image_paths_json, status, datetime.now()))

    conn.commit()
    conn.close()

def get_user_data(user_id):
    conn = sqlite3.connect('app.db')
    c = conn.cursor()
    c.execute('SELECT * FROM user_data WHERE user_id = ?', (user_id,))
    data = c.fetchone()
    conn.close()

    if data:
        # Deserialize JSON strings back to lists
        image_paths = json.loads(data[3]) if data[3] else []
        video_paths = json.loads(data[2]) if data[2] else []
        return {
            "user_id": data[0],
            "prompt": data[1],
            "image_paths": image_paths,
            "video_paths": video_paths,
            "status": data[4],
            "generated_at": data[5]
        }
    return None






4. app/templates/login.html


<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Login</title>
    <style>
        body { font-family: Arial, sans-serif; background-color: #f4f4f9; color: #333; margin: 0; padding: 0; }
        h1 { color: #1520A6; }
        form { margin-top: 50px; text-align: center; }
        input[type="text"] { padding: 10px; font-size: 16px; width: 300px; border: 2px solid #ccc; border-radius: 5px; }
        input[type="submit"] { background-color: #1520A6; color: white; padding: 10px 20px; border: none; border-radius: 5px; cursor: pointer; font-size: 16px; }
        input[type="submit"]:hover { background-color: #757C80; }
    </style>
</head>
<body>
    <h1>Login</h1>
    <form method="POST">
        <label for="user_id">Enter User ID:</label><br>
        <input type="text" id="user_id" name="user_id" required><br><br>
        <input type="submit" value="Log In">
    </form>
</body>
</html>


5. app/templates/home.html



<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Text-to-Image and Video Generator</title>
    <style>
        body { font-family: Arial, sans-serif; background-color: #f4f4f9; color: #333; margin: 0; padding: 0; }
        h1 { text-align: center; color: #1520A6; }
        p { text-align: center; font-size: 18px; }
        form { text-align: center; margin: 20px; }
        input[type="text"] { padding: 10px; font-size: 16px; width: 300px; border: 2px solid #ccc; border-radius: 5px; }
        input[type="submit"] { background-color: #1520A6; color: white; padding: 10px 20px; border: none; border-radius: 5px; cursor: pointer; font-size: 16px; }
        input[type="submit"]:hover { background-color: #45a049; }
        .image-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 10px; }
        .image-grid img { width: 100%; height: auto; border: 2px solid #ccc; border-radius: 8px; }
        .video-gallery { display: flex; flex-wrap: wrap; gap: 20px; justify-content: center; }
        .video-gallery video { width: 300px; height: auto; border: 2px solid #ccc; border-radius: 8px; }
        .notification-message { color: #ff9800; font-weight: bold; text-align: center; padding: 20px; background-color: #fff3cd; border: 1px solid #ffeeba; border-radius: 5px; margin: 20px 0; }
    </style>
</head>
<body>
    <h1>Generate Images and Videos from Text Prompts</h1>
    <p>Logged in as: {{ user_id }}</p>

    <!-- Display the processing message if the status is "Processing" -->
    {% if status == "Processing" %}
        <div class="notification-message">Your content is being generated. Please check back later.</div>
    {% endif %}

    <form method="POST" enctype="multipart/form-data">
        <label for="prompt">Enter Prompt:</label><br>
        <input type="text" id="prompt" name="prompt" required><br><br>
        <input type="submit" value="Generate Images and Videos">
    </form>

    {% if image_urls %}
        <h2>Generated Images:</h2>
        <div class="image-grid">
            {% for url in image_urls %}
                <img src="{{ url }}" alt="Generated Image">
            {% endfor %}
        </div>
    {% endif %}

    {% if video_urls %}
        <h2>Generated Videos:</h2>
        <div class="video-gallery">
            {% for url in video_urls %}
                <video controls>
                    <source src="{{ url }}" type="video/mp4">
                </video>
            {% endfor %}
        </div>
    {% endif %}
</body>
</html>





6. requirements.txt



diffusers==0.23.0
transformers>=4.44.0
peft>=0.4.0
accelerate>=0.21.0
flask
torch
torchvision
pyngrok
safetensors
imageio[ffmpeg]
plyer
huggingface_hub==0.25.0











