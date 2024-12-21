
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
