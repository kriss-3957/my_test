# import sqlite3
# from datetime import datetime

# def init_db():
#     conn = sqlite3.connect('app.db')
#     c = conn.cursor()
#     c.execute('''CREATE TABLE IF NOT EXISTS user_data (
#                     user_id TEXT PRIMARY KEY,
#                     prompt TEXT,
#                     video_paths TEXT,
#                     image_paths TEXT,
#                     status TEXT,
#                     generated_at TIMESTAMP)''')
#     conn.commit()
#     conn.close()

# def save_user_data(user_id, prompt, status, image_paths=None, video_paths=None):
#     conn = sqlite3.connect('app.db')
#     c = conn.cursor()
    
#     # Check if the user_id exists
#     c.execute('SELECT * FROM user_data WHERE user_id = ?', (user_id,))
#     existing_data = c.fetchone()

#     if existing_data:
#         c.execute('''UPDATE user_data SET prompt = ?, video_paths = ?, image_paths = ?, status = ?, generated_at = ? WHERE user_id = ?''',
#                   (prompt, ','.join(video_paths) if video_paths else "", ','.join(image_paths) if image_paths else "", status, datetime.now(), user_id))
#     else:
#         c.execute('''INSERT INTO user_data (user_id, prompt, video_paths, image_paths, status, generated_at) VALUES (?, ?, ?, ?, ?, ?)''',
#                   (user_id, prompt, ','.join(video_paths) if video_paths else "", ','.join(image_paths) if image_paths else "", status, datetime.now()))

#     conn.commit()
#     conn.close()


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

