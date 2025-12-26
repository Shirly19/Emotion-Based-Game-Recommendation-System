from flask import Flask, render_template, Response, jsonify, request
from camera import VideoCamera, get_current_emotion, get_emotion_history
import time
import pymysql
import uuid
import json

app = Flask(__name__)

def create_table():
    connection = pymysql.connect(
        host='localhost',
        user='root',
        password='',
        database='emogame'
    )
    with connection.cursor() as cursor:
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS emotion_game_logs (
                id INT AUTO_INCREMENT PRIMARY KEY,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                session_id VARCHAR(255),
                initial_emotion VARCHAR(50),
                game_played VARCHAR(100),
                duration_seconds INT DEFAULT 0,
                final_emotion VARCHAR(50),
                mood_changes JSON
            )
        """)
    connection.close()

def log_emotion_game(session_id, initial_emotion, game_played, duration=0, mood_changes=[]):
    connection = pymysql.connect(
        host='localhost',
        user='root',
        password='',
        database='emogame'
    )
    with connection.cursor() as cursor:
        final_emotion = mood_changes[-1] if mood_changes else initial_emotion
        cursor.execute("""
            INSERT INTO emotion_game_logs (session_id, initial_emotion, game_played, duration_seconds, final_emotion, mood_changes)
            VALUES (%s, %s, %s, %s, %s, %s)
        """, (session_id, initial_emotion, game_played, duration, final_emotion, json.dumps(mood_changes)))
    connection.commit()
    connection.close()

# ------------------------------
# Route: Homepage
# ------------------------------
@app.route('/')
def index():
    return render_template('index.html')

# ------------------------------
# Route: Video Feed for Webcam
# ------------------------------
def gen(camera):
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen(VideoCamera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# ------------------------------
# Route: Current Detected Emotion
# ------------------------------
@app.route('/get_emotion')
def get_emotion():
    emotion = get_current_emotion()
    return jsonify({'emotion': emotion})

# ------------------------------
# Route: Emotion History
# ------------------------------
@app.route('/get_emotion_history')
def emotion_history():
    history = get_emotion_history()
    return jsonify({'history': history})

# ------------------------------
# Route: Bot Suggestion
# ------------------------------
@app.route('/get_bot_suggestion')
def get_bot_suggestion():
    history = get_emotion_history()
    if not history:
        suggestion = "Let's play a game improve how you feel!"
    else:
        last = history[-1]
        suggestions = {
            "Happy": "Keep up the good mood! Try to challenge yourself with a new game.",
            "Sad": "It's okay to feel sad. Try the uplifting game and see if your mood improves!",
            "Angry": "Take a deep breath. The calming game might help you relax.",
            "Surprised": "Surprises are fun! Enjoy the game and see what's next.",
            "Fearful": "You're safe here. Try the soothing game to feel better.",
            "Disgusted": "Let's distract you with a fun game!",
            "Neutral": "Feeling balanced? Try a relaxing game or explore a new one!"
        }
        suggestion = suggestions.get(last, "Let's play a game to see how you feel!")
    return jsonify({'suggestion': suggestion})

# ------------------------------
# Route: Play Game (Based on Emotion)
# ------------------------------
@app.route('/play_game')
def play_game():
    emotion = request.args.get('emotion', 'neutral').lower()

    valid_emotions = ['happy', 'sad', 'angry', 'fearful', 'disgusted', 'surprised', 'neutral']
    if emotion not in valid_emotions:
        emotion = 'neutral'

    history = get_emotion_history()
    session_id = uuid.uuid4().hex
    log_emotion_game(session_id, emotion, emotion, 0, history)

    # Renders from templates/games/{emotion}_game.html
    return render_template(f'games/{emotion}_game.html')

# ------------------------------
# Main Entry Point
# ------------------------------
import socket

if __name__ == '__main__':
    create_table()
    app.debug = True
    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    print(f" * Running on http://{local_ip}:5000/ (Press CTRL+C to quit)")
    app.run(host='0.0.0.0', port=5000)
