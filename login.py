import oracledb
import subprocess
import webbrowser
import time
import threading
import requests
from flask import Flask, render_template, request, redirect, flash
import os
import secrets
from cryptography.fernet import Fernet
import json
from datetime import datetime


ORACLE_DSN = "localhost:1521/ORCLPDB"   # ✅ Change if needed
ORACLE_USER = "jay_user"
ORACLE_PASS = "oci123456"


def get_public_ip():
    return requests.get('https://api.ipify.org').text


public_ip = get_public_ip()
print("Public IP:", public_ip)

app = Flask(__name__)
app.secret_key = 'TFG@2023'

RECAPTCHA_SECRET_KEY = "6LfkXZQrAAAAAKIosm2eIEKwzw6AmblfqY8NDb3D"   # from Google
RECAPTCHA_SITE_KEY = "6LfkXZQrAAAAANLCHFVeHYym1YO0F_6aa9mcbziC"     # for template use

# -------------------------------
# ORACLE CONNECTION
# -------------------------------
conn = oracledb.connect(
    user=ORACLE_USER,
    password=ORACLE_PASS,
    dsn=ORACLE_DSN
)
cursor = conn.cursor()
print("Oracle cursor connected ✅")

# -------------------------------
# FETCH CREDENTIALS
# -------------------------------


def get_user_credentials():
    cursor.execute("SELECT username, password FROM login_credentials")
    result = cursor.fetchall()
    return {row[0]: row[1] for row in result}

# -------------------------------
# ROUTES
# -------------------------------


@app.route('/')
def login_page():
    return render_template('login.html', site_key=RECAPTCHA_SITE_KEY)


# Generate once and store securely; here hardcoded for demo
fernet_key = b'Sv_cBtT5H5i_fv3sPvRrAe_2z6WRnqbmq-rmfxUyiGQ='  # Fernet.generate_key()
cipher_suite = Fernet(fernet_key)


@app.route('/login', methods=['POST'])
def login():
    username = request.form.get('username')
    password = request.form.get('password')
    recaptcha_response = request.form.get('g-recaptcha-response')

    # Step 1: Verify reCAPTCHA
    captcha_verify = requests.post(
        "https://www.google.com/recaptcha/api/siteverify",
        data={
            'secret': RECAPTCHA_SECRET_KEY,
            'response': recaptcha_response
        }
    )
    result = captcha_verify.json()
    if not result.get('success'):
        flash("reCAPTCHA verification failed ❌")
        return redirect('/')

    # Step 2: Validate username/password
    credentials = get_user_credentials()
    if username in credentials and credentials[username] == password:
        flash("Login successful ✅")

        # Insert login tracker record
        cursor.execute(
            "INSERT INTO login_tracker (username, loginTime) VALUES (:1, :2)",
            (username, datetime.now())
        )
        conn.commit()

        # Generate secure session token
        token = secrets.token_hex(16)
        data = json.dumps({"username": username, "token": token}).encode()
        encrypted_data = cipher_suite.encrypt(data).decode()

        session_data = {
            "username": username,
            "encrypted_data": encrypted_data
        }

        if os.path.exists("session_token.json"):
            os.remove("session_token.json")

        with open("session_token.json", "w") as f:
            json.dump(session_data, f)
            print(f'Session token saved for {username}')

        # Run your Streamlit app after login
        subprocess.Popen([
            'streamlit', 'run', 'Back_test_oracle.py',
            "--server.port", "8501",
            "--server.address", "0.0.0.0",   # ✅ allows external access
            "--server.headless", "true"
        ])

        return redirect(f"http://127.0.0.1:8501")
    else:
        flash("Invalid credentials ❌")
        return redirect('/')


# -------------------------------
# RUN FLASK
# -------------------------------
if __name__ == '__main__':
    def open_browser():
        time.sleep(1)
        webbrowser.open("http://127.0.0.1:5000")
        print("DataGenie Running on http://127.0.0.1:5000")

    threading.Thread(target=open_browser).start()
    print("Flask Server Is Starting...")
    app.run(debug=True, host='127.0.0.1', port=5000)
