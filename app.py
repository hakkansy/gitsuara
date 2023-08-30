from flask import Flask, render_template, Response
from flask_socketio import SocketIO

app = Flask(__name__)


@app.route("/")
def main():
    return "Hello World!"

if __name__ == "__main__":
    app.run()