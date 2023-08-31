from flask import Flask, render_template
from flask_socketio import SocketIO
from threading import Thread
from ascending_method import AscendingMethod
import sys


app = Flask(__name__)
socketio = SocketIO(app, async_mode='threading')

@app.route('/socket')
def index():
    return render_template('home_socket.html')

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/start_test")
def start_test():
    return render_template("start_test.html")

def start():
    try:
        with AscendingMethod(socketio) as asc_method:
            asc_method.run()

    except KeyboardInterrupt:
        # keluar setelah familiarization
        sys.exit('\nInterrupted by user')

if __name__ == '__main__':
    # logging
    # earsides
    # web = Thread(target= app.run(debug=True))
    # web.daemon = True
    # web.start()
    Thread(target=socketio.run,args=(app,)).start()
    # socketio.run(app,debug=True)
    start()
    
    # socketio.run(app,)
    # socketio.run(app, debug=True)