from flask import Flask, render_template
from flask_socketio import SocketIO
from threading import Thread
from ascending_method import AscendingMethod
import signal
import os
import sys

is_cancel = False

app = Flask(__name__)
socketio = SocketIO(app, async_mode='threading')

asc_m = None
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

@socketio.on('cancel')
def handle_cancel(data):
   print('Hai dari client')
   # keyboard interrupt = CTRL+C
   os.kill(os.getpid(), signal.SIGINT)
   

def start():
    try:
        with AscendingMethod(socketio) as asc_method:
            global asc_m
            asc_m = asc_method
            asc_method.run()

    except KeyboardInterrupt:
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