from flask import Flask, render_template
from flask_socketio import SocketIO
from threading import Thread
from ascending_method import AscendingMethod
import sys

# device_index = 29
app = Flask(__name__)
socketio = SocketIO(app, async_mode='threading')
# main = AscendingMethod()

# data_mahasiswa = [
#     {
#         "nama" : "hakkan syukri",
#         'kelas' : 'trpl 4a',
#         "alamat" : 'padang'
#     },
#     {
#         "nama" : "james",
#         'kelas' : 'trpl 4a',
#         "alamat" : 'padang'
#     },
#     {
#         "nama" : "jay",
#         'kelas' : 'trpl 4a',
#         "alamat" : 'padang'
#     },
# ]


# @app.route("/")
# def home():
#     return render_template("home.html")

# @app.route("/about")
# def about():
#     return render_template("about.html")

# @app.route("/data_mahasiswa")
# def data_mhs():
#     return render_template("data_mahasiswa.html", data_mahasiswa = data_mahasiswa)

# @app.route("/alumni")
# def alumni():
#     return render_template("alumni.html")


# @app.route("/artikel/<info>")
# def artikel_info(info):
#     return "Halaman artikel "+ info


# @socketio.on('button_start_tes')
# def start_tes():
#     tes = Thread(target=main.start(socketio))
#     tes.daemon = True
#     tes.start()

@app.route('/')
def index():
    return render_template('home.html')

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