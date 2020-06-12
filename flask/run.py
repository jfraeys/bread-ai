from flask import Flask, request, render_template
from flask_sqlalchemy import SQLAlchemy

user = 'admin'
pwd = 'admin'

app = Flask(__name__)
db = SQLAlchemy()

@app.route("/", methods=["GET", "POST"])
def index():
    return render_template('templates/index.html')

@app.route("/shutdown", methods=['POST'])
def shutdown_server():
    if request.form.get('username') and request.form.get('password'):
        if request.form.get('username') != user or request.form.get('password') != pwd:
            return "The username or password appears to be incorrect."
        shutdown = request.environ.get("werkzeug.server.shutdown")
        if shutdown is None:
            raise RuntimeError('The function is unavailable')
        else:
            shutdown()

    else:
        return "WARNING: You need authorization to shutdown the server!"

if __name__ == "__main__":
    app.run(port=8081, host='0.0.0.0', debug= True)


