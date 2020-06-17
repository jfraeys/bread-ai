from flask import Flask, request, render_template
from flask_sqlalchemy import SQLAlchemy

app = Flask(__name__)

@app.route("/")
def home():
    return "<b>There was a change here</b>"

@app.route("/template")
def template():
    return render_template('index.html')

if __name__ == "__main__":
    app.run(port=8081, host='0.0.0.0', debug=True)


