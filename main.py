import flask
from flask import request, jsonify, render_template, Flask
from visualizations import all_time_scorers

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/components.html')
def components():
    return render_template('components.html')


@app.route('/about.html')
def about():
    return render_template('about.html')


@app.route('/iihf.html')
def iihf():
    return render_template('iihf.html', top_scorers=all_time_scorers)


if __name__ == '__main__':
    app.run(debug=True)
