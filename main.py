from flask import render_template, Flask
from dash_app_visualizations import create_dash_app
from dash_app_analysis import create_dash_app2
from dash_app_prediction_model import create_dash_app3
from latvia_car_price_prediction import create_dash_app4

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


@app.route('/riga_listings.html')
def riga_listings():
    return render_template('riga_listings.html')

@app.route('/car_listings.html')
def car_listings():
    return render_template('latvian_cars_price_prediction.html')


# Initialize Dash app and associate it with the Flask server
dash_app = create_dash_app(app)
dash_app2 = create_dash_app2(app)
dash_app3 = create_dash_app3(app)
dash_app4 = create_dash_app4(app)

if __name__ == '__main__':
    app.run(debug=True)
