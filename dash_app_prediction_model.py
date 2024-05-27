import dash
from dash import html, dcc
from dash.dependencies import Input, Output
import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# Load your data
df_data = pd.read_pickle('static/data/riga_listings_data.pkl')

# Prepare the data
X = df_data[['Rooms', 'Size', 'District', 'Series']]
y = df_data['Price']

# Convert 'Rooms' to a numerical variable if necessary
X.loc[:, 'Rooms'] = pd.to_numeric(X['Rooms'], errors='coerce')

# Drop rows with missing values
X = X.dropna()
y = y[X.index]

# Calculate base price per square meter
base_price_per_sqm = df_data['Price'].sum() / df_data['Size'].sum()

# Calculate average price per square meter for each room count
room_avg_price_per_sqm = df_data.groupby('Rooms').apply(lambda x: x['Price'].sum() / x['Size'].sum())
rooms_premium_percentage_dict = (room_avg_price_per_sqm - base_price_per_sqm) / base_price_per_sqm

# Calculate average price per square meter for each series
series_avg_price_per_sqm = df_data.groupby('Series').apply(lambda x: x['Price'].sum() / x['Size'].sum())
series_premium_percentage_dict = (series_avg_price_per_sqm - base_price_per_sqm) / base_price_per_sqm

# Calculate average price per square meter for each district
district_avg_price_per_sqm = df_data.groupby('District').apply(lambda x: x['Price'].sum() / x['Size'].sum())
district_premium_percentage_dict = (district_avg_price_per_sqm - base_price_per_sqm) / base_price_per_sqm

# Define the pricing function
def calculate_rental_price(size, rooms, series, district):
    base_price = size * base_price_per_sqm
    room_premium_percentage = rooms_premium_percentage_dict.get(rooms, 0)
    series_premium_percentage = series_premium_percentage_dict.get(series, 0)
    district_premium_percentage = district_premium_percentage_dict.get(district, 0)

    # Scale the premiums with the size
    total_premium_percentage = room_premium_percentage + series_premium_percentage + district_premium_percentage
    total_price = base_price * (1 + total_premium_percentage)
    return total_price

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=90)

# Generate predictions for the test set
y_pred = X_test.apply(lambda row: calculate_rental_price(row['Size'], row['Rooms'], row['Series'], row['District']), axis=1)

# Calculate MAE and R²
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Create the scatter plot
fig = px.scatter(
    x=y_test, y=y_pred,
    labels={'x': 'Actual Prices', 'y': 'Predicted Prices'},
    title='Actual Prices vs Predicted Prices'
)

# Update layout to set background color and label colors
fig.update_layout(
    plot_bgcolor='#E3DAE8',  # Set the plot background color (inside the axes)
    paper_bgcolor='rgb(105, 90, 166)',      # Set the paper background color (outside the axes)
    font=dict(color='white'),               # Set the font color for the labels
    title_font=dict(size=30, color='white'),# Set the title font color
    xaxis=dict(
        title=dict(text='Actual Prices', font=dict(color='white')),  # X-axis title font color
        tickfont=dict(color='white')                                 # X-axis tick font color
    ),
    yaxis=dict(
        title=dict(text='Predicted Prices', font=dict(color='white')), # Y-axis title font color
        tickfont=dict(color='white')                                   # Y-axis tick font color
    )
)
# Add a line for perfect prediction
fig.add_shape(
    type='line',
    x0=min(y_test), y0=min(y_test),
    x1=max(y_test), y1=max(y_test),
    line=dict(color='red', dash='dash')
)

def create_dash_app3(server):
    dash_app = dash.Dash(__name__,
                         server=server,
                         url_base_pathname='/dash_regression_model/',
                         suppress_callback_exceptions=True,
                         external_stylesheets=['/static/css/dash_regression_model.css'])

    # Define layout
    dash_app.layout = html.Div(className='main-div', children=[
        html.H1('Riga Rental Estate Price Predictor', className='main-heading'),

        # Input fields for prediction
        html.Div([
            html.Div([
                html.Div([
                    html.Label('Size of Apartment (m²)', className='input-label'),
                    dcc.Input(id='size-input', type='number', value=50, className='input-field')
                ], className='input-group'),

                html.Div([
                    html.Label('Number of Rooms', className='input-label'),
                    dcc.Input(id='rooms-input', type='number', value=2, className='input-field')
                ], className='input-group'),

                html.Div([
                    html.Label('District', className='input-label'),
                    dcc.Dropdown(
                        id='district-input',
                        options=[{'label': district, 'value': district} for district in df_data['District'].unique()],
                        value=df_data['District'].unique()[0],
                        className='dropdown'
                    )
                ], className='input-group'),

                html.Div([
                    html.Label('Series', className='input-label'),
                    dcc.Dropdown(
                        id='series-input',
                        options=[{'label': series, 'value': series} for series in df_data['Series'].unique()],
                        value=df_data['Series'].unique()[0],
                        className='dropdown'
                    )
                ], className='input-group')
            ], className='input-row'),

            html.Button('Predict Price', id='predict-button', className='predict-button')
        ], className='input-container'),

        # Predicted price output
        html.Div(id='predicted-price-output', className='output-container'),

        # Visualization and metrics
        html.Div([
            dcc.Graph(figure=fig, id='scatter-plot'),

            html.Div([
                html.H4(f'Mean Absolute Error (MAE): {mae:.2f}', className='metric'),

                html.H4(f'R-squared (R²) Score: {r2:.2f}', className='metric')
            ], className='metrics-container'),
            html.Div([
                html.P('MAE tells us, on average, how much our predictions differ from the actual rental prices. Lower MAE means better predictions.', className='metric-description'),
                html.P('R² shows how well our predictions match the actual rental prices. An R² of 1 means perfect predictions, while an R² of 0 means the predictions are no better than guessing.', className='metric-description')
            ], className='metrics-description-container')
        ], className='visualization-container')
    ])


    # Callback to predict price based on user input
    @dash_app.callback(
        Output('predicted-price-output', 'children'),
        [Input('predict-button', 'n_clicks')],
        [dash.dependencies.State('size-input', 'value'),
         dash.dependencies.State('rooms-input', 'value'),
         dash.dependencies.State('district-input', 'value'),
         dash.dependencies.State('series-input', 'value')]
    )
    def predict_rental_price(n_clicks, size, rooms, district, series):
        if n_clicks is None:
            return ''
        predicted_price = calculate_rental_price(size, rooms, series, district)
        return html.Div([
            html.H3(f'Predicted Rental Price: {predicted_price:.2f} €/month', className='predicted-price')
        ])

    return dash_app