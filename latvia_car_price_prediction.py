import dash
from dash import html, dcc
from dash.dependencies import Input, Output, State
import pandas as pd
import joblib
import plotly.express as px

# Load data and models
df_data = pd.read_pickle('static/data/latvia_car_data.pkl')
model_stats_df = pd.read_csv('static/data/model_stats.csv')
best_model_name = model_stats_df.loc[model_stats_df['R²'].idxmax()]['Model Name']
best_model = joblib.load('static/models/gradient_boosting_model.pkl')
feature_importance_df = joblib.load('static/data/gradient_boosting_model_feature_importance.pkl')

# Filter models with more than 50 data points
model_counts = df_data['Model'].value_counts()
sufficient_data_models = model_counts[model_counts > 50].index
filtered_data = df_data[df_data['Model'].isin(sufficient_data_models)].copy()

# Feature Engineering: Create new features
filtered_data.loc[:, 'Car Age'] = 2024 - filtered_data['Year']  # Assuming the current year is 2024

# Preprocess the data
filtered_data = filtered_data.dropna().copy()  # Example of handling missing values
filtered_data = pd.get_dummies(filtered_data,
                               columns=['Brand', 'Model', 'Engine Type'])  # Encoding categorical variables
model_features = filtered_data.drop(columns=['Price']).columns


def create_dash_app4(server):
    dash_app = dash.Dash(__name__,
                         server=server,
                         url_base_pathname='/car_prediction/',
                         suppress_callback_exceptions=True,
                         external_stylesheets=['/static/css/dash_car_price_prediction.css'])

    dash_app.layout = html.Div(className='main-div', children=[
        html.H1('Car Price Prediction Analysis', className='main-heading'),

        # Section 1: Model Comparison
        html.Div(className='section-div', children=[
            html.H2('Model Comparison', className='section-heading'),
            html.Div(className='input-group', children=[
                html.Label('Select Metric:', className='input-label'),
                dcc.Dropdown(
                    id='metric-dropdown',
                    options=[
                        {'label': 'Mean Absolute Error (MAE)', 'value': 'MAE'},
                        {'label': 'Mean Squared Error (MSE)', 'value': 'MSE'},
                        {'label': 'R-Squared (R²)', 'value': 'R²'}
                    ],
                    value='R²',
                    className='dropdown'
                )
            ]),
            dcc.Graph(id='model-comparison-graph')
        ]),

        # Section 2: Predicted vs Actual Prices
        html.Div(className='section-div', children=[
            html.H2('Predicted vs Actual Prices', className='section-heading'),
            dcc.Graph(id='predicted-vs-actual-graph')
        ]),

        # Section 3: Feature Importance
        html.Div(className='section-div', children=[
            html.H2('Feature Importance (Top 10)', className='section-heading'),
            dcc.Graph(id='feature-importance-graph')
        ]),

        # Section 4: Price Prediction
        html.Div(className='section-div', children=[
            html.H2('Predict Car Price', className='section-heading'),
            html.Div(className='input-container', children=[
                html.Div(className='input-group', children=[
                    html.Label('Year:', className='input-label'),
                    dcc.Input(id='input-year', type='number', value=2020, className='input-field')
                ]),
                html.Div(className='input-group', children=[
                    html.Label('Mileage (km):', className='input-label'),
                    dcc.Input(id='input-mileage', type='number', value=50000, className='input-field')
                ]),
                html.Div(className='input-group', children=[
                    html.Label('Engine Size (L):', className='input-label'),
                    dcc.Input(id='input-engine-size', type='number', value=2.0, className='input-field')
                ]),
                html.Div(className='input-group', children=[
                    html.Label('Brand:', className='input-label'),
                    dcc.Dropdown(
                        id='input-brand',
                        options=[{'label': brand, 'value': brand} for brand in df_data['Brand'].unique()],
                        value=df_data['Brand'].unique()[0],
                        className='dropdown'
                    )
                ]),
                html.Div(className='input-group', children=[
                    html.Label('Model:', className='input-label'),
                    dcc.Dropdown(
                        id='input-model',
                        options=[],
                        className='dropdown'
                    )
                ]),
                html.Div(className='input-group', children=[
                    html.Label('Engine Type:', className='input-label'),
                    dcc.Dropdown(
                        id='input-engine-type',
                        options=[],
                        className='dropdown'
                    )
                ]),
            ]),
            html.Div(style={'display': 'flex', 'justify-content': 'center'}, children=[
                html.Button('Predict Price', id='predict-button', className='predict-button')
            ]),
            html.Div(id='prediction-result', className='output-container')
        ])
    ])

    # Callback to update Model Comparison graph based on selected metric
    @dash_app.callback(
        Output('model-comparison-graph', 'figure'),
        Input('metric-dropdown', 'value')
    )
    def update_model_comparison_graph(selected_metric):
        sorted_model_stats_df = model_stats_df.sort_values(by=selected_metric, ascending=(selected_metric != 'R²'))

        fig = px.bar(
            sorted_model_stats_df,
            x='Model Type',
            y=selected_metric,
            title=f'Model Comparison by {selected_metric}'
        )

        if selected_metric == 'R²':
            fig.update_yaxes(tick0=0.0, dtick=0.1, range=[0, 1])
        else:
            fig.update_yaxes(tickformat=".2f")

        fig.update_layout(
            paper_bgcolor='rgb(105, 90, 166)',
            plot_bgcolor='rgb(105, 90, 166)',
            font=dict(color='white'),
            xaxis=dict(tickangle=-15)
        )
        return fig

    # Callback to update Predicted vs Actual prices graph
    @dash_app.callback(
        Output('predicted-vs-actual-graph', 'figure'),
        Input('model-comparison-graph', 'figure')
    )
    def update_predicted_vs_actual(_):
        y_test = filtered_data['Price']  # Replace with actual test set
        X_test = filtered_data.drop('Price', axis=1)  # Replace with actual test set features

        # Ensure the test data has the same columns as the training data
        X_test = X_test.reindex(columns=model_features, fill_value=0)

        y_pred_best = best_model.predict(X_test)  # Replace with actual predictions

        fig = px.scatter(
            x=y_test,
            y=y_pred_best,
            labels={'x': 'Actual Prices', 'y': 'Predicted Prices'},
            title='Actual vs Predicted Car Prices'
        )
        fig.add_shape(
            type='line',
            x0=y_test.min(),
            x1=y_test.max(),
            y0=y_test.min(),
            y1=y_test.max(),
            line=dict(color='red', dash='dash')
        )
        fig.update_traces(marker=dict(color='darkblue'))
        fig.update_layout(
            paper_bgcolor='rgb(105, 90, 166)',
            plot_bgcolor='rgb(105, 90, 166)',
            font=dict(color='white')
        )
        return fig

    # Callback to update Feature Importance graph
    @dash_app.callback(
        Output('feature-importance-graph', 'figure'),
        Input('model-comparison-graph', 'figure')
    )
    def update_feature_importance(_):
        top_10_features = feature_importance_df.head(10)
        fig = px.bar(
            top_10_features,
            x='Importance',
            y='Feature',
            orientation='h',
            title='Top 10 Feature Importance'
        )
        fig.update_layout(
            paper_bgcolor='rgb(105, 90, 166)',
            plot_bgcolor='rgb(105, 90, 166)',
            font=dict(color='white'),
            yaxis={'categoryorder': 'total ascending'}
        )
        return fig

    # Callback to update Model dropdown based on selected Brand
    @dash_app.callback(
        Output('input-model', 'options'),
        [Input('input-brand', 'value')]
    )
    def update_model_dropdown(selected_brand):
        models = df_data[df_data['Brand'] == selected_brand]['Model'].unique()
        return [{'label': model, 'value': model} for model in models]

    # Callback to update Engine Type dropdown based on selected Model
    @dash_app.callback(
        Output('input-engine-type', 'options'),
        [Input('input-brand', 'value'),
         Input('input-model', 'value')]
    )
    def update_engine_type_dropdown(selected_brand, selected_model):
        engine_types = df_data[(df_data['Brand'] == selected_brand) & (df_data['Model'] == selected_model)]['Engine Type'].unique()
        return [{'label': etype, 'value': etype} for etype in engine_types]

    # Callback to handle prediction
    @dash_app.callback(
        Output('prediction-result', 'children'),
        [Input('predict-button', 'n_clicks')],
        [State('input-year', 'value'),
         State('input-mileage', 'value'),
         State('input-engine-size', 'value'),
         State('input-brand', 'value'),
         State('input-model', 'value'),
         State('input-engine-type', 'value')]
    )
    def predict_price(n_clicks, year, mileage, engine_size, brand, model, engine_type):
        if n_clicks is not None and n_clicks > 0:
            input_data = pd.DataFrame({
                'Year': [year],
                'Mileage (km)': [mileage],
                'Engine Size (L)': [engine_size],
                'Car Age': [2024 - year],  # Create Car Age feature
                'Brand': [brand],
                'Model': [model],
                'Engine Type': [engine_type]
            })
            input_data = pd.get_dummies(input_data)
            input_data = input_data.reindex(columns=model_features, fill_value=0)

            # Predict the price
            predicted_price = best_model.predict(input_data)[0]
            return f'Predicted Price: €{predicted_price:,.2f}'
        return ''

    return dash_app