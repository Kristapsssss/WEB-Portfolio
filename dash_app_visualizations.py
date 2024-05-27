import dash
from dash import html, dcc
from dash.dependencies import Input, Output
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

df_data = pd.read_pickle('static/data/riga_listings_data.pkl')

# Define custom color scale
colors = [
    [0, 'rgb(0, 0, 255)'],  # Dark blue (low prices)
    [0.35, 'rgb(125, 0, 125)'],  # Blue
    [0.6, 'rgb(255, 0, 0)'],  # Yellow
    [0.8, 'rgb(200, 0, 0)'],  # Dark orange
    [1, 'rgb(150, 0, 0)']  # Red (high prices)
]

# Create the color scale
color_scale = go.Contour.colorscale = colors


def create_dash_app(server):
    dash_app = dash.Dash(__name__,
                         server=server,
                         url_base_pathname='/dash_visualization/',
                         suppress_callback_exceptions=True,
                         external_stylesheets=['/static/css/custom.css'])

    dash_app.layout = html.Div(style={'backgroundColor': '#E3DAE8'}, children=[
        html.H1('Riga Rental Estate Visualization', className='main-heading'),
        html.P('Filter visualizations by district and series',
               className='sub-heading', style={'fontSize': '24px', 'color': 'rgb(105, 90, 166)'}),
        html.Br(),
        html.Div([
            dcc.Dropdown(
                id='district-dropdown',
                options=[{'label': 'All Districts', 'value': 'all'}] +
                        [{'label': district, 'value': district} for district in df_data['District'].unique()],
                value='all',
                className='dropdown',
            ),
            dcc.Dropdown(
                id='series-dropdown',
                options=[{'label': 'All Series', 'value': 'all'}] +
                        [{'label': series, 'value': series} for series in df_data['Series'].unique()],
                value='all',
                className='dropdown',
            ),
            html.Div(id='listing-counter', style={
                'fontSize': '30px',
                'fontWeight': 'bold',
                'color': 'white',
                'padding': '10px',
                'border': '3px solid black',
                'borderRadius': '10px',
                'backgroundColor': 'rgb(105, 90, 166)',
                'fontFamily': 'Open Sans, sans-serif',
                'position': 'relative',
                'left': '25%',
                'transform': 'translateX(-25%)'
            })
        ], className='dropdown-container'),
        html.Br(),
        html.Div([
            html.Div([
                html.Div([
                    # First row
                    html.Div([
                        # 25% width on the left
                        html.Div([
                            dcc.Graph(id='price-histogram',
                                      style={'width': '100%', 'height': '100%', 'margin': '10px'}),
                            dcc.Graph(id='scatter_plot',
                                      style={'width': '100%', 'height': '100%', 'margin': '10px'}),
                        ], style={'width': '25%', 'float': 'left'}),

                        # 75% width on the right
                        html.Div([
                            dcc.Graph(id='map-plot', style={'width': '100%', 'height': '100%', 'margin': '10px'}),
                        ], style={'width': '75%', 'float': 'right'}),
                    ], style={'width': '100%', 'overflow': 'visible'}),

                    # Second row
                    html.Div([

                        html.Div([

                            dcc.Graph(id='price-boxplot', style={'width': '100%', 'height': '100%', 'margin': '10px'}),
                        ], style={'clear': 'both'}
                        )], style={'width': '100%', 'overflow': 'visible'}
                    )], style={'width': '100%'}),
                html.Div([
                    dcc.Graph(id='line_chart', style={'width': '50%', 'height': '100%', 'margin': '10px'}),
                    dcc.Graph(id='sunburst', style={'width': '50%', 'height': '100%', 'margin': '10px'}),

                ], style={'width': '100%', 'overflow': 'visible', 'display': 'flex'})
            ])
        ])
    ])

    @dash_app.callback(
        [Output('map-plot', 'figure'),
         Output('price-boxplot', 'figure'),
         Output('sunburst', 'figure'),
         Output('line_chart', 'figure'),
         Output('scatter_plot', 'figure'),
         Output('price-histogram', 'figure'),
         Output('listing-counter', 'children')],
        [Input('district-dropdown', 'value'),
         Input('series-dropdown', 'value')]
    )
    def update_visualizations(selected_district, selected_series):
        filtered_data = df_data
        if selected_district != 'all':
            filtered_data = filtered_data[filtered_data['District'] == selected_district]
        if selected_series != 'all':
            filtered_data = filtered_data[filtered_data['Series'] == selected_series]

        # Count listings
        listing_count = len(filtered_data)
        counter_text = f'Total Listings: {listing_count}'

        # Scatter Mapbox
        map_fig = px.scatter_mapbox(
            filtered_data,
            lat="Latitude",
            lon="Longitude",
            hover_name="Title",
            hover_data={"District": True,
                        "Price": True,
                        "Rooms": True,
                        "Size": True,
                        "FullAddress": True,
                        "Latitude": False,
                        "Longitude": False},
            color="Price",
            color_continuous_scale=color_scale,
            size="Price",
            zoom=10,
            height=600,
        )
        map_fig.update_layout(mapbox_style="open-street-map",
                              margin={"r": 0, "t": 0, "l": 0, "b": 0},
                              plot_bgcolor='#E3DAE8',
                              paper_bgcolor='#E3DAE8')

        # Histogram
        histogram_fig = px.histogram(filtered_data, x='Price',
                                     nbins=25,
                                     title="Histogram Rent Distribution",
                                     height=300)
        histogram_fig.update_layout(margin={"r": 10, "t": 60, "l": 10, "b": 10},
                                    plot_bgcolor='#E3DAE8',
                                    paper_bgcolor='#E3DAE8',
                                    title={
                                        'text': 'Histogram Rent Distribution (€/Month)',
                                        'y': 0.9,
                                        'x': 0.5,
                                        'xanchor': 'center',
                                        'yanchor': 'top'
                                    })

        # Boxplot
        boxplot_fig = px.box(filtered_data, x='Price', height=200)
        boxplot_fig.update_layout(margin={"r": 10, "t": 60, "l": 10, "b": 10},
                                  plot_bgcolor='#E3DAE8',
                                  paper_bgcolor='#E3DAE8',
                                  title={
                                      'text': 'Boxplot Rent Distribution (€/Month)',
                                      'y': 0.9,
                                      'x': 0.5,
                                      'xanchor': 'center',
                                      'yanchor': 'top'
                                  })

        # Sunburst
        sunburst_fig = px.sunburst(filtered_data, path=['Series', 'Rooms'], height=400)
        sunburst_fig.update_layout(margin={"r": 10, "t": 60, "l": 10, "b": 10},
                                   plot_bgcolor='#E3DAE8',
                                   paper_bgcolor='#E3DAE8',
                                   title={
                                       'text': 'Sunburst Series/Room Count',
                                       'y': 0.95,
                                       'x': 0.5,
                                       'xanchor': 'center',
                                       'yanchor': 'top'
                                   })

        # Line chart
        avg_price_per_room = filtered_data.groupby('Rooms')['Price'].mean().reset_index()
        line_fig = px.line(avg_price_per_room, x='Rooms', y='Price', height=400)
        line_fig.update_layout(
            margin={"r": 20, "t": 60, "l": 20, "b": 20},
            plot_bgcolor='#E3DAE8',
            paper_bgcolor='#E3DAE8',
            xaxis=dict(title='Number of Rooms',
                       range=[avg_price_per_room['Rooms'].min(),
                              avg_price_per_room['Rooms'].max()]),
            yaxis=dict(title='Average Price (€/Month)'),
            title={
                'text': 'Average Price (€/Month) for Number of Rooms',
                'y': 0.9,
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top'
            }
        )
        line_fig.update_traces(
            line=dict(width=4, shape='spline'),
            marker=dict(size=8)
        )

        # Scatter plot
        scatter_plot = px.scatter(filtered_data, x='Size', y='Price', title='Price vs Size', height=300)
        scatter_plot.update_layout(
            margin={"r": 20, "t": 60, "l": 20, "b": 20},
            plot_bgcolor='#E3DAE8',
            paper_bgcolor='#E3DAE8',
            xaxis=dict(title='Size'),
            yaxis=dict(title='Price'),
            title={
                'text': 'Price (€/Month) vs Size (m²)',
                'y': 0.9,
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top'
            }
        )

        return map_fig, boxplot_fig, sunburst_fig, line_fig, scatter_plot, histogram_fig, counter_text

    return dash_app
