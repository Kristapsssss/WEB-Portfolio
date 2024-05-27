import dash
from dash import html, dcc
from dash.dependencies import Input, Output
import pandas as pd

df_data = pd.read_pickle('static/data/riga_listings_data.pkl')

def create_dash_app2(server):
    numeric_rooms = pd.to_numeric(df_data['Rooms'], errors='coerce')
    numeric_rooms = numeric_rooms.dropna()

    min_rooms = int(numeric_rooms.min())
    max_rooms = int(numeric_rooms.max())

    dash_app = dash.Dash(__name__,
                         server=server,
                         url_base_pathname='/dash_analysis/',
                         suppress_callback_exceptions=True,
                         external_stylesheets=['/static/css/custom.css'])

    # Define layout
    dash_app.layout = html.Div(className='main-div', children=[
        html.H1('Riga Rental Estate Analysis', className='main-heading'),

        # Dropdowns for filtering data (centered and side-by-side)
        html.Div([
            dcc.Dropdown(
                id='district-dropdown',
                options=[{'label': 'All Districts', 'value': 'all'}] +
                        [{'label': district, 'value': district} for district in df_data['District'].unique()],
                value='all',
                className='dropdown'
            ),
            dcc.Dropdown(
                id='series-dropdown',
                options=[{'label': 'All Series', 'value': 'all'}] +
                        [{'label': series, 'value': series} for series in df_data['Series'].unique()],
                value='all',
                className='dropdown'
            ),
            dcc.Dropdown(
                id='statistic-dropdown',
                options=[
                    {'label': 'Mean', 'value': 'mean'},
                    {'label': 'Median', 'value': 'median'}
                ],
                value='mean',
                className='dropdown'
            )
        ], className='dropdown-container'),

        # Divided area for statistics
        html.Div(id='statistics', className='statistics-div'),

        # Purple background div with two columns
        html.Div([
            # First column with a table
            html.Div([
                html.H2(id='price-difference-heading', className='sub-heading'),
                html.P('Filter the results by selecting a series and/or switching between median/mean calculations.', className='sub-paragraph'),
                html.P(id='price-difference-subheading', className='sub-paragraph'),
                html.Table(id='price-difference-table')
            ], className='column-div'),

            # Second column for analysis and good deals
            html.Div([
                html.H2('Good Deals Analysis', className='sub-heading'),
                html.P('Find top listings priced below market market avarage/median.', className='sub-paragraph'),
                html.P('Filter listings by selecting a series and/or moving the sliders to'
                       ' change the minimum required rooms or size of the apartment.', className='sub-paragraph'),
                # Add sliders here
                html.Div([
                    html.Label('Minimum number of Rooms', className='slider-label'),
                    dcc.Slider(
                        id='room-slider',
                        min=min_rooms,
                        max=max_rooms,
                        step=1,
                        value=min_rooms,
                        marks={i: str(i) for i in range(min_rooms, max_rooms + 1)},
                        tooltip={'placement': 'bottom', 'always_visible': True},
                        className='slider'
                    ),
                    html.Label('Minimum size of listings (m²)', className='slider-label'),
                    dcc.Slider(
                        id='size-slider',
                        min=df_data['Size'].min(),
                        max=df_data['Size'].max(),
                        step=1,
                        value=df_data['Size'].min(),
                        marks={i: str(i) for i in range(int(df_data['Size'].min()), int(df_data['Size'].max()) + 1)},
                        tooltip={'placement': 'bottom', 'always_visible': True},
                        className='slider'
                    ),
                    html.Div(id='good-deals-analysis')
                ])
            ], className='column-div'),
        ], className='columns-container')
    ])

    # Callback to handle dropdown changes and calculate statistics
    @dash_app.callback(
        Output('statistics', 'children'),
        [Input('district-dropdown', 'value'),
         Input('series-dropdown', 'value'),
         Input('statistic-dropdown', 'value')]
    )
    def update_statistics(selected_district, selected_series, selected_statistic):
        filtered_data = df_data
        if selected_district != 'all':
            filtered_data = filtered_data[filtered_data['District'] == selected_district]
        if selected_series != 'all':
            filtered_data = filtered_data[filtered_data['Series'] == selected_series]

        if selected_statistic == 'mean':
            price_stat = filtered_data['Price'].mean()
            size_stat = filtered_data['Size'].mean()
            price_per_m2_stat = price_stat / size_stat
            rooms_stat = pd.to_numeric(filtered_data['Rooms'], errors='coerce').mean()
        else:  # median
            price_stat = filtered_data['Price'].median()
            size_stat = filtered_data['Size'].median()
            price_per_m2_stat = price_stat / size_stat
            rooms_stat = pd.to_numeric(filtered_data['Rooms'], errors='coerce').median()

        statistics = [
            html.Div([
                html.H4(f'{selected_statistic.capitalize()} Price', className='stat-heading'),
                html.P(f'{price_stat:.2f} €/month', className='stat-value')
            ], className='stat-box'),
            html.Div([
                html.H4(f'{selected_statistic.capitalize()} Size', className='stat-heading'),
                html.P(f'{size_stat:.2f} m²', className='stat-value')
            ], className='stat-box'),
            html.Div([
                html.H4(f'{selected_statistic.capitalize()} Price per m²', className='stat-heading'),
                html.P(f'{price_per_m2_stat:.2f} €/m²', className='stat-value')
            ], className='stat-box'),
            html.Div([
                html.H4(f'{selected_statistic.capitalize()} Rooms', className='stat-heading'),
                html.P(f'{rooms_stat:.2f}', className='stat-value')
            ], className='stat-box')
        ]

        return statistics

    # Callback to update the price difference table and good deals analysis
    @dash_app.callback(
        [Output('price-difference-table', 'children'),
         Output('price-difference-heading', 'children'),
         Output('price-difference-subheading', 'children'),
         Output('good-deals-analysis', 'children')],
        [Input('statistic-dropdown', 'value'),
         Input('series-dropdown', 'value'),
         Input('room-slider', 'value'),  # Add room slider input
         Input('size-slider', 'value')]  # Add size slider input
    )
    def update_price_difference_table(selected_statistic, selected_series, min_rooms, min_size):
        if selected_series != 'all':
            df_filtered = df_data[df_data['Series'] == selected_series]
        else:
            df_filtered = df_data

        # Calculate overall market statistic (mean or median)
        if selected_statistic == 'mean':
            overall_stat = df_filtered['Price'].mean()
            heading_text = 'Average Price Differences Across All Districts'
        else:  # median
            overall_stat = df_filtered['Price'].median()
            heading_text = 'Median Price Differences Across All Districts'

        # Calculate district-level statistics
        if selected_statistic == 'mean':
            district_price_stats = df_filtered.groupby('District')['Price'].mean()
        else:  # median
            district_price_stats = df_filtered.groupby('District')['Price'].median()

        overall_price_stat = overall_stat
        price_diff_percentage = ((district_price_stats - overall_price_stat) / overall_price_stat) * 100
        sorted_districts = price_diff_percentage.sort_values(ascending=False)

        # Note cheapest and most expensive districts
        cheapest_district = sorted_districts.idxmin()
        most_expensive_district = sorted_districts.idxmax()
        subheading_text = (f'Cheapest district is: {cheapest_district}.\n'
                           f'Most expensive district is: {most_expensive_district}.')

        # Create price difference table
        table_rows = [
            html.Tr([html.Th('District')] + [html.Th(selected_statistic.capitalize() + ' Price Difference (%)')])
        ]
        for district, percentage in sorted_districts.items():
            color = 'red' if percentage > 0 else 'green'
            table_rows.append(html.Tr([
                html.Td(district),
                html.Td(f'{percentage:.2f}%', style={'color': color})
            ]))

        min_rooms = int(min_rooms)
        min_size = int(min_size)

        # Analyze data to find great deals
        good_deals = []
        threshold = -40  # Example threshold for a great deal: 40% below overall stat

        df_filtered['Percentage Difference'] = ((df_filtered['Price'] - overall_price_stat) / overall_price_stat) * 100

        for index, row in df_filtered.iterrows():
            if (row['Percentage Difference'] < threshold and
                    int(row['Rooms']) >= min_rooms and
                    int(row['Size']) >= min_size):
                good_deals.append(row)

        # Sort the good deals by percentage difference (most negative first)
        good_deals = sorted(good_deals, key=lambda x: x['Percentage Difference'])[:10]

        # Create the good deals table
        good_deals_table = html.Table([
            html.Thead(html.Tr([
                                   html.Th(col) for col in df_filtered.columns if
                                   col not in ['Latitude', 'Longitude', 'Percentage Difference']
                               ] + [html.Th('Below Market Price (%)')])),
            html.Tbody([
                html.Tr([html.Td(deal[col]) for col in df_filtered.columns if
                         col not in ['Latitude', 'Longitude', 'Percentage Difference']]
                        + [html.Td(f"{deal['Percentage Difference']:.2f}%", style={'color': 'green'})])
                for deal in good_deals
            ])
        ], className='styled-table')

        return (html.Table([
            html.Thead(table_rows[0]),
            html.Tbody(table_rows[1:])
        ], className='styled-table'), heading_text, subheading_text, good_deals_table)

    return dash_app
