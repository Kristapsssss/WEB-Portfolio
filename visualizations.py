from visualization_functions import plotly_line, plotly_bar, plotly_heatmap, plotly_pie, plotly_box, plotly_histogram, plotly_3d
import pandas as pd
import numpy as np
import openpyxl

# Create a DataFrame
df_all_time_latvia_scorers = pd.read_excel('static/data/all_time_latvia_scorers.xlsx', sheet_name=1, skiprows=1)

all_time_scorers = plotly_bar(df_all_time_latvia_scorers, 'Name', 'G', title='All Time Latvia Scorers', x_title='Player', y_title='Goals', color_title='Goals', x_log=False, y_log=False)