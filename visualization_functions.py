import plotly.express as px
import plotly.graph_objects as go

def plotly_scatter(df, x, y, color=None, size=None, title=None, x_title=None, y_title=None, color_title=None, size_title=None, x_log=False, y_log=False, size_log=False):
    fig = px.scatter(df, x=x, y=y, color=color, size=size, title=title, labels={x: x_title, y: y_title, color: color_title, size: size_title})
    if x_log:
        fig.update_xaxes(type="log")
    if y_log:
        fig.update_yaxes(type="log")
    if size_log:
        fig.update_traces(marker=dict(sizeref=2. * max(df[size]) / (40 ** 2)))
    return fig

def plotly_line(df, x, y, color=None, title=None, x_title=None, y_title=None, color_title=None, x_log=False, y_log=False):
    fig = px.line(df, x=x, y=y, color=color, title=title, labels={x: x_title, y: y_title, color: color_title})
    if x_log:
        fig.update_xaxes(type="log")
    if y_log:
        fig.update_yaxes(type="log")
    return fig


def plotly_bar(df, x, y, color=None, title=None, x_title=None, y_title=None, color_title=None, x_log=False, y_log=False):
    fig = px.bar(df, x=x, y=y, color=color, title=title, labels={x: x_title, y: y_title, color: color_title})
    if x_log:
        fig.update_xaxes(type="log")
    if y_log:
        fig.update_yaxes(type="log")
    return fig.to_html()


def plotly_heatmap(df, x, y, z, title=None, x_title=None, y_title=None, z_title=None, x_log=False, y_log=False):
    fig = px.imshow(df, x=x, y=y, z=z, title=title, labels={x: x_title, y: y_title, z: z_title})
    if x_log:
        fig.update_xaxes(type="log")
    if y_log:
        fig.update_yaxes(type="log")
    return fig


def plotly_pie(df, names, values, title=None):
    fig = px.pie(df, names=names, values=values, title=title)
    return fig


def plotly_box(df, x, y, color=None, title=None, x_title=None, y_title=None, color_title=None, x_log=False, y_log=False):
    fig = px.box(df, x=x, y=y, color=color, title=title, labels={x: x_title, y: y_title, color: color_title})
    if x_log:
        fig.update_xaxes(type="log")
    if y_log:
        fig.update_yaxes(type="log")
    return fig


def plotly_histogram(df, x, color=None, title=None, x_title=None, y_title=None, color_title=None, x_log=False, y_log=False):
    fig = px.histogram(df, x=x, color=color, title=title, labels={x: x_title, y: y_title, color: color_title})
    if x_log:
        fig.update_xaxes(type="log")
    if y_log:
        fig.update_yaxes(type="log")
    return fig


def plotly_3d(df, x, y, z, color=None, title=None, x_title=None, y_title=None, z_title=None, color_title=None, x_log=False, y_log=False, z_log=False):
    fig = px.scatter_3d(df, x=x, y=y, z=z, color=color, title=title, labels={x: x_title, y: y_title, z: z_title, color: color_title})
    if x_log:
        fig.update_xaxes(type="log")
    if y_log:
        fig.update_yaxes(type="log")
    if z_log:
        fig.update_zaxes(type="log")
    return fig


def plotly_surface(df, x, y, z, title=None, x_title=None, y_title=None, z_title=None, x_log=False, y_log=False, z_log=False):
    fig = go.Figure(data=[go.Surface(z=df[z].values, x=df[x].values, y=df[y].values)])
    fig.update_layout(title=title, scene = dict(xaxis_title=x_title, yaxis_title=y_title, zaxis_title=z_title))
    if x_log:
        fig.update_xaxes(type="log")
    if y_log:
        fig.update_yaxes(type="log")
    if z_log:
        fig.update_zaxes(type="log")
    return fig

