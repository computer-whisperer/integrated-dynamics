from bokeh.plotting import figure, curdoc
from bokeh.models.widgets.tables import TableColumn, DataTable
from bokeh.models import ColumnDataSource
from bokeh.io import show, vform
import simplestreamer
import numpy as np


# create a plot and style its properties
p = figure(x_range=(-100, 100), y_range=(-100, 100), toolbar_location=None)
p.border_fill_color = 'black'
p.background_fill_color = 'black'
p.outline_line_color = None
p.grid.grid_line_color = None


SAMPLES_PER_SEC = 100
SECS_REMEMBERED = 2
value_plots = {}
value_lines = {}

sim_circle = p.circle([0, 1], [0, -1], size=5, color="blue")
est_circle = p.circle([0, 1], [0, 1], size=5, color="red")


streamer = simplestreamer.SimpleStreamer(5805)
streamer.subscribe("127.0.0.1", 5803, "simulation", updates_per_sec=SAMPLES_PER_SEC)
streamer.subscribe("127.0.0.1", 5804, "estimation", updates_per_sec=SAMPLES_PER_SEC)


def update():
    sim_data = streamer.get_data("simulation")
    if "loads" not in sim_data:
        return
    pos_data = sim_data["loads"]["drivetrain"]
    sim_circle.data_source.data['x'] = [pos_data["position"][0], pos_data["velocity"][0]]
    sim_circle.data_source.data['y'] = [pos_data["position"][1], pos_data["velocity"][1]]

    est_data = streamer.get_data("estimation")
    if "loads" not in est_data:
        return
    pos_data = est_data["loads"]["drivetrain"]
    est_circle.data_source.data['x'] = [pos_data["position"][0], pos_data["velocity"][0]]
    est_circle.data_source.data['y'] = [pos_data["position"][1], pos_data["velocity"][1]]
    #update_plots(stream_data)


def update_plots(dictionary, prefix=""):
    for dict_key in dictionary:
        if isinstance(dictionary[dict_key], dict):
            update_plots(dictionary[dict_key], "/".join((prefix, dict_key)))
        elif hasattr(dictionary[dict_key], 'shape') and dictionary[dict_key].size > 1:
            i = 0
            for value in np.nditer(dictionary[dict_key]):
                set_plot("/".join((prefix, dict_key, str(i))), value)
                i += 1
        else:
            set_plot("/".join((prefix, dict_key)), dictionary[dict_key])

def set_plot(key, value):
    if key not in value_plots:
        fig = figure(x_range=(0, SAMPLES_PER_SEC), y_range=(-1, 1), toolbar_location=None)
        fig.border_fill_color = 'black'
        fig.background_fill_color = 'black'
        fig.outline_line_color = None
        fig.grid.grid_line_color = None
        value_plots[key] = fig
        value_lines[key] = fig.line(
                [i for i in range(SAMPLES_PER_SEC*SECS_REMEMBERED)],
                [0 for _ in range(SAMPLES_PER_SEC*SECS_REMEMBERED)],
                color='#A6CEE3', legend=key)
    if value < value_plots[key].data_source.data['y_range'][0]:
        value_plots[key].data_source.data['y_range'][0] = value
    if value > value_plots[key].data_source.data['y_range'][1]:
        value_plots[key].data_source.data['y_range'][1] = value
    value_lines[key].data_source.data['y'].pop()
    value_lines[key].data_source.data['y'].append(value)

curdoc().add_periodic_callback(update, 1000/SAMPLES_PER_SEC)
