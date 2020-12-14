import plotly
import plotly.graph_objects as go
import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_table
from dash.dependencies import Input, Output
from dash.exceptions import PreventUpdate
from datetime import datetime, timedelta
import glob
import ntpath
import plotly.subplots as sbplts
import plotly.express as px
import os
import re
import sys
from dash_helper import get_run_info, get_eval_info, get_eval_all_info, if_eval_ready
from plotly_helper import plot_each as ph
from plotly_helper import plot_all as pa
import plotly.io as pio



def path_leaf(path):
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)

def format_time(x):
    hours = int(x//3600)
    minutes = int((x%3600)//60)
    seconds = int(x%60)

    return "{:d}:{:02d}:{:02d}".format(hours, minutes, seconds)

app = dash.Dash(__name__)
evals = {}
done_tabs = {}
is_finished = False
is_evaluate = False

overviewTab = []

heading_style = {
    'padding' : '20px' ,
    'backgroundColor' : '#1B2C3F',
    'borderTop': '20px solid #3333cc',
    'borderBottom': '20px solid #3333cc',
    'color': "#ffffff",
    'font-size': '200%',
    'font-family': 'Calibri',
    'text-decoration':'underline overline dotted orange'
}

tab_style = {
    'width': '100%',
    'borderTop': '1px solid #d6d6d6',
    'borderBottom': '1px solid #d6d6d6',
    'padding': '6px',
    'fontWeight': 'bold',
    'text-align':'left'
}

tab_selected_style = {
    'width': '100%',
    'borderTop': '1px solid #d6d6d6',
    'borderBottom': '1px solid #d6d6d6',
    'backgroundColor': '#119DFF',
    'color': 'white',
    'padding': '6px',
    'fontWeight': 'bold',
    'text-align':'center'
}

tab_heading_style = {
    # 'display': 'inline',
    # 'padding' : '10px' ,
    'backgroundColor' : '#2B7C3F',
    'color': "#ffffff",
    'font-size': '150%',
    'font-family': 'Calibri',
    'text-align': 'center',
    'fontWeight': 'bold'
}


app.layout = html.Div([
    dcc.Interval(
                id='custom_interval',
                disabled=False,     #if True, the counter will no longer update
                interval=1*5000,    #increment the counter n_intervals every interval milliseconds
                n_intervals=0,      #number of times the interval has passed
                max_intervals=-1,    #number of times the interval will be fired.
                                    #if -1, then the interval has no limit (the default)
                                    #and if 0 then the interval stops running.
    ),

    html.Div([
        html.H1("Voltage Imaging Analysis"),
        ], style = heading_style),

    dcc.Tabs(id='tabs', vertical=True)

])


def get_overviewTab_children(info, scope='simple'):

    tinfo = info['tinfo']
    is_finished = False
    children = []

    children.append(
        html.H2("Overview", style = tab_heading_style),
    )

    children.append(html.H3("File Timing Information:", style={"padding": '10px', 'textAlign':'center', 'color':'#000080'}))

    children.append(dash_table.DataTable(
        id='table',
        columns=[{"name": i, "id": i} for i in tinfo.columns],
        data=tinfo.to_dict('records'),
        style_table={
            'width': '25%',
            'padding-left': '18%',
            'textAlign': 'center'
        },
        style_data_conditional=[
            {
                'if': {'row_index': 'odd'},
                'backgroundColor': 'rgb(248, 248, 248)'
            }
        ],
        style_header={
            'backgroundColor': 'rgb(230, 230, 230)',
            'fontWeight': 'bold'
        },
        style_cell={
            'textAlign': 'center',
            'minWidth': '100px', 
            'width': '200px', 
            'maxWidth': '380px',
        }
    ))
    
    if(scope == 'full'):
        if(is_evaluate == True and info['is_finished'] == True):
            eval_info = get_eval_all_info()

            children.append(html.H2("Evaluation Results:", style={"padding": '10px', 'textAlign':'center', 'color':'teal'}))

            children.append(html.H1("F1 Score: " + str(round(eval_info['rep'][0], 2)), style={"padding": '10px', 'textAlign':'left', 'color':'#008000'}))

            fig = pa.plot_accuracy_summary(eval_info, dash=True)
            children.append(dcc.Graph(id="Accuracy Summary", figure=fig))

            fig = pa.plot_per_dataset_score(eval_info, 'F1', dash=True)
            children.append(dcc.Graph(id="per_datasets", figure=fig))
            fig = pa.plot_per_dataset_score(eval_info, 'Precision', dash=True)
            children.append(dcc.Graph(id="per_datasets", figure=fig))
            fig = pa.plot_per_dataset_score(eval_info, 'Recall', dash=True)
            children.append(dcc.Graph(id="per_datasets", figure=fig))
            fig = pa.show_individual_accuracy(eval_info, dash=True)
            children.append(dcc.Graph(id="indiv accuracy", figure=fig))
        is_finished = True

    return children, is_finished


def get_fileTab_children(tag, info):
    children = []
    finished = False

    children.append(
        html.H2("File Tag: " + tag, style = tab_heading_style),
    )

    children.append(html.H3("File Execution Information:", style={"padding": '10px', 'textAlign':'center', 'color':'#000080'}))

    children.append(dash_table.DataTable(
        id='table',
        columns=[{"name": i, "id": i} for i in info[tag]['tinfo'].columns],
        data=info[tag]['tinfo'].to_dict('records'),
        style_table={
            'width': '25%',
            'padding-left': '18%',
            'textAlign': 'center'
        },
        style_data_conditional=[
            {
                'if': {'row_index': 'odd'},
                'backgroundColor': 'rgb(248, 248, 248)'
            }
        ],
        style_header={
            'backgroundColor': 'rgb(230, 230, 230)',
            'fontWeight': 'bold'
        },
        style_cell={
            'textAlign': 'center',
            'minWidth': '100px', 
            'width': '200px', 
            'maxWidth': '380px',
        }
    ))
    if('mask' in info[tag].keys()):
        
        children.append(html.H3("Segmented Output:", style={"padding": '10px', 'textAlign':'center', 'color':'#000080'}))
        fig1 = px.imshow(info[tag]['pred'], color_continuous_scale='gray')
        fig1.update_layout(coloraxis_showscale=False)
        children.append(dcc.Graph(id="File Mask", figure=fig1))

        children.append(html.H3("Demixed Output:", style={"padding": '10px', 'textAlign':'center', 'color':'#000080'}))
        fig1 = px.imshow(info[tag]['mask'], color_continuous_scale='gray')
        fig1.update_layout(coloraxis_showscale=False)
        children.append(dcc.Graph(id="File Mask", figure=fig1))

        

        global is_evaluate
        if(is_evaluate == True):
            fname = info[tag]['fname']
            if(if_eval_ready(fname) == True):

                if(tag not in evals.keys()):
                    evals[tag] = get_eval_info(fname)
                
                children.append(html.H2("Evaluation Results:", style={"padding": '10px', 'textAlign':'center', 'color':'#000080'}))
                
                fig = ph.show_image(evals[tag]['summary_image'], 'Summary Image', gray=True, dash=True)
                children.append(dcc.Graph(id="Summary Image", figure=fig))
                
                fig = ph.show_image(evals[tag]['output_rois'], 'Predicted ROIs', dash=True, ann=evals[tag]['output_roi_ann'])
                children.append(dcc.Graph(id="output_rois", figure = fig))

                fig = ph.show_image(evals[tag]['gt_rois'], 'Ground Truth ROIs', dash=True, ann=evals[tag]['gt_roi_ann'])
                children.append(dcc.Graph(id="gt_rois", figure = fig))

                children.append(html.H3("Accuracy Information:", style={"padding": '10px', 'textAlign':'center', 'color':'#000080'}))

                fig_list = ph.show_accuracy_results(evals[tag], dash=True)
                for fig in fig_list:
                    children.append(dcc.Graph(id="accuracy plots", figure = fig))                

                children.append(html.H4("Prediction Score:", style={"padding": '10px', 'textAlign':'center', 'color':'#000080'}))

                children.append(dash_table.DataTable(
                    id='table',
                    columns=[{"name": i, "id": i} for i in evals[tag]['prediction_df'].columns],
                    data=evals[tag]['prediction_df'].round(2).to_dict('records'),
                    style_table={
                        'width': '25%',
                        'padding-left': '35%',
                        'textAlign': 'center'
                    },
                    style_data_conditional=[
                        {
                            'if': {'row_index': 'odd'},
                            'backgroundColor': 'rgb(248, 248, 248)'
                        }
                    ],
                    style_header={
                        'backgroundColor': 'rgb(230, 230, 230)',
                        'fontWeight': 'bold'
                    },
                    style_cell={
                        'textAlign': 'center',
                        'minWidth': '100px', 
                        'width': '200px', 
                        'maxWidth': '380px',
                    }
                ))

                children.append(html.P('\n\n', style = {  'padding' : '30px' }))
                children.append(html.H3("Individual Evaluations:", style={"padding": '10px', 'textAlign':'center', 'color':'#000080'}))

                children.append(html.H4("Prediction Evaluations:", style={"padding": '10px', 'textAlign':'center', 'color':'#000080'}))
                
                fig_list = ph.show_individual_evaluation(evals[tag], dash=True)
                for fig in fig_list:
                    children.append(dcc.Graph(id="individual eval plots", figure = fig))

                children.append(html.H4("Accuracy Variance:", style={"padding": '10px', 'textAlign':'center', 'color':'#000080'}))
                fig = ph.plot_accuracy_variance(evals[tag], dash=True)
                children.append(dcc.Graph(id="accuracy evaluations", figure = fig))

                finished = True
        else:
            finished = True

    return children, finished

@app.callback(Output('tabs', 'children'),
    [Input('custom_interval', 'n_intervals')]
)
def tab_callback(num):

    info = get_run_info()
    num_tabs = len(info['tags'])
    global is_finished
    global overviewTab

    if(len(done_tabs) != num_tabs):
        overviewTab, is_finished = get_overviewTab_children(info, scope='simple')
    else:
        if(is_finished == False):
            overviewTab, is_finished = get_overviewTab_children(info, scope='full')
    
    overviewtab = [dcc.Tab(label='Overview', id='tab1', value='OverviewTab', children = overviewTab, 
                    style=tab_style, selected_style=tab_selected_style)]

    file_tabs = []
    if(is_finished == False):

        for i in range(num_tabs):
            
            tag = info['tags'][i]
            if(tag not in done_tabs.keys()):
                children_vals, finished = get_fileTab_children(tag, info)
                if(finished == True):
                    done_tabs[tag] = children_vals
            else:
                children_vals = done_tabs[tag]

            file_tabs.append(
                dcc.Tab(label=str("File Tag: " + tag), id='tab%d' %i, value='Tab%d' %i, children = children_vals,
                 style=tab_style, selected_style=tab_selected_style)
            )

    else:
        for i in range(len(list(done_tabs.keys()))):
            tag = list(done_tabs.keys())[i]
            children_vals = done_tabs[tag]

            file_tabs.append(
                dcc.Tab(label=str("File Tag: " + tag), id='tab%d' %i, value='Tab%d' %i, children = children_vals,
                 style=tab_style, selected_style=tab_selected_style)
            )

    return overviewtab + file_tabs


def start_dash(dash_port, evaluate):
    global is_evaluate
    is_evaluate = evaluate
    app.run_server(port=dash_port, debug=True)



if __name__ == '__main__':
    start_dash(int(sys.argv[1]), bool(int(sys.argv[2])))
