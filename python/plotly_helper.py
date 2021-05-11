import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import tifffile as tiff
from scipy.ndimage import center_of_mass
import random
import plotly.subplots as sbplts
from plotly.subplots import make_subplots
import plotly.express as px
import pathlib
import plotly.graph_objects as go
import cv2

css_colors = [
'darkviolet', 
'purple', 
'navy', 
'blue', 
'cyan', 
'green', 
'greenyellow', 
'olive', 
'lime', 
'yellowgreen',
'black', 
'gray', 
'brown', 
'fuchsia', 
'gold', 
'orange', 
'red', 
'silver', 
'yellow']

def show_image(img, title, dash = False, gray=False, ann=None):
        if(gray == True):
            fig = px.imshow(img, color_continuous_scale='gray')
        else:
            fig = px.imshow(img)
        fig.update_layout(coloraxis_showscale=False)
        fig.update_xaxes(showticklabels=False)
        fig.update_yaxes(showticklabels=False)
        fig.update_layout(
            font_family="Calibri",
            font_color="blue",
            title_font_family="Calibri",
            title_font_color="Black",
            legend_title_font_color="green",
            title = {
                'text': title,
                'y':0.95,
                'x':0.5,
                'xanchor': 'center',
                'yanchor': 'top'}
        )

        if(ann is not None):
            for an in ann:
                if('gt' in an[2].lower()):
                    fig.add_annotation( x=an[1], y=an[0], text='<b>' + an[2] + '</b>',
                                        showarrow=True,
                                        arrowcolor='red', arrowwidth=2, arrowhead=2,
                                        font={'family':'Calibri', 'size':16, 'color':'silver'},
                                        yshift=0, hovertext=an[2])
                else:
                    fig.add_annotation( x=an[1], y=an[0], text='<b>' + an[2] + '</b>',
                                        showarrow=True,
                                        arrowcolor='green', arrowwidth=2, arrowhead=2,
                                        font={'family':'Calibri', 'size':16, 'color':'silver'},
                                        ayref='y', ay=an[0] + 10,
                                        yshift=0, hovertext=an[2])


        fig.update_traces(hovertemplate="x: %{x} <br>y: %{y} <br>Intensity: %{z}", name='')
        if(dash == False):
            fig.show()    
        else:
            return fig

class plot_each:

    def __init__(self):
        pass

    def show_image(img, title, dash = False, gray=False, ann=None):
        return show_image(img, title, dash = dash, gray = gray, ann = ann)

    def show_accuracy_results(eval_info, dash = False):
        
        if(dash == True):
            fig_list = []
            fig_list.append(show_image(eval_info['output_vs_gt'], 'Prediction (yellow) vs Ground Truth (cyan)', ann = eval_info['output_vs_gt_ann'], dash = True))
        else:
            show_image(eval_info['output_vs_gt'], 'Prediction (yellow) vs Ground Truth (cyan)', ann = eval_info['output_vs_gt_ann'])
        
        fig = px.imshow(eval_info['consensus_IoU'], zmin=0, zmax=1, color_continuous_scale='spectral',
             x = ['GT ' + str(c) for c in range(eval_info['num_gt_masks'])], 
               y = ['Pred ' + str(c) for c in range(eval_info['num_eval_masks'])])
        fig.update_layout(
            font_family="Calibri",
            font_color="blue",
            title_font_family="Calibri",
            title_font_color="Black",
            legend_title_font_color="green",
            title = {
                'text': "IoU Matrix",
                'y':0.95,
                'x':0.5,
                'xanchor': 'center',
                'yanchor': 'top'},
            xaxis_title = 'Ground Truth',
            yaxis_title = 'Prediction'
        )
        fig.update_traces(hovertemplate="x: %{x} <br>y: %{y} <br>Score: %{z}", name='')
        if(dash == True):
            fig_list.append(fig)
        else:
            fig.show()
        
        
        fig = go.Figure(layout=go.Layout(width=1200, height=600, margin = {'l' : 700}))
        fig.add_trace(go.Scatter(x = eval_info['consensus_thresholds'], y = eval_info['consensus_precision'], name='precision'))
        fig.add_trace(go.Scatter(x = eval_info['consensus_thresholds'], y = eval_info['consensus_recall'], name='recall'))
        fig.add_trace(go.Scatter(x = eval_info['consensus_thresholds'], y = eval_info['consensus_f1'], name='F1'))
        fig.add_trace(go.Scatter(x=[eval_info['representative_iou'], eval_info['representative_iou']], y=[0,1], 
        mode="lines", name="indication", line={'color':'gray','dash':'dashdot'}, showlegend=False))
        indices = np.where(eval_info['consensus_thresholds'] >= eval_info['representative_iou'])
        fig.update_layout(
            font_family="Calibri",
            font_color="blue",
            title_font_family="Calibri",
            title_font_color="Black",
            legend_title_font_color="green",
            title = {
                'text': 'F1 = %.2f at IoU = %.1f' % (eval_info['consensus_f1'][indices[0][0]], eval_info['representative_iou']),
                'y':0.90,
                'x':0.75,
                'xanchor': 'center',
                'yanchor': 'top'},
            yaxis_title = 'Score',
            xaxis_title = 'IoU Threshold'
        )

        fig.add_annotation(x = eval_info['representative_iou'], y = eval_info['consensus_f1'][indices[0][0]], 
                        text = "F1:" + str((round(eval_info['consensus_f1'][indices[0][0]], 2))), arrowhead=True, arrowsize=2, 
                        arrowcolor='indigo', arrowwidth=2, font={'family':'Calibri', 'size':16, 'color':'indigo'})

        if(dash == True):
            fig_list.append(fig)
            return fig_list
        else:
            fig.show()

    def display_df(df):
        with pd.option_context('display.max_rows', None, 'display.max_columns', None):
            display(df)

    def plot_accuracy_variance(eval_info, dash = False):

        def plot_scores(eval_info, idx, intitle, fig):
            if(idx == 0):
                sl = True
            else:
                sl = False
            for gt_id in eval_info['gt_ids']:
                fig.add_trace(go.Scatter(x = eval_info['consensus_thresholds'], y = eval_info[gt_id]['scores'][idx], 
                                         name=gt_id, marker_color=color_gtid[gt_id], showlegend=sl), row = 1, col = idx + 1)
            fig.add_trace(go.Scatter(x=[eval_info['representative_iou'], eval_info['representative_iou']], y=[0,1], 
            mode="lines", name="indication", line={'color':'gray','dash':'dashdot'}, showlegend=False), row = 1, col = idx + 1)
            indices = np.where(eval_info['consensus_thresholds'] >= eval_info['representative_iou'])
            fig.update_xaxes(title_text="IoU Threshold", row=1, col=idx + 1)
            fig.update_yaxes(title_text=intitle, row=1, col=idx + 1)


        fig = make_subplots(rows=1, cols=3, subplot_titles=("F1 Score", "Precision", "Recall"))
        colors = [css_colors[x] for x in random.sample(range(1, len(css_colors)), len(eval_info['gt_ids']))]
    
        color_gtid = {}
        for i in range(len(eval_info['gt_ids'])):
            gt_id = eval_info['gt_ids'][i]
            color_gtid[gt_id] = colors[i]
            
        plot_scores(eval_info, 0, 'F1 Score', fig)

        plot_scores(eval_info, 1, 'Precision', fig)

        plot_scores(eval_info, 2, 'Recall', fig)    

        fig.update_layout(
            font_family="Calibri",
            font_color="blue",
            title_font_family="Calibri",
            title_font_color="Black",
            legend_title_font_color="green",
            title = {
                'text': 'Accuracy Variance',
                'x':0.5,
                'xanchor': 'center',
                'yanchor': 'top'},
        )

        if(dash == False):
            fig.show()   
        else:
            return fig

    def show_individual_evaluation(eval_info, dash = False):
        if(dash == False):
            for gt_id in eval_info['gt_ids']:
                show_image(eval_info[gt_id]['output'], 'Predicted (yellow) vs GT (cyan) by ' + gt_id, ann = eval_info[gt_id]['output_ann'])
        else:
            fig_list = []
            for gt_id in eval_info['gt_ids']:
                fig_list.append(show_image(eval_info[gt_id]['output'], 'Predicted (yellow) vs GT (cyan) by ' + gt_id, dash = True, ann = eval_info[gt_id]['output_ann']))
            return fig_list

class plot_all:

    def __init__(self):
        pass

    def plot_accuracy_summary(eval_info, dash = False):
        
        fig = make_subplots(rows=1, cols=4, subplot_titles=("All datsets", "16x datasets", "20x datasets", "40x datasets"))
        colors = [css_colors[x] for x in random.sample(range(1, len(css_colors)), 3)]
        layout = go.Layout(yaxis=dict(scaleanchor="x", scaleratio=1))

        def plot_f1_score(eval_info, idx, sl):
            fig.add_trace(go.Scatter(x = eval_info['thresholds'], y = eval_info['df'][idx]['Precision'], name = 'Precision', 
                                    marker_color=colors[0], showlegend = sl), row = 1, col = idx + 1)
            fig.add_trace(go.Scatter(x = eval_info['thresholds'], y = eval_info['df'][idx]['Recall'], name = 'Recall', 
                                    marker_color=colors[1], showlegend = sl), row = 1, col = idx + 1)
            fig.add_trace(go.Scatter(x = eval_info['thresholds'], y = eval_info['df'][idx]['F1'], name = 'F1 Score', 
                                    marker_color=colors[2], showlegend = sl), row = 1, col = idx + 1)
            fig.add_trace(go.Scatter(x=[eval_info['representative_iou'], eval_info['representative_iou']], y=[0,1], 
        mode="lines", name="indication", line={'color':'gray','dash':'dashdot'}, showlegend=False), row = 1, col = idx + 1)
            
            fig.update_xaxes(title_text = 'IoU Threshold', row = 1, col = idx + 1)
            fig.update_yaxes(title_text = 'Score', row = 1, col = idx + 1)
            indices = np.where(eval_info['thresholds'] >= eval_info['representative_iou'])
            
            fig.add_annotation(x = eval_info['representative_iou'], y = eval_info['df'][idx]['F1'][indices[0][0]], 
                        xref = 'x'  + str(idx + 1), yref = 'y' + str(idx + 1),
                        text = "F1:" + str((round(eval_info['df'][idx]['F1'][indices[0][0]], 2))), arrowhead=True, arrowsize=2, 
                        arrowcolor='indigo', arrowwidth=2, font={'family':'Calibri', 'size':16, 'color':'indigo'})
            
        plot_f1_score(eval_info, 0, True)
        plot_f1_score(eval_info, 1, False)
        plot_f1_score(eval_info, 2, False)
        plot_f1_score(eval_info, 3, False)
        
        
        fig.update_layout(
        font_family="Calibri",
        font_color="blue",
        title_font_family="Calibri",
        title_font_color="Black",
        legend_title_font_color="green",
        title = {
            'text': 'Accuracy Summary',
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'},
        )
        
        if(dash == False):
            fig.show()
        else:
            return fig
    
    def plot_per_dataset_score(eval_info, idx, dash = False):
        
        fig = px.bar(eval_info['df_each'], x='Dataset', y=idx, color='Magnification')
        fig.update_xaxes(type='category')
        fig.update_layout(
            font_family="Calibri",
            font_color="blue",
            title_font_family="Calibri",
            title_font_color="Black",
            legend_title_font_color="green",
            title = {
                'text': 'Per-dataset ' + idx + ' score at IoU = %.1f' %eval_info['representative_iou'],
                'x':0.5,
                'xanchor': 'center',
                'yanchor': 'top'},
            )
        if(dash == False):
            fig.show()
        else:
            return fig

    def show_individual_accuracy(eval_info, dash = False):
        
        def plot_score_multiple_gt(eval_info, idx, label, fig):

            if(idx == 0):
                sl = True
            else:
                sl = False
            
            for gt_id in eval_info['gt_ids']:
                fig.add_trace(go.Scatter(x = eval_info['thresholds'], y = eval_info[gt_id]['scores'][cols[idx]], 
                                         name=gt_id, marker_color=color_gtid[gt_id], showlegend=sl), row = 1, col = idx + 1)
            
            fig.add_trace(go.Scatter(x=[eval_info['representative_iou'], eval_info['representative_iou']], y=[0,1], 
                mode="lines", name="indication", line={'color':'gray','dash':'dashdot'}, showlegend=False), row = 1, col = idx + 1)
            fig.update_xaxes(title_text="IoU Threshold", row=1, col=idx + 1)
            fig.update_yaxes(title_text=label, row=1, col=idx + 1)
        
        
        
        cols = ['F1', 'Precision', 'Recall']
        fig = make_subplots(rows=1, cols=3, subplot_titles=("F1 Score", "Precision", "Recall"))
        colors = [css_colors[x] for x in random.sample(range(1, len(css_colors)), len(eval_info['gt_ids']))]
        
        color_gtid = {}
        for i in range(len(eval_info['gt_ids'])):
            gt_id = eval_info['gt_ids'][i]
            color_gtid[gt_id] = colors[i]
        
        plot_score_multiple_gt(eval_info, 0, 'F1 Score', fig)
        plot_score_multiple_gt(eval_info, 1, 'Precision', fig)
        plot_score_multiple_gt(eval_info, 2, 'Recall', fig)
        
        fig.update_layout(
            font_family="Calibri",
            font_color="blue",
            title_font_family="Calibri",
            title_font_color="Black",
            legend_title_font_color="green",
            title = {
                'text': 'Accuracy for Individual GTs',
                'x':0.5,
                'xanchor': 'center',
                'yanchor': 'top'},
        )
        
        if(dash == False):
            fig.show()
        else:
            return fig
