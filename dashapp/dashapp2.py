

import sys,os
if  not os.path.abspath('./') in sys.path:
    sys.path.append(os.path.abspath('./'))
if  not os.path.abspath('../') in sys.path:
    sys.path.append(os.path.abspath('../'))

import dash
import dash_html_components as html
import dash_core_components as dcc
import plotly.graph_objs as go
from plotly.graph_objs.layout import Margin#,Font
from dash.dependencies import Input, Output,State
from dash.exceptions import PreventUpdate
from plotly.subplots import make_subplots

import dash_table
import pandas as pd
import numpy as np
import json
import inspect
import logging
import pdb
import copy
import base64
import io
import datetime
import pytz
import uuid
import itertools


# In[2]:


DEFAULT_LOG_PATH = './logfile.log'
DEFAULT_LOG_LEVEL = 'INFO'

def _new_uuid():
    return str(uuid.uuid1())

# def init_root_logger(logfile=DEFAULT_LOG_PATH,logging_level=DEFAULT_LOG_LEVEL):
def init_root_logger(logfile=DEFAULT_LOG_PATH,logging_level=None):
    level = logging_level
    if level is None:
        level = logging.INFO
    # get root level logger
    logger = logging.getLogger()
    if len(logger.handlers)>0:
        return logger
    logger.setLevel(logging.getLevelName(level))

    fh = logging.FileHandler(logfile)
    fh.setLevel(logging.DEBUG)
    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    # create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    # add the handlers to the logger
    logger.addHandler(fh)
    logger.addHandler(ch)   
    return logger

def str_to_date(d,sep='-'):
    try:
        dt = datetime.datetime.strptime(str(d)[:10],f'%Y{sep}%m{sep}%d')
    except:
        return None
    return dt

def grouper(n, it):
    '''
    split an iterable into sub-iterables, each with at most n elements in each group
    Example:
    grouper(3, "ABCDEFG") --> ABC DEF G

    :param n: elements per group
    :param it: python iterable which represents a group to be sub-divifed
    '''
    it = iter(it)
    return iter(lambda: list(itertools.islice(it, n)), [])

def generate_hilo_ranges(df,col,num_elements_per_group):
    '''
    Generate lists of tuples, with at most "num_elements_per_group" 
    as number of tuples in each list of tuples.
    
    Each tuple contains a  (low_value,high_value) for each group.
     
    :param df: DataFrame with a column whose values will be grouped
                 into a number of groups, where the group size <= num_elements_per_group
    :param col: column to group
    :param num_elements_per_group: max size each of group
    '''
    v = sorted(df[col].unique())
    n = int(len(v)/num_elements_per_group)
    n = 1 if n<1 else n
    g = grouper(n,v)
    return g

def generate_sub_dfs(df,col,num_elements_per_group):
    '''
    generate sub DataFrames where each sub DataFrame contains all of the
      values of a specific column that fall between 2 values.
    Example: 
    import pandas as pd
    from dashapp import dashapp2 as dashapp
    from IPython import display
    
    df = pd.DataFrame({'x':np.arange(100),'y':np.arange(100)*10})
    for df_sub in generate_sub_dfs(df,'x',10):
        xlow = df_sub.x.min()
        xhigh = df_sub.x.max()
        df_sub2 = df_sub.style.set_caption(f"<div style='text-align:center;'>{xlow} to {xhigh} </div>")
        display.display(df_sub2)
    
        
    :param df: DataFrame to be broken up into groups
    :param col: Column in DataFrame that holds grouping values
    :param num_elements_per_group: max size of each group
    '''
    sublist_generator = generate_hilo_ranges(df,col,num_elements_per_group)
    for sg in sublist_generator:
        df_sub = df[(df[col]>=sg[0]) & (df[col]<=sg[-1])].copy()
        yield df_sub

def flatten_columns(df,index,columns,values=None):
    df2 = pd.pivot_table(df,index=index,columns=columns)
    new_columns = df2.columns.values
    new_columns = df2.index.names + [c[1] for c in new_columns]
    df2.reset_index( drop=False, inplace=True)
    df2.columns = new_columns
    return df2


logger = init_root_logger(logging_level='DEBUG')

def stop_callback(errmess,logger=None):
    m = "****************************** " + errmess + " ***************************************"     
    if logger is not None:
        logger.debug(m)
    raise PreventUpdate()


def figure_crosshairs(fig):
    fig['layout'].hovermode='x'
    fig['layout'].yaxis.showspikes=True
    fig['layout'].xaxis.showspikes=True
    fig['layout'].yaxis.spikemode="toaxis+across"
    fig['layout'].xaxis.spikemode="toaxis+across"
    fig['layout'].yaxis.spikedash="solid"
    fig['layout'].xaxis.spikedash="solid"
    fig['layout'].yaxis.spikethickness=1
    fig['layout'].xaxis.spikethickness=1
    fig['layout'].spikedistance=1000
    return fig
    

def plotly_plot(df_in,x_column,plot_title=None,
                y_left_label=None,y_right_label=None,
                bar_plot=False,width=800,height=400,
                number_of_ticks_display=20,
                yaxis2_cols=None,
                x_value_labels=None,
               modebar_orientation='v',modebar_color='grey',
               legend_orientation='h',
               autosize=True):
    ya2c = [] if yaxis2_cols is None else yaxis2_cols
    ycols = [c for c in df_in.columns.values if c != x_column]
    # create tdvals, which will have x axis labels
#     td = list(df_in[x_column]) 
    td = df_in[x_column].values
    nt = len(df_in)-1 if number_of_ticks_display > len(df_in) else number_of_ticks_display
    spacing = len(td)//nt
    tdvals = td[::spacing]
    tdtext = tdvals
    if x_value_labels is not None:
        tdtext = [x_value_labels[i] for i in tdvals]
    # create data for graph
    data = []
    # iterate through all ycols to append to data that gets passed to go.Figure
    for ycol in ycols:
        if bar_plot:
            b = go.Bar(x=td,y=df_in[ycol],name=ycol,yaxis='y' if ycol not in ya2c else 'y2')
        else:
            b = go.Scatter(x=td,y=df_in[ycol],name=ycol,yaxis='y' if ycol not in ya2c else 'y2')
        data.append(b)

    # create a layout
    
    layout = go.Layout(
        title=plot_title,
        xaxis=dict(
            ticktext=tdtext,
            tickvals=tdvals,
            tickangle=45,
            type='category',
            showspikes=True,
            spikemode='toaxis+across',
            spikedash='solid',
            spikethickness=1,
            ),
        yaxis=dict(
            title='y main' if y_left_label is None else y_left_label,
            showspikes=True,
            spikemode='toaxis+across',
            spikedash='solid',
            spikethickness=1,
        ),
        yaxis2=dict(
            title='y alt' if y_right_label is None else y_right_label,
            overlaying='y',
            side='right'),
        autosize=autosize,
#         autosize=False,
#         width=width,
#         height=height,
        margin=Margin(
            b=100
        ),
        modebar={'orientation': modebar_orientation,'bgcolor':modebar_color},
        hovermode='x',
        spikedistance=1000        
    )

    fig = go.Figure(data=data,layout=layout)
    layout_legend = {} if legend_orientation == 'v' else {'orientation':legend_orientation,'x':0, 'y':1.1}
    fig.update_layout(
        title={
            'text': plot_title,
            'y':0.9,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'},
#         legend={'orientation':legend_orientation,'x':0, 'y':1.1},
        legend=layout_legend,
        )
    return fig

def plotly_shaded_rectangles(beg_end_date_tuple_list,fig):
    ld_shapes = []
    for beg_end_date_tuple in beg_end_date_tuple_list:
        ld_beg = beg_end_date_tuple[0]
        ld_end = beg_end_date_tuple[1]
        ld_shape = dict(
            type="rect",
            # x-reference is assigned to the x-values
            xref="x",
            # y-reference is assigned to the plot paper [0,1]
            yref="paper",
            x0=ld_beg,
            y0=0,
            x1=ld_end,
            y1=1,
            fillcolor="LightSalmon",
            opacity=0.5,
            layer="below",
            line_width=0,
        )
        ld_shapes.append(ld_shape)

    fig.update_layout(shapes=ld_shapes)
    return fig

def create_subplot_df_figure(col_name_list,x_column,row_list,
                             col_list,is_secondary_list,yaxis_title_list):
    y_defs = [['name','x_column','row','col','is_secondary','yaxis_title']]
    df_figure = pd.DataFrame(
        {
            'name':col_name_list,
            'x_column':[x_column for _ in range(len(col_name_list))],
            'row':row_list,
            'col':col_list,
            'is_secondary':is_secondary_list,
            'yaxis_title':yaxis_title_list
        }
    )
    return df_figure

def plotly_subplots(df,df_figure,num_ticks_to_display=20,title="",
                   subplot_titles=None):
    '''
    Create a plotly figure instance, where the x,y data for the figure come from column values in 
    the DataFrame df, and the values that determine where this data appears in the subplots comes 
    from values in the DataFrame df_figure.
    
    
    :param df: Pandas DataFrame with data to graph in it's columns.  Required
    :param df_figure: DataFrame that defines the placement and style of each
                       plotly.graph_objs graph object that is associated with columns in df. Required 
    :param num_ticks_to_display: Number of xaxis ticks to display.  Default= 20
    :param title:  Main figure title.  Default is ''
    :param subplot_titles: titles for subplots.  Default = None
    
    Example:
    
    df = pd.DataFrame({'x':[1,2,3,4,5],
                       'y1':[10,11,12,13,14],'y2':[19,18,17,16,15],
                       'y3':[20,21,22,23,24],'y4':[29,28,27,26,25]})
    # define rows of df_fig, which defines the look of the subplots
    y_defs = [
        ['name','x_column','row','col','is_secondary','yaxis_title'],
        ['y1','x',1,1,False,'y1 values'],
        ['y2','x',1,1,True, 'y2 values'],
        ['y3','x',2,1,False,'y3 values'],
        ['y4','x',2,1,True, 'y4 values']
    ]
    
    df_fig = pd.DataFrame(y_defs[1:],columns=y_defs[0])
    fig_title = "Example with 2 rows and 1 column, and 4 lines"
    sp_titles = ['y1 and y2 plots','y3 and y4 plots']
    fig_test = plotly_subplots(df,df_fig,num_ticks_to_display=15,title=fig_title,subplot_titles = sp_titles)
    iplot(fig_test)
        
    '''
    # determine number of rows and columns in subplot grid
    rows = int(df_figure['row'].max())
    cols = int(df_figure['col'].max())
    
    # Create a matrix of yaxis subscript numbers: 
    #   Each cell in the subplot grid has 2 yaxis, where:
    #      The left yaxis = f"yaxis{(row-1)*cols + 1}
    #      The right yaxis = f"yaxis{(row-1)*cols + 1 + col}
    #   The right axis is only used if the is_secondary column of the df_yaxis matrix for
    #      a specific graph object is set to True.
    i = 0
    yaxis_number_matrix = []
    for _ in range(rows):
        for _ in range(cols):
            yaxis_number_matrix.append([i+1,i+2])
            i+=2

    vert_spacing = .05
    specs = [[{"secondary_y": True} for i in range(cols)] for _ in range(rows)]
    if subplot_titles is None:
        fig = make_subplots(rows=rows, cols=cols,
                        specs=specs,shared_xaxes=True,vertical_spacing=vert_spacing)
    else:
        vert_spacing +=.03
        fig = make_subplots(rows=rows, cols=cols,
                        specs=specs,shared_xaxes=True,
                           subplot_titles=subplot_titles,vertical_spacing=vert_spacing)

    df_yp = df_figure.copy()
    df_yp['row'].fillna(1)
    df_yp['col'].fillna(1)
    df_yp.is_secondary.fillna(False)
    df_yp.yaxis_title.fillna('')
    yaxis_title_dict = {}
    # add traces
    for i in range(len(df_yp)):
        r = df_yp.iloc[i]
        td = df[r.x_column].values
        nt = len(df)-1 if num_ticks_to_display > len(df) else num_ticks_to_display
        spacing = len(td)//nt
        tdvals = td[::spacing]
        tdtext = tdvals
#         if x_value_labels is not None:
#             tdtext = [x_value_labels[i] for i in tdvals]
        x = td
        y = df[r['name']].values
        row = int(r['row'])
        col = int(r['col'])
        is_secondary = r.is_secondary
        yaxis_title = r.yaxis_title
        name=r['name']
        go_trace = go.Scatter(x=x, y=y, name=name)
        if 'trace' in df_yp.columns.values:
            go_trace = r['trace']
        fig.add_trace(
#             go.Scatter(x=x, y=y, name=name),
            go_trace,
            row=row, col=col, secondary_y=is_secondary)
        fig.update_xaxes(
            ticktext=tdtext,
            tickvals=tdvals,
            tickangle=45,
            type='category', row=row, col=col)
        
        yaxis_nums = yaxis_number_matrix[(row-1)*cols + (col-1)]
        yaxis_num = yaxis_nums[1] if is_secondary else yaxis_nums[0]
        yaxis_num = '' if yaxis_num == 1 else yaxis_num
        yaxis_name = f"yaxis{yaxis_num}"
        yaxis_title_dict[yaxis_name] = yaxis_title
#         fig.update_yaxes(title_text=yaxis_title, row=row, col=col)  
    
    fig.update_layout(
        title={
            'text': title,
            'y':0.9,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'},
    )
    fig = figure_crosshairs(fig)
    figdictstring = fig.to_json()
    figdict = json.loads(figdictstring)
    for yax in yaxis_title_dict.keys():
        figdict['layout'][yax]['title'] = yaxis_title_dict[yax]
    return figdict


def print_axis_info(ff):
    print(ff['layout']['title']['text'])
    print(ff['layout'].keys())
    xsd = lambda k,j:None if j not in ff['layout'][k] else ff['layout'][k][j]
    xs = [(k,[xsd(k,j) for j in ['anchor','domain','type','title']]) for k in ff['layout'].keys() if 'xaxis' in k]
    print(xs)
    ysd = lambda k,j:None if j not in ff['layout'][k] else ff['layout'][k][j]
    # ys = [(k,fig2['layout'][k]) for k in fig2['layout'].keys() if 'yaxis' in k]
    ys = [(k,[ysd(k,j) for j in ['anchor','domain','overlaying','title']]) for k in ff['layout'].keys() if 'yaxis' in k]
    print(ys)
    
    
class PlotlyCandles():
    BAR_WIDTH=.5
    def __init__(self,df,date_column='date',
                 title='candle plot',number_of_ticks_display=20,
                 price_rounding=4,modebar_orientation='v',modebar_color='grey'):
        '''
        Use Plotly to create a financial candlestick chart.
        The DataFrame df_in must have columns called:
         'date','open','high','low','close'
        
        :param df:
        :param title:
        :param number_of_ticks_display:
        '''
        self.df = df.copy()
        #  and make sure the first index is 1 NOT 0!!!!!!
        self.df.index = np.array(list(range(len(df))))+1
        
        self.title = title
        self.number_of_ticks_display = number_of_ticks_display
        self.price_rounding=price_rounding
        self.modebar_orientation = modebar_orientation
        self.modebar_color = modebar_color
        self.date_column=date_column
        
    def get_candle_shapes(self):
        df = self.df.copy()
        xvals = df.index.values #chg    
        lows = df.low.values
        highs = df.high.values
        closes = df.close.values
        opens = df.open.values
        df['is_red'] = df.open>=df.close
        is_reds = df.is_red.values
        lines_below_box = [{
                    'type': 'line',
                    'x0': xvals[i],
                    'y0': lows[i],
                    'x1': xvals[i],
                    'y1': closes[i] if is_reds[i] else opens[i],
                    'line': {
                        'color': 'rgb(55, 128, 191)',
                        'width': 1.5,
                    }
                } for i in range(len(xvals))
        ]

        lines_above_box = [{
                    'type': 'line',
                    'x0': xvals[i],
                    'y0': opens[i] if is_reds[i] else closes[i],
                    'x1': xvals[i],
                    'y1': highs[i],
                    'line': {
                        'color': 'rgb(55, 128, 191)',
                        'width': 1.5,
                    }
                }for i in range(len(xvals))
        ]


        boxes = [{
                    'type': 'rect',
                    'xref': 'x',
                    'yref': 'y',
                    'x0': xvals[i]- PlotlyCandles.BAR_WIDTH/2,
                    'y0': closes[i] if is_reds[i] else opens[i],
                    'x1': xvals[i]+ PlotlyCandles.BAR_WIDTH/2,
                    'y1': opens[i] if is_reds[i] else closes[i],
                    'line': {
                        'color': 'rgb(55, 128, 191)',
                        'width': 1,
                    },
                    'fillcolor': 'rgba(255, 0, 0, 0.6)' if is_reds[i] else 'rgba(0, 204, 0, 0.6)',
                } for i in range(len(xvals))
        ]
        shapes = lines_below_box + boxes + lines_above_box
        return shapes

    
    def get_figure(self):
        '''
        Use Plotly to create a financial candlestick chart.
        The DataFrame df_in must have columns called:
         'date','open','high','low','close'
        '''
        # Step 0: get important constructor values (so you don't type 'self.'' too many times)
        df_in = self.df.copy()
        title=self.title
        number_of_ticks_display=self.number_of_ticks_display
        
        # Step 1: only get the relevant columns and sort by date
#         cols_to_keep = ['date','open','high','low','close','volume']
#         df = df_in[cols_to_keep].sort_values('date')
        cols_to_keep = [self.date_column,'open','high','low','close','volume']
        df = df_in[cols_to_keep].sort_values(self.date_column)
        # Step 2: create a data frame for "green body" days and "red body" days
        # Step 3: create the candle shapes that surround the scatter plot in trace1
        shapes = self.get_candle_shapes()

        # Step 4: create an array of x values that you want to show on the xaxis
        spaces = len(df)//number_of_ticks_display
        indices = list(df.index.values[::spaces]) + [max(df.index.values)]
#         tdvals = df.loc[indices].date.values
        tdvals = df.loc[indices][self.date_column].values

        # Step 5: create a layout
        layout1 = go.Layout(
            showlegend=False,
            title={
            'text': title,
            'y':0.9,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'},
            modebar={'orientation': self.modebar_orientation,'bgcolor':self.modebar_color},
            hovermode='x',
#             margin = dict(t=50),
            xaxis = go.layout.XAxis(
                tickmode = 'array',
                tickvals = indices,
                ticktext = tdvals,
                tickangle=45,
                showgrid = True,
                showticklabels=True,
                anchor='y2', 
            ),       
            yaxis1 = go.layout.YAxis(
                range =  [min(df.low.values), max(df.high.values)],
                domain=[.17,1]
            ),
            yaxis2 = go.layout.YAxis(
                range =  [0, max(df.volume.values)],
                domain=[0,.15]
            ),
            shapes = shapes
        )

        # Step 6: create a scatter object, and put it into an array
        def __hover_text(r):
#             d = r.date
            d = r[self.date_column]
            o = round(r.open,self.price_rounding)
            h = round(r.high,self.price_rounding)
            l = round(r.low,self.price_rounding)
            c = round(r.close,self.price_rounding)
            v = r.volume
            t = f'date: {d}<br>open: {o}<br>high: {h}<br>low: {l}<br>close: {c}<br>volume: {v}' 
            return t
        df['hover_text'] = df.apply(__hover_text,axis=1)
        hover_text = df.hover_text.values

        # Step 7: create scatter (close values) trace.  The candle shapes will surround the scatter trace
#         trace1 = go.Scatter(
        trace1 = go.Scattergl(
            x=df.index.values,
            y=df.close.values,
            mode = 'markers',
            text = hover_text,
            hoverinfo = 'text',
            xaxis='x',
            yaxis='y1',
            marker={'symbol':'line-ew'}
        )

        # Step 8: create the bar trace (volume values)
        trace2 = go.Bar(
            x=df.index.values,
            y=df.volume.values,
            width = PlotlyCandles.BAR_WIDTH,
            xaxis='x',
            yaxis='y2'
        )

        # Step 9: create the final figure and pass it back to the caller
        fig1 = {'data':[trace1,trace2],'layout':layout1}
#         fig1['layout'].margin = {'t':150,'l':3,'r':3}
        fig1['layout'].margin = {'t':150}
        fig1['layout'].hovermode='x'
        fig1['layout'].yaxis.showspikes=True
        fig1['layout'].xaxis.showspikes=True
        fig1['layout'].yaxis.spikemode="toaxis+across"
        fig1['layout'].xaxis.spikemode="toaxis+across"
        fig1['layout'].yaxis.spikedash="solid"
        fig1['layout'].xaxis.spikedash="solid"
        fig1['layout'].yaxis.spikethickness=1
        fig1['layout'].xaxis.spikethickness=1
        fig1['layout'].spikedistance=1000
        return fig1
    
    def plot(self):
        fig = self.get_figure()
        return fig    
DEFAULT_TIMEZONE = 'US/Eastern'


# ************************* define useful factory methods *****************

def parse_contents(contents):
    '''
    app.layout contains a dash_core_component object (dcc.Store(id='df_memory')), 
      that holds the last DataFrame that has been displayed. 
      This method turns the contents of that dash_core_component.Store object into
      a DataFrame.
      
    :param contents: the contents of dash_core_component.Store with id = 'df_memory'
    :returns pandas DataFrame of those contents
    '''
    c = contents.split(",")[1]
    c_decoded = base64.b64decode(c)
    c_sio = io.StringIO(c_decoded.decode('utf-8'))
    df = pd.read_csv(c_sio)
    # create a date column if there is not one, and there is a timestamp column instead
    cols = df.columns.values
    cols_lower = [c.lower() for c in cols] 
    if 'date' not in cols_lower and 'timestamp' in cols_lower:
        date_col_index = cols_lower.index('timestamp')
        # make date column
        def _extract_dt(t):
            y = int(t[0:4])
            mon = int(t[5:7])
            day = int(t[8:10])
            hour = int(t[11:13])
            minute = int(t[14:16])
            return datetime.datetime(y,mon,day,hour,minute,tzinfo=pytz.timezone(DEFAULT_TIMEZONE))
        # create date
        df['date'] = df.iloc[:,date_col_index].apply(_extract_dt)
    return df

def make_df(dict_df):
    if type(dict_df)==list:
        if type(dict_df[0])==list:
            dict_df = dict_df[0]
        return pd.DataFrame(dict_df,columns=dict_df[0].keys())
    else:
        return pd.DataFrame(dict_df,columns=dict_df.keys())

def add_df_title(df,title):
    return df.style.set_caption(f"<div style='text-align:center;'>{title}</div>")

class BadColumnsException(Exception):
    def __init__(self,*args,**kwargs):
        Exception.__init__(self,*args,**kwargs)

def _get_filter_expression(df,filter):
    filter_dict = {
                    '>=':None,
                    '<=':None,
                    '>':None,
                    '<':None,
                    '!=':None,
                    '=':None,
                    'contains':None
    }
    
    key = None               
    for k in filter_dict.keys():
        if k in filter:
            key = k 
            break
    if key is None:
        return df
    
    fparts = filter.strip().replace('{','').replace('}','').split(key)
    filter_column = fparts[0].replace(' ','')
    filter_value = fparts[1].replace(' ','')
    t = df[filter_column].dtype
    try:
        filter_value = pd.Series([str(filter_value)]).astype(t)[0]
    except:
        stop_callback(f"_get_filter_expression conversion error: {filter_value}")
        
    filter_dict = {
                    '>=':df[filter_column]>=filter_value,
                    '<=':df[filter_column]<=filter_value,
                    '>':df[filter_column]>filter_value,
                    '<':df[filter_column]<filter_value,
                    '!=':df[filter_column]!=filter_value,
                    '=':df[filter_column]==filter_value,
                    'contains':df[filter_column].astype(str).str.contains(str(filter_value)),
                    }
    filter_expression = filter_dict[key]
    df_filtered = df[filter_expression]
    return df_filtered



def _dash_table_update_paging_closure(df,input_store_key):
    def _dash_table_update_paging(input_list):
        if (input_list is None) or (len(input_list)<3):
            stop_callback(f"_dash_table_update_paging - insufficent data input_list {input_list}")
        page_current = input_list[0]
        page_size = input_list[1]
        filter_query = input_list[2]
        if len(input_list)>3 and input_list[3] is not None:
            # take first key from keys()
            store_data_dict = input_list[3]
            if input_store_key is not None:
                data_key = input_store_key
            else:
                data_key = list(store_data_dict.keys())[0]
            df_new = pd.DataFrame(store_data_dict[data_key])
        else:
            df_new = df.copy()
        
        # check to see if an "ERROR" data frame has been sent to this method
        if len(df_new.columns)==1 and df_new.columns.values[0].lower()=='error':
            # df_new is an error alert, so just send it without any other processing or checking
            print(f"_dash_table_update_paging RETURNING ERROR DataFrame")
            print(df_new)
            return [df_new.to_dict('records')]
        

        if (filter_query is not None) and (len(filter_query)>0):
            print(f"_dash_table_update_paging filter_query: {filter_query}")
            filters = filter_query.split("&&")            
            for f in filters:
                df_new = _get_filter_expression(df_new,f)
        beg_row = page_current*page_size
        if page_current*page_size > len(df_new):
            beg_row = len(df_new) - page_size
        print(f"_dash_table_update_paging_closure: page_current:{page_current} page_size: {page_size}")
        df_new =  df_new.iloc[
            beg_row:beg_row + page_size
        ]
        return [df_new.to_dict('records')]
    return _dash_table_update_paging


def make_dashtable(dtable_id,df_in,
                  input_store = None,
                  input_store_key=None,
                  columns_to_display=None,
                  editable_columns_in=None,
                  title='Dash Table',logger=None,
                  title_style=None,
                  filtering=False,
                  max_width='120vh',
                  displayed_rows=20,
                  editable=True):
    '''
    Create an instance of dash_table.DataTable
    
    :param dtable_id: The id for your DataTable
    :param df_in:     The pandas DataFrame that is the source of your DataTable (Default = None)
                        If None, then the DashTable will be created without any data, and await for its
                        data from a dash_html_components or dash_core_components instance.
    :param columns_to_display:    A list of column names which are in df_in.  (Default = None)
                                    If None, then the DashTable will display all columns in the DataFrame that
                                    it receives via df_in or via a callback.  However, the column
                                    order that is displayed can only be guaranteed using this parameter.
    :param editable_columns_in:    A list of column names that contain "modifiable" cells. ( Default = None)
    :param title:    The title of the DataFrame.  (Default = Dash Table)
    :param logger:
    :param title_style: The css style of the title. Default is dgrid_components.h4_like.
    :param filtering: True if you want each column to have filtering.  Default is False.
    
    :return dash_table_instance,DashLink_for_dynamic_paging

    '''
    # create logger 
    lg = init_root_logger() if logger is None else logger
    
    lg.debug(f'{dtable_id} entering create_dt_div')
    
    # create list that 
    editable_columns = [] if editable_columns_in is None else editable_columns_in
    datatable_id = dtable_id
    
    # create filter_action
    filter_action = 'none'
    if filtering:
        filter_action = 'fe'
    df_start = df_in.iloc[0:displayed_rows]
    dt = dash_table.DataTable(
        page_current= 0,
#         page_size= 100,
        page_size=displayed_rows,
        page_action='custom',        
        
        filter_action=filter_action,#'none', # 'fe',
#         fixed_rows={'headers': True, 'data': 0},
        style_data_conditional=[
            {
                'if': {'row_index': 'odd'},
                'backgroundColor': 'rgb(248, 248, 248)'
            }
        ],
        style_cell_conditional=[
            {
                'if': {'column_id': c},
                'textAlign': 'left',
            } for c in ['symbol', 'underlying']
        ],

        style_as_list_view=False,
        style_table={
#             'maxHeight':'450px','overflowX': 'scroll','overflowY':'scroll'
            'overflowX':'scroll','overflowY':'scroll',
             'maxWidth': max_width
        } ,
        
        style_data={
            'whiteSpace': 'normal',
            'height': 'auto'
        },        
        editable=editable,
#         css=[{"selector": "table", "rule": "width: 100%;"}],
        css=[{"selector": "table"}],   
        id=datatable_id
    )
    if columns_to_display is not None:
        if any([c not in df_start.columns.values for c in columns_to_display]):
            m = f'{columns_to_display} are missing from input data. Your input Csv'
            raise BadColumnsException(m)           
        df_start = df_start[columns_to_display]
            
    dt.data=df_start.to_dict('rows')
    dt.columns=[{"name": i, "id": i,'editable': True if i in editable_columns else False} for i in df_start.columns.values]                    
    lg.debug(f'{dtable_id} exiting create_dt_div')
    
    # create DashLink for dynamic paging
    input_tuples = [(dtable_id, "page_current"),(dtable_id, "page_size"),(dtable_id,"filter_query")]
    if input_store is not None:
        input_tuples = input_tuples + [(input_store,'data')]
    output_tuples = [(dtable_id, 'data')]
    link_for_dynamic_paging = DashLink(input_tuples,output_tuples,_dash_table_update_paging_closure(df_in[df_start.columns.values],input_store_key))
    return dt,link_for_dynamic_paging


# In[7]:


class_converters = {
    dcc.Checklist:lambda v:v,
    dcc.DatePickerRange:lambda v:v,
    dcc.DatePickerSingle:lambda v:v,
    dcc.Dropdown:lambda v:v,
    dcc.Input:lambda v:v,
    dcc.Markdown:lambda v:v,
    dcc.RadioItems:lambda v:v,
    dcc.RangeSlider:lambda v:v,
    dcc.Slider:lambda v:v,
    dcc.Store:lambda v:v,
    dcc.Textarea:lambda v:v,
    dcc.Upload:lambda v:v,
    dash_table.DataTable:lambda v:v,
}

html_members = [t[1] for t in inspect.getmembers(html)]
dcc_members = [t[1] for t in inspect.getmembers(dcc)]
all_members = html_members + dcc_members

class DashLink():
    def __init__(self,in_tuple_list, out_tuple_list,io_callback=None,
                 state_tuple_list= None,logger=None):
        self.logger = init_root_logger() if logger is None else logger
        _in_tl = [(k.id if type(k) in all_members else k,v) for k,v in in_tuple_list]
        _out_tl = [(k.id if type(k) in all_members else k,v) for k,v in out_tuple_list]
        self.output_table_names = _out_tl
        
        self.inputs = [Input(k,v) for k,v in _in_tl]
        self.outputs = [Output(k,v) for k,v in _out_tl]
        
        self.states = [] 
        if state_tuple_list is not None:
            _state_tl = [(k.id if type(k) in all_members else k,v) for k,v in state_tuple_list]
            self.states = [State(k,v) for k,v in _state_tl]
        
        self.io_callback = lambda input_list:input_list[0] 
        if io_callback is not None:
            self.io_callback = io_callback
                       
    def callback(self,theapp):
        @theapp.callback(
            self.outputs,
            self.inputs,
            self.states
            )
        def execute_callback(*inputs_and_states):
            l = list(inputs_and_states)
            if l is None or len(l)<1:
                stop_callback(f'execute_callback no data for {self.output_table_names}',self.logger)
            if all([l[i] is None for i in range(len(l))]):
                stop_callback(f'execute_callback no data for {self.output_table_names}',self.logger)
            ret = self.io_callback(l)
            return ret if type(ret) is list else [ret]
        return execute_callback
        


# ### ```make_panel``` creates a div which wraps components in css panels 
# (*see the ```stles.css``` in the ```assets``` folder*)
# * ```rpanel```:  css that should emulated a raised panel
# * ```rpanelnc```: css that should emulate a raised panel with not background color 
# 

# In[8]:


pn = 'rpanel' # see the oil_gas.css file for how this css class is defined
pnnc = 'rpanelnc'
pnnm = 'rpanelnm'
pnncnm = 'rpanelncnm'

# def panel_cell(child,div_id=None):
def panel_cell(child,div_id=None,panel_background_color=None):
    s = {} if panel_background_color is None else {'background-color':panel_background_color}
    return html.Div(child,className=pn,style=s,
                    id=_new_uuid() if div_id is None else div_id)

# def nopanel_cell(child,div_id=None):
def nopanel_cell(child,div_id=None,panel_background_color=None):
    s = {} if panel_background_color is None else {'background-color':panel_background_color}
    return html.Div(child,className=pnnc,style=s,
                    id=_new_uuid() if div_id is None else div_id)

def multi_cell_panel(children,grid_template=None,
                    parent_class=None,
                    orientation_is_rows=True,
                    panel_background_color=None,
#                     div_id=None):
                    div_id=None):
    gtr = ' '.join(['1fr' for _ in range(len(children))]) if grid_template is  None else grid_template
    orientation = 'grid-template-rows' if orientation_is_rows else 'grid-template-columns'
    opposite_orientation = 'grid-template-rows' if not orientation_is_rows else 'grid-template-columns'
    style = {'display':'grid',orientation:gtr,opposite_orientation:'1fr'}
    if panel_background_color is not None:
        style['background-color'] = panel_background_color
    panel_html = html.Div([c for c in children],
                          id=_new_uuid() if div_id is None else div_id,
                          className=parent_class,style=style)
    return panel_html


def multi_row_panel(children,grid_template=None,parent_class=None,
                    panel_background_color=None,
                    div_id=None):
    return multi_cell_panel(children,
                            grid_template=grid_template,
                            parent_class=parent_class,
                            orientation_is_rows=True,
                            panel_background_color=panel_background_color,
                            div_id=_new_uuid() if div_id is None else div_id)

def multi_column_panel(children,grid_template=None,parent_class=None,
                    panel_background_color=None,
                    div_id=None):
    return multi_cell_panel(children,grid_template=grid_template,
                            parent_class=parent_class,
                            orientation_is_rows=False,
                            panel_background_color=panel_background_color,
                            div_id=_new_uuid() if div_id is None else div_id)



# ### Define component builders from DataFrames

# In[9]:


def make_radio(df,comp_id,value_column,label_column=None,current_value=None,className=None):
    lc = value_column if label_column is None else label_column
    df_temp = df[[lc,value_column]].drop_duplicates().sort_values(label_column)
    df_temp.index = list(range(len(df_temp)))
    options = [{"label": wt[0], "value": wt[1]} for wt in df_temp.values]
    comp = dcc.RadioItems(
        id=comp_id,
        options = options,
        value=current_value,
        className=className
    )
    return comp

def make_dropdown(df,comp_id,value_column,label_column=None,current_value=None,multi=False,className=None):
    lc = value_column if label_column is None else label_column
    if lc == value_column:
        df_temp = df[[lc]].drop_duplicates().sort_values(lc)
        df_temp[f"{lc}_"] = df_temp[lc]
    else:
        df_temp = df[[lc,value_column]].drop_duplicates().sort_values(lc)
    df_temp.index = list(range(len(df_temp)))
    options = [{"label": wt[0], "value": wt[1]} for wt in df_temp.values]
    comp = dcc.Dropdown(
        id=comp_id,
        options = options,
        value=current_value,
        multi=multi,
        className=className
    )
    return comp

def make_slider(df,comp_id,value_column,className=None):
    min_val = df[value_column].min()
    max_val = df[value_column].max()
    slider = dcc.RangeSlider(
        id=comp_id,
        className=className,
        min=min_val,
        max=max_val,
        value=[min_val,max_val]
    )
    return slider

def make_datepicker(df,comp_id,timestamp_column,
                    init_date=0,className=None,style=None):
    min_date = df[timestamp_column].min()
    min_date - pd.to_datetime(min_date)
    max_date = df[timestamp_column].max()
#     max_date - pd.to_datetime(max_date)
    max_date = pd.to_datetime(max_date)
    idate = min_date if init_date == 0 else (max_date if init_date==1 else init_date)
    dp = dcc.DatePickerSingle(
        id=comp_id,
        min_date_allowed=min_date,
        max_date_allowed=max_date,
        date=idate,
        className=className,
        style=style
    )
    return dp

def make_page_title(title_text,div_id=None,html_container=None,parent_class=None,
                   panel_background_color='#CAE2EB'):
    par_class = parent_class
    if par_class is None:
        par_class = pnnm
    htmc = html_container
    if htmc is None:
        htmc = html.H2
        
    title_parts = title_text.split('\n')
    

    title_list = [htmc(tp,className=pnncnm) for tp in title_parts]
    r = multi_row_panel(title_list,
                 parent_class=par_class,
                 div_id=div_id,
                 panel_background_color=panel_background_color) 
    return r   

def make_text_centered_div(text):    
    col_inner_style = {
        'margin':'auto',
        'word-break':'break-all',
        'word-wrap': 'break-word'
    }
    return html.Div([text],style=col_inner_style)


# ### Define DashLink generators for common sets of components
def radio_to_dropdown_options_link(radio_comp,dropdown_comp,build_dropdown_options_callback):
    def _build_link_radio_dropdown(input_list):
        radio_value =  input_list[0]
        ret = build_dropdown_options_callback(radio_value)
        return [ret]
    link_radio1_dropdown_options = DashLink(
        [(radio_comp,'value')],
        [(dropdown_comp,'options')],
        _build_link_radio_dropdown)
    return link_radio1_dropdown_options

def radio_to_dropdown_value_link(radio_comp,dropdown_comp,build_dropdown_value_callback):
    def _build_link_radio_dropdown(input_list):
        radio_value =  input_list[0]
        ret = build_dropdown_value_callback(radio_value)
        return [ret]
    link_radio_dropdown_value = DashLink(
        [(radio_comp,'value')],
        [(dropdown_comp,'value')],
        _build_link_radio_dropdown)
    return link_radio_dropdown_value


# In[11]:


class DashApp():
    def __init__(self):
        self.all_dash_links = []
    
    def add_links(self,dashlink_list):
        for l in dashlink_list:
            self.add_link(l)
            
    def add_link(self,dashlink):
        link_already_in_list = False
        for otn in dashlink.output_table_names:
            for adl in self.all_dash_links:
                for adl_otn in adl.output_table_names:
                    if otn == adl_otn:
                        link_already_in_list = True
                        break
        if not link_already_in_list:
            self.all_dash_links.append(dashlink)
        else:
            print(f'add_link output {otn} already in output in all_dask_links')

    def register_callbacks(self,app):
        for dl in self.all_dash_links:
            dl.callback(app)
    
    def create_app(self,layout_html,run=True,url_base_pathname=None,app_host='127.0.0.1',app_port=8800,
                  app_title='dashapp',external_stylesheets=None,**kwargs):

        if url_base_pathname is not None:
            app = dash.Dash(__name__, url_base_pathname=url_base_pathname,external_stylesheets=external_stylesheets)
        else:
            app = dash.Dash(__name__,external_stylesheets=external_stylesheets)
        app.title = app_title
        app.layout = html.Div([layout_html])
        self.register_callbacks(app)
        full_url = f"http://{app_host}:{app_port}" + ('' if url_base_pathname is None else url_base_pathname)
        logger.info(f"This app will run at the URL: {full_url}")
        if run:
            app.run_server(host=app_host,port=app_port,**kwargs)
        return app


# In[23]:


# !jupyter nbconvert --to script dash_oil_gas.ipynb

