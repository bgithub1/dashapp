

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



# In[2]:


DEFAULT_LOG_PATH = './logfile.log'
DEFAULT_LOG_LEVEL = 'INFO'

def _new_uuid():
    return str(uuid.uuid1())

def init_root_logger(logfile=DEFAULT_LOG_PATH,logging_level=DEFAULT_LOG_LEVEL):
    level = logging_level
    if level is None:
        level = logging.DEBUG
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



# In[3]:


logger = init_root_logger(logging_level='DEBUG')

def stop_callback(errmess,logger=None):
    m = "****************************** " + errmess + " ***************************************"     
    if logger is not None:
        logger.debug(m)
    raise PreventUpdate()




def plotly_plot(df_in,x_column,plot_title=None,
                y_left_label=None,y_right_label=None,
                bar_plot=False,width=800,height=400,
                number_of_ticks_display=20,
                yaxis2_cols=None,
                x_value_labels=None,
               modebar_orientation='v',modebar_color='grey'):
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
            type='category'),
        yaxis=dict(
            title='y main' if y_left_label is None else y_left_label
        ),
        yaxis2=dict(
            title='y alt' if y_right_label is None else y_right_label,
            overlaying='y',
            side='right'),
        autosize=True,
#         autosize=False,
#         width=width,
#         height=height,
        margin=Margin(
            b=100
        ),
        modebar={'orientation': modebar_orientation,'bgcolor':modebar_color}
    )

    fig = go.Figure(data=data,layout=layout)
    fig.update_layout(
        title={
            'text': plot_title,
            'y':0.9,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'})
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

def _dash_table_update_paging_closure(df):
    def _dash_table_update_paging(input_list):
        if (input_list is None) or (len(input_list)<3):
            stop_callback(f"_dash_table_update_paging - insufficent data input_list {input_list}")
        page_current = input_list[0]
        page_size = input_list[1]
        filter_query = input_list[2]
        df_new = df.copy()
#        {Cnty} = 61 && {Well_Name} contains Sea
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
            'overflowY':'scroll','overflowY':'scroll',
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
    output_tuples = [(dtable_id, 'data')]
    link_for_dynamic_paging = DashLink(input_tuples,output_tuples,_dash_table_update_paging_closure(df_in[df_start.columns.values]))
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
            if l is None or len(l)<1 or l[0] is None:
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
#     panel_html = html.Div([html.Div(c,className=child_class) for c in children],
#                           id=_new_uuid() if div_id is None else div_id,
#                           className=parent_class,style=style)
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
    max_date - pd.to_datetime(max_date)
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
                  app_title='dashapp',external_stylesheets=None):

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
            app.run_server(host=app_host,port=app_port)
        return app


# In[23]:


# !jupyter nbconvert --to script dash_oil_gas.ipynb

