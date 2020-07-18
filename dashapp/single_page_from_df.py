'''
Created on Jul 15, 2020

@author: bperlman1
'''
import pandas as pd
import numpy as np

# import datetime
# from tqdm import tqdm,tqdm_notebook
from dashapp import dashapp2 as dashapp

html = dashapp.html
dcc = dashapp.dcc

_create_dropdown_callback = lambda df,comp_id,col,init_value,multi: dashapp.make_dropdown(
            df,comp_id,col,current_value=init_value,multi=multi)

_inputbox_style = {"font-size":"18px","text-align":"center","position":"relative",
    "display":"inline-block","width":"130px","height":"45px"}

_blue_button_style={
    'line-height': '40px',
#     'borderWidth': '1px',
#     'borderStyle': borderline,
#     'borderRadius': '1px',
    'textAlign': 'center',
    'background-color':'#A9D0F5',#ed4e4e
    'vertical-align':'middle',
}
    
def components_from_df(df,columns,d_prefix,init_values=None,multi=False):
    dict_component = {}
    for i in range(len(columns)):
        col = columns[i]
        init_value = None if init_values is None else init_values[i]
        if multi:
            init_value = [init_value]
        comp_id = f"{d_prefix}_{col}"
        comp = _create_dropdown_callback(
            df,comp_id,col,init_value,multi)
        dict_component[col] = comp
    return dict_component 

_query_operators_options = [
    {'label':'','value':''},
    {'label':'=','value':'=='},
    {'label':'!=','value':'!='},
    {'label':'>','value':'>'},
    {'label':'<','value':'<'},
    {'label':'>=','value':'>='},
    {'label':'<=','value':'<='},
    {'label':'btwn','value':'btwn'},
] 

class FilterDfComponent(html.Div):
    '''
    A class to that creates an html component that you can use to query 
    columns of a DataFrame
    Each FilterDfComponent is an html.Div, with 2 children:
      1.  A dcc.Dropdown with a list of comparison operators
      2.  A dcc.Input in which you can enter the predicate of your query
      
    Th
    '''
    def __init__(self,comp_id,col,text=None):
        super(FilterDfComponent,self).__init__(id=comp_id)
        self.col = col
        div_text = col if text is None else text
        self.col_div = dashapp.make_text_centered_div(div_text)
        dd_id = f"ddop_{comp_id}"
        self.operator_dd = dcc.Dropdown(id=dd_id,options=_query_operators_options,value='')
        input_id = f"inputop_{comp_id}"
        self.operand_input = dcc.Input(id=input_id,value='',debounce=True)
        
        mcp =  dashapp.multi_column_panel(
            [self.col_div,self.operator_dd,self.operand_input],
            grid_template='1fr 1fr 1fr'
        )
        self.children = mcp#.children
        
class FilterDescriptor():
    '''
    Simple class to contain a DataFrame column name, an intial filter value, and
      a plain text of the column name.
    Example usage:
         FilterDescriptor('commod','CL','Commodity: ')
         FilterDescriptor('contract_year',2020,'Contract Year: ')
    
    '''
    def __init__(self,column,init_value,text):
        '''
        Example usage:
             FilterDescriptor('commod','CL','Commodity: ')
             FilterDescriptor('contract_year',2020,'Contract Year: ')

        '''
        self.column=column
        self.init_value=init_value
        self.text=text

def progressive_dropdowns(value_list,dropdown_id,number_of_dropdowns,
                         label_list=None,
                         title_list=None):
    current_parent = None
    pd_dd_list = []
    pd_div_list = []
    pd_link_list = []
    current_value_list = value_list.copy()
    for i in range(number_of_dropdowns):
        curr_id = f"{dropdown_id}_v{i}"
        title = None if (title_list is None) or (len(title_list) < i+1) else title_list[i]
        pd_dd,pd_link = progressive_dropdown(current_value_list,curr_id,current_parent,
                                            title=title,label_list=label_list)
        current_parent = pd_dd
        pd_link_list.append(pd_link)        
        # wrap dropdown with title 
        dropdown_rows = [pd_dd]
        if (title_list is not None) and (len(title_list)>i):
            title_div = dashapp.make_text_centered_div(title_list[i])
            dropdown_rows = [title_div,pd_dd]
        dropdown_div = dashapp.multi_row_panel(dropdown_rows)
        # append new dropdown, wrapped in title, to list of dropdown htmls
        pd_dd_list.append(pd_dd)
        pd_div_list.append(dropdown_div)
        
    return pd_div_list,pd_link_list[1:],pd_dd_list

def progressive_dropdown(
    value_list,dropdown_id,parent_dropdown,label_list=None,title=None,
    current_value=None,multi=True,className=None):
    '''
    '''
#     options = [{"label": c, "value": c} for c in value_list]
    options = [
        {"label": value_list[i] if (label_list is None) or (len(label_list)<i+1) else label_list[i], 
         "value": value_list[i]} for i in range(len(value_list))
    ]
    this_dropdown = dcc.Dropdown(
        id=dropdown_id,
        options = options,
        value=current_value,
        multi=multi,
        className=className
    )
        
    # create DashLink
    from_parent_link = None
    if parent_dropdown is not None:
        # callback
        def _choose_options(input_data):
            parent_value = input_data[0]
            parent_options = input_data[1]
            if type(parent_value) != list:
                parent_value = [parent_value]            
            child_options = [po for po in parent_options if po['value'] not in parent_value]
            return [child_options]
        # make DashLink instance
        from_parent_link = dashapp.DashLink(
            [(parent_dropdown,'value'),(parent_dropdown,'options')],[(this_dropdown,'options')],_choose_options)
    return this_dropdown,from_parent_link
    
# graph definition dropdowns
# define x column
# define graph grouping columns: columns whose unique values create separate graphs 
# define y_main_columns
# define y_secondary columns
# X Column     Y Grouping Column   Y Left Axis Columns Y Right Axis Columns
# Display Graph button
class XyGraphDefinition(html.Div):
    def __init__(self,columns,data_store,div_id,num_graph_filter_rows=2,
                 logger=None):
        titles = ['X Column','Y Left Axis','Y Right Axis']
        prog_dd_divs,prog_dd_links,prog_dds = progressive_dropdowns(
            columns,f'{div_id}_dropdowns',len(titles),title_list=titles)
        
        # create input divs for inputing y left and right axis titles
        y_left_axis_input = dcc.Input(value="Y MAIN",id=f"{div_id}_y_left_axis_input")
        y_left_axis_input_title = dashapp.make_text_centered_div("Y Left Axis")
        y_left_axis_div = dashapp.multi_row_panel([y_left_axis_input_title,y_left_axis_input])

        y_right_axis_input = dcc.Input(value="Y ALT",id=f"{div_id}_y_right_axis_input")
        y_right_axis_input_title = dashapp.make_text_centered_div("Y Right Axis")
        y_right_axis_div = dashapp.multi_row_panel([y_right_axis_input_title,y_right_axis_input])

        # create the graph button
        graph_button = html.Button('Click for Graph',id=f'{div_id}_graph_button')
        graph_button_title = dashapp.make_text_centered_div('Refresh Graph')
        graph_button_div = dashapp.multi_row_panel([graph_button_title,graph_button])
        
        # create the graph title
        graph_title_input = dcc.Input(value="XY Graph",id=f"{div_id}_graph_title")
        graph_title_title = dashapp.make_text_centered_div("Graph Title")
        graph_title_div = dashapp.multi_row_panel([graph_title_title,graph_title_input])
        
        
        # create the Graph Component, with no graph in it as of yet
        graph_id = f"{div_id}_graph"
        graph_comp = dcc.Graph(id=graph_id)
        
        # build graph DashLink
        inputs = [(graph_button,'n_clicks')]
        outputs = [(graph_comp,'figure')]
        states = [(dd,'value') for dd in prog_dds] + [(data_store,'data')] + [(y_left_axis_input,'value'),(y_right_axis_input,'value'),(graph_title_input,'value')]
        
        def _build_fig_callback(input_data):
            x_col = input_data[1]
            if type(x_col) == list:
                x_col = x_col[0]
            y_left_cols = input_data[2]
            y_right_cols = input_data[3]
            if any([x_col is None,y_left_cols is None]):
                dashapp.stop_callback("No columns selected for graph",logger)
                
            if input_data[4] is None:
                dict_df = []
            else:
                dict_df = list(input_data[4].values())[0]
            df = pd.DataFrame(dict_df)
            y_left_label = input_data[5]
            y_right_label = input_data[6]
            graph_title = input_data[7]
            
            yrc = [] if (y_right_cols is None) or (y_right_cols[0] is None) else y_right_cols
            df = df[[x_col] + y_left_cols + yrc]
            fig = dashapp.plotly_plot(
                df_in=df,x_column=x_col,yaxis2_cols=y_right_cols,
                y_left_label=y_left_label,y_right_label=y_right_label,
                plot_title=graph_title)
            return [fig]
        dlink = dashapp.DashLink(inputs,outputs,_build_fig_callback,states)
        
        # arrange prog_dd_divs
        filter_rows = prog_dd_divs + [y_left_axis_div,y_right_axis_div] + [graph_button_div,graph_title_div]
        fr1 = dashapp.multi_column_panel(filter_rows[0:3])
        fr2 = dashapp.multi_column_panel(filter_rows[3:6])
        fr3 = dashapp.multi_column_panel(filter_rows[6:7])
        filter_div = dashapp.multi_row_panel([fr1,fr2,fr3])
        self.filter_rows = filter_rows
        self.filter_div = filter_div
        self.div_id = div_id
        self.graph = graph_comp
        self.dashlinks = prog_dd_links + [dlink]

def _make_fd(col,df=None):
    csplit = [c[0].upper() + c[1:] for c in col.split('_')]
    cupper = ' '.join(csplit) + ':'
    init_value = None if df is None else df.iloc[0][col]
    return FilterDescriptor(col,init_value,cupper)

def graph_from_csv_page(
    dt_id,
    page_title="DataFrame Viewer",
    app_title="DataFrame Viewer",
    app_host='127.0.0.1',
    app_port=8800,
    loading_full_screen=True,
    dataframe_title='Selected Data',
    filters_per_filter_row=2,
    run=True,
    logger= None,
    **kwargs
):
    if logger is None:
        logger = dashapp.init_root_logger()
    
    csvfup = dashapp.CsvFileUploader(f"{dt_id}_csvuploader")
    csvfup_title = dashapp.make_text_centered_div("Choose Csv File For Your Data")
    csvfup_div = dashapp.multi_row_panel([csvfup_title,csvfup])
    
    filter_div = html.Div(id=f"{dt_id}_filter_div")
    def _populate_filter_div(input_data):
        df = pd.DataFrame(input_data[0])
        def _get_data():
            return df
        d = create_dashgraph_page('dg_page', _get_data,run=False,logger=logger)
        return [d['app'].layout]

    
    filter_div_link = dashapp.DashLink(
        [(csvfup.data_store,'data')],[(filter_div,'children')],_populate_filter_div
    )
    dap = dashapp.DashApp()
    dap.add_links([filter_div_link,csvfup.uploader_link])
    layout_html = html.Div([csvfup_div,filter_div])
    dap.create_app(layout_html,app_host=app_host,app_port=app_port, app_title=app_title,run=run)

def create_dashgraph_page(
    dt_id,
    data_source,
    filter_discriptor_list=None,
    page_title="DataFrame Viewer",
    app_title="DataFrame Viewer",
    app_host='127.0.0.1',
    app_port=8800,
    multi=False,
    loading_full_screen=True,
    dataframe_title='Selected Data',
    filters_per_filter_row=2,
    run=True,
    logger= None,
    **kwargs
):
    '''
    Build all of rows that contain html and dcc components, that make up the
     single page of a Dashapp app.
    Also return all of the DashLink instances that contain callbacks that reactively
     update the page.
     
    @param df: The main DataFrame that supports the page
    @param dt_id: The DashTable id that will display the DataFrame data
    @param filter_discriptor_list: A list of instances of the class FilterDiscriptor
    @param get_data_callback: A method that get's called when a filter is changed and
        a new query is required.
    @param multi:
    @param loading_full_screen: show waiting icon on full screen, or just inside the component area
    @param dataframe_title: title of div with DataFrame of filtered data
    @param graph_title: title of div with graph
    @param filters_per_filter_row:
    @return dap,theapp,return_div,all_links
            dap is an instance of DashApp
            theapp is an instance of Dash
            return_div is the main div which can be used in another Dash app
            all_links are all of the DashLinks used by the DashApp.  DashLinks
              instances define call backs between components, without
              the need for all of the  repetitive Dash Boilerplate.
    '''
    if logger is None:
        logger = dashapp.init_root_logger()
    
    # Get initial data
    if callable(data_source):
        df_init = data_source()
    else:
        if data_source.data is None:
            df_init = pd.DataFrame()
        else:
            df_init = pd.DataFrame(data_source.data)
    # Get a lit of FilterDescriptor instances, if the caller provides them
    #   The caller can specify columns that he wants to filter on or
    #     he can provide filters for every column
    fdd = filter_discriptor_list
    if fdd is None:
        fdd = [_make_fd(c) for c in df_init.columns.values]
    
    # Get the columns that have filters
    filter_columns = [filter_comp.column for filter_comp in fdd]
    # Create instances of FilterDfComponent, which implement the logic
    #  to execute the filtering of the data that is read by the get_data_callback
    #  method that the caller must provide.
    filter_components = []
    for filter_comp in fdd:
        col = filter_comp.column
        t = filter_comp.text
        fid = f"{dt_id}_{col}_fc"
        fdf_comp = FilterDfComponent(fid,col,t)
        filter_components.append(fdf_comp)
    
    # Create a dcc.Store object to hold the contents of the get_data_callback data.
    data_store = dcc.Store(id=f'{dt_id}_data_store')
    data_store_loading = dcc.Loading(
        id='data_store_loading',children=[data_store],fullscreen=loading_full_screen)
    
    # Define the DashLink callback to populate data_store's dataframe using the filter components values
    #   that were filled in by the user
    id_df = f'df_{dt_id}'
    def _filter_data(input_data_and_data_source):
        if callable(data_source):
            df = data_source()
            input_data = input_data_and_data_source
        else:
            input_data = input_data_and_data_source[1:]
            dict_df = input_data_and_data_source[0]
            try:
                df = pd.DataFrame(dict_df)
            except Exception as e:
                dashapp.stop_callback(f"_filter_data: {str(e)} ",logger)
        # Get len of input_data array.  
        # Half of the elemennts of that array are from dcc.Inputs that you 
        #   specify in the input_tuple_list argument to the DashLink constructor.
        # The other half of that array are from dcc.Dropdowns that you
        #   specify in the state_tuple_list argument to the DashLink constructor.
        input_data_len = len(input_data)
        num_inputs = int(input_data_len/2)
        # Iterate through each filter, and successively create more
        #   narrow slices of the data to show the user.
        for i in range(num_inputs):
            # Get the dropdown, and make sure there is something in it.
            dropdown_value = input_data[i+num_inputs]
            if dropdown_value is None:
                continue
            if len(dropdown_value)<=0:
                # If there is nothing, then treat this filter as if it
                #   had a "contains" verb
                dropdown_value = 'contains'
            # Get the data to which the verb is applied when executing
            #   the pandas Dataframe.query method or the Series.contains method.
            input_value = input_data[i]            
            if (input_value is not None) and (len(input_value.strip())>0) and (dropdown_value is not None):
                # Everything is in order to attempt to execute DataFrame.query
                col = filter_columns[i]
                if dropdown_value == 'contains':
                    # Execute Series.contains
                    print(f"filter contains: {input_value}")
                    try:
                        #
                        df = df[df[col].astype(str).str.contains(input_value,regex=True)]
                        continue
                    except Exception as e:
                        print(f'returning ERROR dataframe from filter contains: {e}')
                        df = pd.DataFrame({'ERROR':[str(e)]})
                        return [{id_df:df.to_dict('rows')}]
                
                if dropdown_value=='btwn':
                    print(f"filter btwn: {input_value}")
                    try:
                        in_low_high = input_value.split(',')
                        in_low = in_low_high[0]
                        in_high = in_low if len(in_low_high)<2 else in_low_high[1]
                        df = df[(df[col]>=in_low) and df[col]<=in_high]
                        continue
                    except Exception as e:
                        print(f'returning ERROR dataframe from filter contains: {e}')
                        df = pd.DataFrame({'ERROR':[str(e)]})
                        return [{id_df:df.to_dict('rows')}]
                        
                # If you come here, you will execute Dataframe.query    
                # see if column is numeric
                needs_quotes = False
                try:
                    df[col].astype(float).sum()
                except:
                    needs_quotes = True
                    print('needs quotes')
                ipv = "''" if input_value is None else input_value
                ipv = f"'{ipv}'" if needs_quotes else ipv
                filter_query = f"{col} {dropdown_value} {ipv}"
                print(f"_filter_query: {filter_query}")
                try:
                    df = df.query(filter_query)
                except Exception as e:
                    print(f'returning ERROR from filter_query: {e}')
                    df = pd.DataFrame({'ERROR':[str(e)]})
                    return [{id_df:df.to_dict('rows')}]
        # Return the fully sliced new Dataframe 
        return [{id_df:df.to_dict('rows')}]
    
    filter_inputs = [(fc.operand_input,'value') for fc in filter_components]
    if hasattr(data_source,'data'):
        filter_inputs = [(data_source,'data')] + filter_inputs 
    filter_dropdowns = [(fc.operator_dd,'value') for fc in filter_components]
    # Create the DashLink object that links the filter inputs with the 
    #   dcc.Store
    data_store_link = dashapp.DashLink(
        filter_inputs,[(data_store,'data')],
        io_callback=_filter_data,
        state_tuple_list=filter_dropdowns
    )
    
    # Create a dash_table intance, using the data_store
    dt_values,dt_nav_link = dashapp.make_dashtable(
        dt_id,df_in=df_init,max_width=None,
        input_store = data_store,
        input_store_key=id_df,
    )
    
    
    dataframe_title_row = None
    if dataframe_title is not None:
        dataframe_title_row = dashapp.make_page_title(dataframe_title,div_id=f"{dt_id}_dataframe_title_row",html_container=html.H3)  

        
    
    # make the multi column panel for multiple filter rows
    filter_rows = []
    fc_lists = [filter_components[i:i+filters_per_filter_row] for i in np.arange(0,len(filter_components),filters_per_filter_row)]
    fr_index = 0
    for fc_list in fc_lists:
        filter_htmls = [
            dashapp.multi_row_panel(
                [fc_list[i]],
                grid_template=['1fr'],parent_class=None,            
            ) for i in range(len(fc_list))
        ]

        fr_grid_template = [' '.join(['1fr' for _ in range(len(filter_htmls))])]
        filter_row = dashapp.multi_column_panel(
            filter_htmls,parent_class=dashapp.pn,div_id=f'{dt_id}_filter_row_{fr_index}',
            grid_template=fr_grid_template
        )
        filter_rows.append(filter_row)
        fr_index += 1
    
    # row with dashtable
    dt_values_row = html.Div(dt_values,style={'width':'95vw'})
    #create non-graph rows
    non_graph_rows = filter_rows + [dataframe_title_row,dt_values_row,data_store_loading]
    if type(data_source) == dcc.Store:
        non_graph_rows = non_graph_rows + [data_source]
    non_graph_div = dashapp.multi_row_panel([html.Div(non_graph_rows)])
    
    # create graph rows
    graph_title_row = dashapp.make_page_title("XY Graph",div_id=f"{dt_id}_graph_title_row",html_container=html.H3)  
    
    graph_def = XyGraphDefinition(df_init.columns.values,data_store,'main_graph',logger)
    graph_div = html.Div(["Select X, Y Left and Y Right Axis Columns",graph_def.filter_div,graph_title_row,graph_def.graph])
    
    # create return div, and DashLinks list
    return_div = dashapp.multi_row_panel([non_graph_div,graph_div])
    all_links = [dt_nav_link,data_store_link] + graph_def.dashlinks 
    dap = dashapp.DashApp()
    
    # *********** Assemble all of he rows and columns below ***************
    r1 = dashapp.make_page_title(page_title,div_id='r1',html_container=html.H3)                  
    all_rows = html.Div([r1,return_div])
    # Add all of the DashLinks to the DashApp instance (dap)
    dap.add_links(all_links)
    # Create the dash app object by calling the create_app method of dap (the instance of DashApp)
    theapp = dap.create_app(
        all_rows,app_host=app_host,app_port=app_port,run=run,
                   app_title=app_title,**kwargs)

    return {'all_rows':all_rows,'all_links':all_links,
            'app':theapp}


       