
#PROGETTO BISF 

import dash
from dash.dependencies import Input, Output
import dash_core_components as dcc
import dash_html_components as html
import pandas_datareader.data as web
from datetime import datetime as dt
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.svm import SVR


external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div([
    
    html.Div([html.Strong("Web Application BISF")], style={'font-family':'sans-serif', 'font-size':40, 'text-align': 'center'}),
    html.Hr(style={'border':'3px solid black'}),
    html.Div([html.Strong("Descriptive Analysis")], style={'font-family':'sans-serif', 'font-size':25, 'text-align': 'center'}),
    html.Br(),
    html.Strong("Select the company to analyze, the return and the granularity:", style={'font-family':'sans-serif','font-size':18, 'margin-left':'70px'}),
    html.Br(),
    html.Br(),
    dcc.Dropdown(
        id='stock_type',
        style={'font-family':'sans-serif', 'width':'50%', 'margin-left':'40px'},
        options=[
            {'label': 'Airbus', 'value': 'AIR.PA'},
            {'label': 'Carnival Corporation', 'value': 'CCL'},
            {'label': 'Amazon', 'value': 'AMZN'},
            {'label': 'Walmart', 'value': 'WMT'},
            {'label': 'Canadian Solar Inc.', 'value': 'CSIQ'},
            {'label': 'Iberdrola', 'value': 'IBDRY'}
            ], value='AIR.PA'),
    html.Br(),
    
    dcc.Dropdown(
        id='return_type',
        style={'font-family':'sans-serif', 'width':'50%', 'margin-left':'40px'},
        options=[
            {'label': 'Simple return', 'value': 'Adj Close'},
            {'label': 'Log return', 'value': 'Compound'}
            ], value='Adj Close'),
    html.Br(),
    
    dcc.Dropdown(
        id='granularity_type',
        style={'font-family':'sans-serif', 'width':'50%', 'margin-left':'40px'},
        options=[
            {'label': 'Day', 'value': 'D'},
            {'label': 'Week', 'value': 'W'},
            {'label': 'Month', 'value': 'M'},
            {'label': 'Year', 'value': 'Y'}
            ], value='W'),
    
    html.Div([
        html.Strong('Add stock on graph:', style={'font-family':'sans-serif','font-size':18}),
        html.Br(),
        html.Br(),
        dcc.Checklist( 
            id='add-stock',
            options=[
                    {'label': 'Airbus', 'value': 'AIR.PA'},
                    {'label': 'Carnival Corporation', 'value': 'CCL'},
                    {'label': 'Amazon', 'value': 'AMZN'},
                    {'label': 'Walmart', 'value': 'WMT'},
                    {'label': 'Canadian Solar Inc.', 'value': 'CSIQ'},
                    {'label': 'Iberdrola', 'value': 'IBDRY'}
                    ], value=[]),
        
        ], style={'font-family':'sans-serif', 'position':'relative', 'left':900, 'top':-209}),
    

    dcc.Graph(id='first_graph', style={'position':'relative', 'top':-100}),
    html.Hr(),          
    html.Div("Univariate Statistics", style={'font-family':'sans-serif', 'font-size':25, 'text-align': 'center'}),
    html.Br(),

    html.Div([
        dcc.Graph(id='qqplot_graph', style={'display': 'inline-block'}),
        dcc.Graph(id='boxplot_graph', style={'display': 'inline-block'})
        ], style={'textAlign':'center'}),
    html.Br(),
    
    html.Div([
        dcc.Graph(id='hist_graph', style={'display': 'inline-block'}),
        dcc.Graph(id='normal_graph', style={'display': 'inline-block'})
        ], style={'textAlign':'center'}),
    html.Br(),
    
    
    html.Div([
        
        html.Div([
            html.Div("Univariate statistical analysis of", style={'font-family':'sans-serif', 'font-size':20, 'display': 'inline-block'}),
            html.Div(id='name', style={'font-family':'sans-serif', 'font-size':20, 'display': 'inline-block'}),
            html.Br(),
            html.Br(),
            html.Div(id='stat-desc', style={'font-family':'sans-serif', 'font-size':15}),
            html.Br(),
            html.Br(),
            ], style={'textAlign':'left' ,'display': 'inline-block', 'margin-left':70}),
    
        html.Div([
            html.Div("Data Summary", style={'font-family':'sans-serif', 'font-size':20, 'display': 'inline-block'}),
            html.Br(),
            html.Br(),
            html.Div("Stocks with the highest indices:", style={'font-family':'sans-serif'}),
            html.Div(id='max-mean', style={'font-family':'sans-serif'}),
            html.Div(id='max-dev', style={'font-family':'sans-serif'}),
            html.Div(id='max-skew', style={'font-family':'sans-serif'}),
            html.Div(id='max-kurt', style={'font-family':'sans-serif'}),
            
            html.Br(),
            html.Div("Stocks with the lowest indices:", style={'font-family':'sans-serif'}),
            html.Div(id='min-mean', style={'font-family':'sans-serif'}),
            html.Div(id='min-dev', style={'font-family':'sans-serif'}),
            html.Div(id='min-skew', style={'font-family':'sans-serif'}),
            html.Div(id='min-kurt', style={'font-family':'sans-serif'}),   
            html.Br(),
            html.Div('Please note: lowest skew and kurt are meant stocks with closer distance (in abs) to zero', style={'font-family':'sans-serif', 'font-size':10})
            ], style={'textAlign':'left','display': 'inline-block', 'margin-left':300 }),
        
        html.Div(className="clearer"),
    ], style={'textAlign':'center'}, className="split2"),
    
    html.Br(),
    html.Br(),       
    
    html.Hr(),
    html.Div(["Multivariate Statistics"], style={'font-family':'sans-serif', 'font-size':25, 'text-align': 'center'}),
    dcc.Graph(id='tab-cov'),
    html.Div([
    dcc.Graph(id='tab-cor', style={'position':'relative', 'top':-150})], style={'height': 300} ),
    
    
    html.Div([
    html.P("Select two companies and visualize the scatter-plot of correlations:", style={'font-family':'sans-serif','font-size':18, 'margin-left':'70px', 'position':'relative', 'top':-50 }),
    
    dcc.Dropdown(
        id='stock-one',
        style={'font-family':'sans-serif', 'width':'50%', 'margin-left':'40px'},
        options=[
            {'label': 'Airbus', 'value': 'AIR.PA'},
            {'label': 'Carnival Corporation', 'value': 'CCL'},
            {'label': 'Amazon', 'value': 'AMZN'},
            {'label': 'Walmart', 'value': 'WMT'},
            {'label': 'Canadian Solar Inc.', 'value': 'CSIQ'},
            {'label': 'Iberdrola', 'value': 'IBDRY'}
            ], value='AIR.PA'),
    html.Br(),
    
    dcc.Dropdown(
        id='stock-two',
        style={'font-family':'sans-serif', 'width':'50%', 'margin-left':'40px'},
        options=[
            {'label': 'Airbus', 'value': 'AIR.PA'},
            {'label': 'Carnival Corporation', 'value': 'CCL'},
            {'label': 'Amazon', 'value': 'AMZN'},
            {'label': 'Walmart', 'value': 'WMT'},
            {'label': 'Canadian Solar Inc.', 'value': 'CSIQ'},
            {'label': 'Iberdrola', 'value': 'IBDRY'}
            ], value='CCL'),
    html.Br(),
    
    dcc.Dropdown(
        id='granularity_type_2',
        style={'font-family':'sans-serif', 'width':'50%', 'margin-left':'40px'},
        options=[
            {'label': 'Day', 'value': 'D'},
            {'label': 'Week', 'value': 'W'},
            {'label': 'Month', 'value': 'M'},
            {'label': 'Year', 'value': 'Y'}
            ], value='W'),
    
    
    dcc.Graph(id='graph-cor', style={'height':700})
    ]),
    
    html.Br(),
    
    html.Hr(),
    html.Div("Beta Index", style={'font-family':'sans-serif', 'font-size':25, 'text-align': 'center'}),
    html.Br(),
    html.P("Select the company and the time range to calculate the Beta index:", style={'font-family':'sans-serif','font-size':18, 'margin-left':'70px'}),
    html.Br(),
    
    html.Div([
        dcc.Dropdown(
            id='beta-stock',
            style={'font-family':'sans-serif', 'width':'50%', 'margin-left':'40px'},
            options=[
                    {'label': 'Airbus', 'value': 'AIR.PA'},
                    {'label': 'Carnival Corporation', 'value': 'CCL'},
                    {'label': 'Amazon', 'value': 'AMZN'},
                    {'label': 'Walmart', 'value': 'WMT'},
                    {'label': 'Canadian Solar Inc.', 'value': 'CSIQ'},
                    {'label': 'Iberdrola', 'value': 'IBDRY'}
                    ], value='AIR.PA')]),
    html.Br(),
    html.Div([
        dcc.DatePickerRange(id='date-picker-range',	
                            style={'font-family':'sans-serif', 'width':'50%'},
                            start_date=dt(2010,1,1),					
                            initial_visible_month=dt(2010,1,1),				
                            end_date=dt.today()) ], style={'margin-left':'80px'}),
    
    html.Div(id='beta', style={'font-size':30, 'position':'relative', 'top':-80, 'right':-800}),
    html.Br(),
    
    html.P("Select the delta (monthly granularity) to plot the Beta:", style={'font-family':'sans-serif','font-size':18, 'margin-left':'70px'}),
    html.Br(),
    
    html.Div([
    dcc.Slider(id='delta-beta',
                min=3,
                max=20,
                step=None,
                marks = {i: str(i) for i in [3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]},
                value=5)], style={'width':'80%', 'margin-left':'70px'} ),
    
    html.Br(),
    dcc.Graph(id='beta-graph'),
    html.Br(),

    
    html.Hr(),
    html.Div([html.Strong("Predictive Analysis")], style={'font-family':'sans-serif', 'font-size':25, 'text-align': 'center'}),
    html.Br(),
    html.P("Forecast analysis of the period: {} till {}".format((dt.today()-pd.DateOffset(months=10)).strftime("%d %b %Y"), dt.today().strftime("%d %b %Y")) , style={'font-family':'sans-serif','font-size':18, 'margin-left':'70px'}),
    
    dcc.RadioItems(id='fake-input', style={'opacity':0}, options=[{'label': '', 'value': 'go'}], value='go'),
    html.Br(),
    
    dcc.Loading(
        id="loading-1",
        type="default",
        children = dcc.Graph(id='graph-forecast')
                ),
    html.Br(),
    
    html.Hr(),
    html.Div([html.Strong("Real simple ret vs Forecast simple ret")], style={'font-family':'sans-serif', 'font-size':25, 'text-align': 'center'}),
    dcc.Loading(
        id="loading-5",
        type="default",
        children = dcc.Graph(id='fig4')
                ),
    html.Br(),
    
    html.Hr(),
    html.Div([html.Strong("Portfolio Management")], style={'font-family':'sans-serif', 'font-size':25, 'text-align': 'center'}),
    html.Br(),
    
    dcc.Loading(
        id="loading-2",
        type="default",
        children = dcc.Graph(id='markowitz-past')
                ),
    html.P("Best sharpe-ratio portfolio:", style={'font-family':'sans-serif','font-size':18, 'margin-left':'70px'}),
    html.Div(id='past-ret', style={'font-family':'sans-serif', 'font-size':15, 'margin-left':'80px'}),
    html.Div(id='past-vol', style={'font-family':'sans-serif', 'font-size':15, 'margin-left':'80px'}),
    html.Br(),
    
    dcc.Loading(
        id="loading-3",
        type="default",
        children = dcc.Graph(id='markowitz-future')
                ),
    html.P("Best sharpe-ratio portfolio:", style={'font-family':'sans-serif','font-size':18, 'margin-left':'70px'}),
    html.Div(id='future-ret', style={'font-family':'sans-serif', 'font-size':15, 'margin-left':'80px'}),
    html.Div(id='future-vol', style={'font-family':'sans-serif', 'font-size':15, 'margin-left':'80px'}),
    html.Br(),
    html.P("Weights:", style={'font-family':'sans-serif','font-size':18, 'margin-left':'70px'}),
    html.Div(id='air-p', style={'font-family':'sans-serif', 'font-size':15, 'margin-left':'80px'}),
    html.Div(id='ccl-p', style={'font-family':'sans-serif', 'font-size':15, 'margin-left':'80px'}),
    html.Div(id='amzn-p', style={'font-family':'sans-serif', 'font-size':15, 'margin-left':'80px'}),
    html.Div(id='wmt-p', style={'font-family':'sans-serif', 'font-size':15, 'margin-left':'80px'}),
    html.Div(id='csiq-p', style={'font-family':'sans-serif', 'font-size':15, 'margin-left':'80px'}),
    html.Div(id='ibdry-p', style={'font-family':'sans-serif', 'font-size':15, 'margin-left':'80px'}),
    html.Br(),
    
    html.Hr(),
    html.Div("Investment Calculator", style={'font-family':'sans-serif', 'font-size':25, 'text-align': 'center'}),
    html.Br(),
    html.P("In this section you can observe the return that you would have obtained in lasts 10 months (till today) investing 10.000€ in portfolio calculated above (Markowitz model with forecast) and compare the Effective return with Expected return.",
           style={'font-family':'sans-serif','font-size':18, 'margin-left':'70px', 'margin-right':'70px'}),
    html.P("Please note: has been applied a transaction cost of 1%, moreover you have to consider the Capital Gain Tax of 26% on the net return",
           style={'font-family':'sans-serif','font-size':18, 'margin-left':'70px', 'margin-right':'70px'}),
    html.Br(),
    html.Div(id='effective-return', style={'font-family':'sans-serif', 'font-size':15, 'margin-left':'80px'}),
    html.Div(id='future-ret2', style={'font-family':'sans-serif', 'font-size':15, 'margin-left':'80px'}),
    html.Br(),
    dcc.Loading(
        id="loading-4",
        type="default",
        children = dcc.Graph(id='portfolio')),
                
    html.Div(id='gross-return', style={'font-family':'sans-serif', 'font-size':15, 'margin-left':'80px'}),
    html.Div(id='net-return', style={'font-family':'sans-serif', 'font-size':15, 'margin-left':'80px'}),
    html.Div(id='net-return-taxed', style={'font-family':'sans-serif', 'font-size':15, 'margin-left':'80px'}),
    html.Br(),
    html.Hr(),
                      ], className="container-fluid")




#OUTPUT ACTIONS
@app.callback(Output('first_graph', 'figure'), 				
              [Input('stock_type', 'value'),
               Input('return_type', 'value'),
               Input('granularity_type', 'value'),
               Input('add-stock', 'value')])

def update_graph(selected_dropdown_value,				
                 selected_dropdown_return,				
                 selected_dropdown_granularity,
                 add_stock_name):
    
    def stock_check(ticker, granularity):
        df = web.DataReader(ticker, data_source='yahoo', start=dt(2010, 1, 1), end=dt.now())   
        df = df['Adj Close']
        df_ret = np.log(df/df.shift(1)) 
        df_ret = df_ret.dropna()
        df = df.to_frame()
        df_ret = df_ret.to_frame()
        df_ret.columns = ["Compound"]
        df = df.join(df_ret)
        
        if(granularity=="M"):
            df = df.groupby(pd.Grouper(freq='M')).mean() 
        if(granularity=="W"):
            df = df.groupby(pd.Grouper(freq='W')).mean()
        if(granularity=='Y'):
            df = df.groupby(pd.Grouper(freq='Y')).mean()
        
        return df
    
    if(len(add_stock_name)!=0):
        if 'AIR.PA' in add_stock_name:
            air = stock_check('AIR.PA', selected_dropdown_granularity)
        if 'CCL' in add_stock_name:
            ccl = stock_check('CCL', selected_dropdown_granularity)
        if 'AMZN' in add_stock_name:
            amzn = stock_check('AMZN', selected_dropdown_granularity)
        if 'WMT' in add_stock_name:
            wmt = stock_check('WMT', selected_dropdown_granularity)
        if 'CSIQ' in add_stock_name:
            csiq = stock_check('CSIQ', selected_dropdown_granularity)
        if 'IBDRY' in add_stock_name:
            ibdry = stock_check('IBDRY', selected_dropdown_granularity)
					
            
    df = web.DataReader(selected_dropdown_value, data_source='yahoo', start=dt(2010, 1, 1), end=dt.now())   
    df = df['Adj Close']
    df_ret = np.log(df/df.shift(1)) 
    df_ret = df_ret.dropna()
    df = df.to_frame()
    df_ret = df_ret.to_frame()
    df_ret.columns = ["Compound"]
    df = df.join(df_ret)
    

    if(selected_dropdown_granularity=="M"):
        df = df.groupby(pd.Grouper(freq='M')).mean() 
    if(selected_dropdown_granularity=="W"):
        df = df.groupby(pd.Grouper(freq='W')).mean()
    if(selected_dropdown_granularity=='Y'):
        df = df.groupby(pd.Grouper(freq='Y')).mean()
    
    name = ""
    if(selected_dropdown_return=="Adj Close"):
        name = name + "Simple Return"
    else:
        name = name + "Continuous Compound Return"
    
    
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(x=df[selected_dropdown_return].index, y=df[selected_dropdown_return], name=selected_dropdown_value, mode='lines'
        ))
    
    if('AIR.PA' in add_stock_name):   
        fig.add_trace(go.Scatter(x = air[selected_dropdown_return].index, y = air[selected_dropdown_return], name='AIR.PA', mode='lines'))
    if('CCL' in add_stock_name):  
        fig.add_trace(go.Scatter(x = ccl[selected_dropdown_return].index, y = ccl[selected_dropdown_return], name='CCL', mode='lines'))
    if('AMZN' in add_stock_name): 
        fig.add_trace(go.Scatter(x = amzn[selected_dropdown_return].index, y = amzn[selected_dropdown_return], name='AMZN', mode='lines'))
    if('WMT' in add_stock_name): 
        fig.add_trace(go.Scatter(x = wmt[selected_dropdown_return].index, y = wmt[selected_dropdown_return], name='WMT', mode='lines'))
    if('CSIQ' in add_stock_name):     
        fig.add_trace(go.Scatter(x = csiq[selected_dropdown_return].index, y = csiq[selected_dropdown_return], name='CSIQ', mode='lines'))
    if('IBDRY' in add_stock_name):      
        fig.add_trace(go.Scatter(x = ibdry[selected_dropdown_return].index, y = ibdry[selected_dropdown_return], name='IBDRY', mode='lines'))
    
    fig.update_layout(title='Stock graph', height=700)
    
    return fig


@app.callback(Output('hist_graph', 'figure'), 				
              [Input('stock_type', 'value'),
               Input('return_type', 'value'),
               Input('granularity_type', 'value')])

def update_graph_hist(selected_dropdown_value,				
                 selected_dropdown_return,				
                 selected_dropdown_granularity):
    
    df = web.DataReader(
        selected_dropdown_value, data_source='yahoo',
        start=dt(2010, 1, 1), end=dt.now())
    
    if(selected_dropdown_granularity=="M"):
        df = df.groupby(pd.Grouper(freq='M')).mean() 
    if(selected_dropdown_granularity=="W"):
        df = df.groupby(pd.Grouper(freq='W')).mean()
    if(selected_dropdown_granularity=='Y'):
        df = df.groupby(pd.Grouper(freq='Y')).mean()
    
    df = df['Adj Close']
    df_ret = np.log(df/df.shift(1)) 
    df_ret = df_ret.dropna()
    df_ret = df_ret.to_frame()

    fig = px.histogram(df_ret, x='Adj Close', title="Histogram", width=700)
    
    return fig


@app.callback(Output('boxplot_graph', 'figure'), 				
              [Input('stock_type', 'value'),
               Input('return_type', 'value'),
               Input('granularity_type', 'value')])

def update_graph_boxplot(selected_dropdown_value,				
                 selected_dropdown_return,				
                 selected_dropdown_granularity):
    
    df = web.DataReader(
        selected_dropdown_value, data_source='yahoo',
        start=dt(2010, 1, 1), end=dt.now())
    
    if(selected_dropdown_granularity=="M"):
        df = df.groupby(pd.Grouper(freq='M')).mean() 
    if(selected_dropdown_granularity=="W"):
        df = df.groupby(pd.Grouper(freq='W')).mean()
    if(selected_dropdown_granularity=='Y'):
        df = df.groupby(pd.Grouper(freq='Y')).mean()
    
    df = df['Adj Close']
    df_ret = np.log(df/df.shift(1)) 
    df_ret = df_ret.dropna()
    df_ret = df_ret.to_frame()

    fig = px.box(df_ret, y='Adj Close', title="Boxplot")
    
    return fig


@app.callback(Output('qqplot_graph', 'figure'), 				
              [Input('stock_type', 'value'),
               Input('return_type', 'value'),
               Input('granularity_type', 'value')])

def update_graph_qqplot(selected_dropdown_value,				
                 selected_dropdown_return,				
                 selected_dropdown_granularity):
    
    df = web.DataReader(
        selected_dropdown_value, data_source='yahoo',
        start=dt(2010, 1, 1), end=dt.now())
    
    if(selected_dropdown_granularity=="M"):
        df = df.groupby(pd.Grouper(freq='M')).mean() 
    if(selected_dropdown_granularity=="W"):
        df = df.groupby(pd.Grouper(freq='W')).mean()
    if(selected_dropdown_granularity=='Y'):
        df = df.groupby(pd.Grouper(freq='Y')).mean()
    
    df = df['Adj Close']
    df_ret = np.log(df/df.shift(1)) 
    df_ret = df_ret.dropna()
    df_ret = df_ret.to_frame()

    qq = stats.probplot(df_ret['Adj Close'], dist='norm', sparams=(1))
    x = np.array([qq[0][0][0], qq[0][0][-1]])

    fig = go.Figure()
    fig.add_scatter(x=qq[0][0], y=qq[0][1], mode='markers')
    fig.add_scatter(x=x, y=qq[1][1] + qq[1][0]*x, mode='lines')
    fig.layout.update(title="QQPlot",showlegend=False)
    
    return fig


@app.callback(Output('stat-desc', 'children'),
              [Input('stock_type', 'value'),
               Input('granularity_type', 'value')])

def update_univariate_stat(selected_dropdown_value,				
                           selected_dropdown_granularity):
    
    df = web.DataReader(
        selected_dropdown_value, data_source='yahoo',
        start=dt(2010, 1, 1), end=dt.now())
    
    if(selected_dropdown_granularity=="M"):
        df = df.groupby(pd.Grouper(freq='M')).mean() 
    if(selected_dropdown_granularity=="W"):
        df = df.groupby(pd.Grouper(freq='W')).mean()
    if(selected_dropdown_granularity=='Y'):
        df = df.groupby(pd.Grouper(freq='Y')).mean()
    
    df = df['Adj Close']
    df_ret = np.log(df/df.shift(1)) 
    df_ret = df_ret.dropna()
    df_ret = df_ret.to_frame()
    
    out = ['Mean: {}'.format(np.round(df_ret['Adj Close'].mean(), 6)),
           html.Br(),
           'Variance: {}'.format(np.round(df_ret['Adj Close'].var(), 6)),
           html.Br(),
           'Std Dev: {}'.format(np.round(df_ret['Adj Close'].std(), 6)),
           html.Br(),
           'Skewness: {}'.format(np.round(df_ret['Adj Close'].skew(), 6)),
           html.Br(),
           'Kurtosis: {}'.format(np.round(df_ret['Adj Close'].kurtosis(), 6)),
           html.Br(),
           html.Br(),
           'Min: {}'.format(np.round(df_ret['Adj Close'].describe()[3], 6)),
           html.Br(),
           '25%: {}'.format(np.round(df_ret['Adj Close'].describe()[4], 6)),
           html.Br(),
           'Median: {}'.format(np.round(df_ret['Adj Close'].describe()[5], 6)),
           html.Br(),
           '75%: {}'.format(np.round(df_ret['Adj Close'].describe()[6], 6)),
           html.Br(),
           'Max: {}'.format(np.round(df_ret['Adj Close'].describe()[7], 6))
           
           ]
    
    return out


@app.callback(Output('name', 'children'),
              [Input('stock_type','value')])

def update_name(selected_dropdown_value):
    name = ""
    if(selected_dropdown_value=="AIR.PA"):
        name = name + "Airbus"
    if(selected_dropdown_value=="CCL"):
        name = name + "Carnival Corporation"
    if(selected_dropdown_value=="AMZN"):
        name = name + "Amazon"
    if(selected_dropdown_value=="WMT"):
        name = name + "Walmart"
    if(selected_dropdown_value=="CSIQ"):
        name = name + "Canadian Solar"
    if(selected_dropdown_value=="IBDRY"):
        name = name + "Iberdrola"
    return ': {}'.format(name)


@app.callback(Output('normal_graph', 'figure'), 				
              [Input('stock_type', 'value'),
               Input('granularity_type', 'value')])

def update_graph_normal(selected_dropdown_value, selected_dropdown_granularity):
    
    df = web.DataReader(
        selected_dropdown_value, data_source='yahoo',
        start=dt(2010, 1, 1), end=dt.now())
    
    if(selected_dropdown_granularity=="M"):
        df = df.groupby(pd.Grouper(freq='M')).mean() 
    if(selected_dropdown_granularity=="W"):
        df = df.groupby(pd.Grouper(freq='W')).mean()
    if(selected_dropdown_granularity=='Y'):
        df = df.groupby(pd.Grouper(freq='Y')).mean()
    
    df = df['Adj Close']
    df_ret = np.log(df/df.shift(1)) 
    df_ret = df_ret.dropna()
    df_ret = df_ret.to_frame()
    
    x = np.array(df_ret['Adj Close'])
    hist_data = [x]
    
    group_labels = ['Density'] # name of the dataset
    fig = ff.create_distplot(hist_data, group_labels, show_hist=False, show_rug=False)
    fig.update_layout(title='Density', showlegend=False)
    
    
    return fig


@app.callback(Output('max-mean', 'children'),
              Output('max-dev', 'children'),
              Output('min-mean', 'children'),
              Output('min-dev', 'children'),
              Output('min-skew', 'children'),
              Output('min-kurt', 'children'),
              Output('max-skew', 'children'),
              Output('max-kurt', 'children'),
              [Input('granularity_type', 'value')])

def data_summary(selected_dropdown_granularity):
    
    stock_list = ['AIR.PA','CCL','AMZN','WMT','CSIQ','IBDRY']
    stock_list_name = ['Airbus','Carnival Corporation','Amazon','Walmart','Canadian Solar','Iberdrola']
    
    skew_list_abs = []
    kurt_list_abs = []
    mean_list = []
    dev_list = []
    
    for name in stock_list:
    
        df = web.DataReader(
            name, data_source='yahoo',
            start=dt(2010, 1, 1), end=dt.now())
    
        if(selected_dropdown_granularity=="M"):
            df = df.groupby(pd.Grouper(freq='M')).mean() 
        if(selected_dropdown_granularity=="W"):
            df = df.groupby(pd.Grouper(freq='W')).mean()
        if(selected_dropdown_granularity=='Y'):
            df = df.groupby(pd.Grouper(freq='Y')).mean()
    
        df = df['Adj Close']
        df_ret = np.log(df/df.shift(1)) 
        df_ret = df_ret.dropna()
        df_ret = df_ret.to_frame()
        
        skew_list_abs.append(np.round(abs(0-df_ret['Adj Close'].skew()), 4))
        kurt_list_abs.append(np.round(abs(0-df_ret['Adj Close'].kurtosis()), 4))
        mean_list.append(np.round(df_ret['Adj Close'].mean(), 6))
        dev_list.append(np.round(df_ret['Adj Close'].std(), 6))
    
    return 'Mean: {}'.format(stock_list_name[mean_list.index(max(mean_list))]), 'Std Dev: {}'.format(stock_list_name[dev_list.index(max(dev_list))]), 'Mean: {}'.format(stock_list_name[mean_list.index(min(mean_list))]), 'Std Dev: {}'.format(stock_list_name[dev_list.index(min(dev_list))]), 'Skewness: {}'.format(stock_list_name[skew_list_abs.index(min(skew_list_abs))]), 'Kurtosis: {}'.format(stock_list_name[kurt_list_abs.index(min(kurt_list_abs))]), 'Skewness: {}'.format(stock_list_name[skew_list_abs.index(max(skew_list_abs))]), 'Kurtosis: {}'.format(stock_list_name[kurt_list_abs.index(max(kurt_list_abs))])


@app.callback(Output('tab-cor', 'figure'),
              [Input('granularity_type', 'value')])

def correlation(selected_dropdown_granularity):
    
    AIR = web.DataReader('AIR.PA', data_source='yahoo', start=dt(2010, 1, 1), end=dt.now())
    CCL = web.DataReader('CCL', data_source='yahoo', start=dt(2010, 1, 1), end=dt.now())
    AMZN = web.DataReader('AMZN', data_source='yahoo', start=dt(2010, 1, 1), end=dt.now())
    WMT = web.DataReader('WMT', data_source='yahoo', start=dt(2010, 1, 1), end=dt.now())
    CSIQ = web.DataReader('CSIQ', data_source='yahoo', start=dt(2010, 1, 1), end=dt.now())
    IBDRY = web.DataReader('IBDRY', data_source='yahoo', start=dt(2010, 1, 1), end=dt.now())
    
    if(selected_dropdown_granularity=="M"):
        AIR = AIR.groupby(pd.Grouper(freq='M')).mean() 
        CCL = CCL.groupby(pd.Grouper(freq='M')).mean()
        AMZN = AMZN.groupby(pd.Grouper(freq='M')).mean()
        WMT = WMT.groupby(pd.Grouper(freq='M')).mean()
        CSIQ = CSIQ.groupby(pd.Grouper(freq='M')).mean()
        IBDRY = IBDRY.groupby(pd.Grouper(freq='M')).mean()
    if(selected_dropdown_granularity=="W"):
        AIR = AIR.groupby(pd.Grouper(freq='W')).mean() 
        CCL = CCL.groupby(pd.Grouper(freq='W')).mean()
        AMZN = AMZN.groupby(pd.Grouper(freq='W')).mean()
        WMT = WMT.groupby(pd.Grouper(freq='W')).mean()
        CSIQ = CSIQ.groupby(pd.Grouper(freq='W')).mean()
        IBDRY = IBDRY.groupby(pd.Grouper(freq='W')).mean()
    if(selected_dropdown_granularity=='Y'):
        AIR = AIR.groupby(pd.Grouper(freq='Y')).mean() 
        CCL = CCL.groupby(pd.Grouper(freq='Y')).mean()
        AMZN = AMZN.groupby(pd.Grouper(freq='Y')).mean()
        WMT = WMT.groupby(pd.Grouper(freq='Y')).mean()
        CSIQ = CSIQ.groupby(pd.Grouper(freq='Y')).mean()
        IBDRY = IBDRY.groupby(pd.Grouper(freq='Y')).mean()
    
    AIR = AIR['Adj Close']
    CCL = CCL['Adj Close']
    AMZN = AMZN['Adj Close']
    WMT = WMT['Adj Close']
    CSIQ = CSIQ['Adj Close']
    IBDRY = IBDRY['Adj Close']
    
    AIR = np.log(AIR/AIR.shift(1))
    CCL = np.log(CCL/CCL.shift(1))  
    AMZN = np.log(AMZN/AMZN.shift(1))
    WMT = np.log(WMT/WMT.shift(1))
    CSIQ = np.log(CSIQ/CSIQ.shift(1))
    IBDRY = np.log(IBDRY/IBDRY.shift(1))
    
    AIR = AIR.dropna()
    CCL = CCL.dropna()
    AMZN = AMZN.dropna()
    WMT = WMT.dropna()
    CSIQ = CSIQ.dropna()
    IBDRY = IBDRY.dropna()
    
    data = pd.concat([AIR, CCL, AMZN, WMT, CSIQ, IBDRY], axis=1)
    data.columns = ['AIR','CCL','AMZN','WMT','CSIQ','IBDRY']
    data = data.dropna()
    
    matrix_corr = data.corr()
    matrix = np.matrix(matrix_corr)
    matrix = np.round(matrix,3)
    
    names = np.array(['ρij','AIR','CCL','AMZN','WMT','CSIQ','IBDRY'])
    names2 = np.array(['AIR','CCL','AMZN','WMT','CSIQ','IBDRY'])
    
    np.fill_diagonal(matrix, 0)  
    top = matrix.max()
    
    np.fill_diagonal(matrix, 1)
    bot = matrix.min()
    
    vals0 = matrix[:,0]
    vals1 = matrix[:,1]
    vals2 = matrix[:,2]
    vals3 = matrix[:,3]
    vals4 = matrix[:,4]
    vals5 = matrix[:,5]
    
    
    fig = go.Figure(data = go.Table(header=dict(values=names, fill_color='rgba(182, 200, 235, 0.8)'), 
                                    cells=dict(values=(names2,matrix[:,0],matrix[:,1],matrix[:,2],matrix[:,3],matrix[:,4],matrix[:,5]),
                                               fill_color=['rgba(182, 200, 235, 0.8)', 
                                                          ['rgba(0, 250, 0, 0.8)' if val == top else 'red' if val == bot else 'rgb(245, 245, 245)' for val in vals0],
                                                          ['rgba(0, 250, 0, 0.8)' if val == top else 'red' if val == bot else 'rgb(245, 245, 245)' for val in vals1],
                                                          ['rgba(0, 250, 0, 0.8)' if val == top else 'red' if val == bot else 'rgb(245, 245, 245)' for val in vals2],
                                                          ['rgba(0, 250, 0, 0.8)' if val == top else 'red' if val == bot else 'rgb(245, 245, 245)' for val in vals3],
                                                          ['rgba(0, 250, 0, 0.8)' if val == top else 'red' if val == bot else 'rgb(245, 245, 245)' for val in vals4],
                                                          ['rgba(0, 250, 0, 0.8)' if val == top else 'red' if val == bot else 'rgb(245, 245, 245)' for val in vals5]
                                                          ]
                                               
                                               )))
    
    fig.layout.update(title="Matrix of Correlations",showlegend=False)
        
    
    return fig


@app.callback(Output('tab-cov', 'figure'),
              [Input('granularity_type', 'value')])

def covariance(selected_dropdown_granularity):
    
    AIR = web.DataReader('AIR.PA', data_source='yahoo', start=dt(2010, 1, 1), end=dt.now())
    CCL = web.DataReader('CCL', data_source='yahoo', start=dt(2010, 1, 1), end=dt.now())
    AMZN = web.DataReader('AMZN', data_source='yahoo', start=dt(2010, 1, 1), end=dt.now())
    WMT = web.DataReader('WMT', data_source='yahoo', start=dt(2010, 1, 1), end=dt.now())
    CSIQ = web.DataReader('CSIQ', data_source='yahoo', start=dt(2010, 1, 1), end=dt.now())
    IBDRY = web.DataReader('IBDRY', data_source='yahoo', start=dt(2010, 1, 1), end=dt.now())
    
    if(selected_dropdown_granularity=="M"):
        AIR = AIR.groupby(pd.Grouper(freq='M')).mean() 
        CCL = CCL.groupby(pd.Grouper(freq='M')).mean()
        AMZN = AMZN.groupby(pd.Grouper(freq='M')).mean()
        WMT = WMT.groupby(pd.Grouper(freq='M')).mean()
        CSIQ = CSIQ.groupby(pd.Grouper(freq='M')).mean()
        IBDRY = IBDRY.groupby(pd.Grouper(freq='M')).mean()
    if(selected_dropdown_granularity=="W"):
        AIR = AIR.groupby(pd.Grouper(freq='W')).mean() 
        CCL = CCL.groupby(pd.Grouper(freq='W')).mean()
        AMZN = AMZN.groupby(pd.Grouper(freq='W')).mean()
        WMT = WMT.groupby(pd.Grouper(freq='W')).mean()
        CSIQ = CSIQ.groupby(pd.Grouper(freq='W')).mean()
        IBDRY = IBDRY.groupby(pd.Grouper(freq='W')).mean()
    if(selected_dropdown_granularity=='Y'):
        AIR = AIR.groupby(pd.Grouper(freq='Y')).mean() 
        CCL = CCL.groupby(pd.Grouper(freq='Y')).mean()
        AMZN = AMZN.groupby(pd.Grouper(freq='Y')).mean()
        WMT = WMT.groupby(pd.Grouper(freq='Y')).mean()
        CSIQ = CSIQ.groupby(pd.Grouper(freq='Y')).mean()
        IBDRY = IBDRY.groupby(pd.Grouper(freq='Y')).mean()
    
    AIR = AIR['Adj Close']
    CCL = CCL['Adj Close']
    AMZN = AMZN['Adj Close']
    WMT = WMT['Adj Close']
    CSIQ = CSIQ['Adj Close']
    IBDRY = IBDRY['Adj Close']
    
    AIR = np.log(AIR/AIR.shift(1))
    CCL = np.log(CCL/CCL.shift(1))  
    AMZN = np.log(AMZN/AMZN.shift(1))
    WMT = np.log(WMT/WMT.shift(1))
    CSIQ = np.log(CSIQ/CSIQ.shift(1))
    IBDRY = np.log(IBDRY/IBDRY.shift(1))
    
    AIR = AIR.dropna()
    CCL = CCL.dropna()
    AMZN = AMZN.dropna()
    WMT = WMT.dropna()
    CSIQ = CSIQ.dropna()
    IBDRY = IBDRY.dropna()
    
    data = pd.concat([AIR, CCL, AMZN, WMT, CSIQ, IBDRY], axis=1)
    data.columns = ['AIR','CCL','AMZN','WMT','CSIQ','IBDRY']
    data = data.dropna()
    
    matrix_cov = data.cov()
    matrix = np.matrix(matrix_cov)
    matrix = np.round(matrix,8)
    
    names = np.array(['σij','AIR','CCL','AMZN','WMT','CSIQ','IBDRY'])
    names2 = np.array(['AIR','CCL','AMZN','WMT','CSIQ','IBDRY'])
    
    np.fill_diagonal(matrix, 0)
    top = matrix.max()
    
    matrix_cov = data.cov()
    matrix = np.matrix(matrix_cov)
    matrix = np.round(matrix,8)
    bot = matrix.min()
    
    vals0 = matrix[:,0]
    vals1 = matrix[:,1]
    vals2 = matrix[:,2]
    vals3 = matrix[:,3]
    vals4 = matrix[:,4]
    vals5 = matrix[:,5]
       
    fig = go.Figure(data = go.Table(header=dict(values=names, fill_color='rgba(182, 200, 235, 0.8)'), 
                                    cells=dict(values=(names2,matrix[:,0],matrix[:,1],matrix[:,2],matrix[:,3],matrix[:,4],matrix[:,5]),
                                               fill_color=['rgba(182, 200, 235, 0.8)', 
                                                          ['rgba(0, 250, 0, 0.8)' if val == top else 'red' if val == bot else 'rgb(245, 245, 245)' for val in vals0],
                                                          ['rgba(0, 250, 0, 0.8)' if val == top else 'red' if val == bot else 'rgb(245, 245, 245)' for val in vals1],
                                                          ['rgba(0, 250, 0, 0.8)' if val == top else 'red' if val == bot else 'rgb(245, 245, 245)' for val in vals2],
                                                          ['rgba(0, 250, 0, 0.8)' if val == top else 'red' if val == bot else 'rgb(245, 245, 245)' for val in vals3],
                                                          ['rgba(0, 250, 0, 0.8)' if val == top else 'red' if val == bot else 'rgb(245, 245, 245)' for val in vals4],
                                                          ['rgba(0, 250, 0, 0.8)' if val == top else 'red' if val == bot else 'rgb(245, 245, 245)' for val in vals5]
                                                          ]
                                               )))
    
    fig.layout.update(title="Matrix of Covariances",showlegend=False)
        
    
    return fig


@app.callback(Output('graph-cor', 'figure'),
              [Input('stock-one', 'value'),
               Input('stock-two','value'),
               Input('granularity_type_2','value')])

def update_correlation_graph(stock_one, stock_two, selected_dropdown_granularity):
    
    S1 = web.DataReader(stock_one, data_source='yahoo', start=dt(2010, 1, 1), end=dt.now())
    S2 = web.DataReader(stock_two, data_source='yahoo', start=dt(2010, 1, 1), end=dt.now())
    
    if(selected_dropdown_granularity=="M"):
        S1 = S1.groupby(pd.Grouper(freq='M')).mean() 
        S2 = S2.groupby(pd.Grouper(freq='M')).mean()
    if(selected_dropdown_granularity=="W"):
        S1 = S1.groupby(pd.Grouper(freq='W')).mean() 
        S2 = S2.groupby(pd.Grouper(freq='W')).mean()
    if(selected_dropdown_granularity=='Y'):
        S1 = S1.groupby(pd.Grouper(freq='Y')).mean() 
        S2 = S2.groupby(pd.Grouper(freq='Y')).mean()
    
    S1_ret = np.log(S1/S1.shift(1)) 
    S2_ret = np.log(S2/S2.shift(1)) 
    
    data = pd.concat([S1_ret['Adj Close'], S2_ret['Adj Close']], axis=1)
    data = data.dropna()
    data.columns = [stock_one,stock_two]
    
    fig = ff.create_scatterplotmatrix(data, diag='histogram', title="Matrix of Correlations", height=750, width=1400,)
    

    return fig



@app.callback(Output('graph-forecast', 'figure'),
              Output('markowitz-future', 'figure'),
              Output('portfolio', 'figure'),
              Output('fig4', 'figure'),
              Output('future-ret', 'children'),
              Output('future-ret2', 'children'),
              Output('future-vol', 'children'),
              Output('air-p', 'children'),
              Output('ccl-p', 'children'),
              Output('amzn-p', 'children'),
              Output('wmt-p', 'children'),
              Output('csiq-p', 'children'),
              Output('ibdry-p', 'children'),
              Output('effective-return', 'children'),
              Output('gross-return', 'children'),
              Output('net-return', 'children'),
              Output('net-return-taxed', 'children'),
              [Input('fake-input', 'value')])

def update_forecast_graph(selected_dropdown_value):
    
    
    def stock_preparator(ticker):                                           #data, data_target, dataset_P, dataset_I, dt_simple_I
        start = dt(2010,1,1)
        end = dt.today()
    
        dataset = web.get_data_yahoo(ticker,start,end)                                           
        dataset = dataset.groupby(pd.Grouper(freq='M')).mean()
        
        dt_simple_I = dataset['Adj Close'].tail(11)
        dt_simple_I = dt_simple_I.to_frame()
                                         
        dataset = np.log(dataset['Adj Close']/dataset['Adj Close'].shift(1)).to_frame()         
        dataset = dataset.dropna()

        dataset_I = dataset.tail(10)
        dataset_P = dataset.head(len(dataset.index)-10)
    
        dataset = dataset.head(len(dataset.index)-10)
        dataset.columns = ["TARGET"] 
    
        N = 3

        for i in range(N):                                                                      
            dataset['Lag.' + str(i+1)] = dataset['TARGET'].shift(i+1)
    
        names_columns = list(dataset.columns[1:len(dataset.columns)])                           
        dataset = dataset.reindex(np.append(names_columns, ['TARGET']), axis=1)                  

        dataset = dataset.dropna()                              
        data = dataset.iloc[:,0:3]	         
        data_target = dataset.iloc[:,3] 
    
        return data, data_target, dataset_P, dataset_I, dt_simple_I
  
    def analysis(svm, data, data_target, opt_values):                       #opt_values
    
        mse_svm = cross_val_score(svm, data, data_target, cv=10, 
                                      scoring='neg_mean_squared_error')
        mse_svm = abs(np.average(mse_svm))
        #print("Iteration", counter, "> MSE", mse_svm)
        if(opt_values[0] > mse_svm):
            opt_values[0] = mse_svm
            opt_values[1] = Cs[a]
            opt_values[2] = gammas[b]
            opt_values[3] = epsilons[c]
        return opt_values
    
    def predictor(opt_values, data, data_target, dataset_P, dataset_I):     #dataset_P, dataset_I
    
        end = dt.today()
    
        opt_model_svm = opt_values
        #BEST MODEL
        svm_best_model = SVR(kernel='rbf', C=opt_model_svm[1], gamma=opt_model_svm[2], epsilon=opt_model_svm[3])

        #SPLITTO I DATI in Training e Test per allenare il modello
        X_train, X_test, y_train, y_test = train_test_split(data, data_target, 
                                                            test_size=0.30,
                                                            random_state=42)

        #ADDESTRO IL MODELLO
        svm_trained_model = svm_best_model.fit(X_train, y_train)
    
        x_forecast = data.tail(10)      #Variabile per la previsione

        prediction = svm_trained_model.predict(x_forecast)  

        #Gabola
        end2 = end.replace(month=(end.month + 1))
        
        #AGGIUNGO LA PREVISIONE
        future_date = pd.date_range(start=dataset_I.index[0], end=end2, freq='M')
                      
        past_date = dataset_P.index                                         
        total_date = past_date.append(future_date)                      

        for i in range(len(future_date)):
            dataset_P = dataset_P.append({"Adj Close":prediction[i]}, ignore_index = True)      
     
        dataset_P.index = total_date          

        return dataset_P, dataset_I

    AIR = stock_preparator('AIR.PA')
    CCL = stock_preparator('CCL')
    AMZN = stock_preparator('AMZN')
    WMT = stock_preparator('WMT')
    CSIQ = stock_preparator('CSIQ')
    IBDRY = stock_preparator('IBDRY')
    
    air_simple = AIR[4]
    ccl_simple = CCL[4]
    amzn_simple = AMZN[4]
    wmt_simple = WMT[4]
    csiq_simple = CSIQ[4]
    ibdry_simple = IBDRY[4]
    
    
    #GRID SEARCH E CROSS-VALIDATION 
    Cs = np.array([0.001,0.01,0.1,1,10,100,1000])
    gammas = np.array([0.001,0.01,0.1,1,10,100,1000])  
    epsilons = np.array([0.0001, 0.001, 0.01, 0.1])

    opt_values_air = [np.inf, np.inf, np.inf, np.inf]
    opt_values_ccl = [np.inf, np.inf, np.inf, np.inf]
    opt_values_amzn = [np.inf, np.inf, np.inf, np.inf]
    opt_values_wmt = [np.inf, np.inf, np.inf, np.inf]
    opt_values_csiq = [np.inf, np.inf, np.inf, np.inf]
    opt_values_ibdry = [np.inf, np.inf, np.inf, np.inf]
    
    counter = 1
    for a in range(0,len(Cs)):
        for b in range(0,len(gammas)):
            for c in range(0, len(epsilons)):
            
                svm = SVR(kernel='rbf', C=Cs[a], gamma=gammas[b], epsilon=epsilons[c])
            
                opt_air = analysis(svm, AIR[0], AIR[1], opt_values_air)
                opt_values_air = opt_air
                
                opt_ccl = analysis(svm, CCL[0], CCL[1], opt_values_ccl)
                opt_values_ccl = opt_ccl
            
                opt_amzn = analysis(svm, AMZN[0], AMZN[1], opt_values_amzn)
                opt_values_amzn = opt_amzn
            
                opt_wmt = analysis(svm, WMT[0], WMT[1], opt_values_wmt)
                opt_values_wmt = opt_wmt
            
                opt_csiq = analysis(svm, CSIQ[0], CSIQ[1], opt_values_csiq)
                opt_values_csiq = opt_csiq
            
                opt_ibdry = analysis(svm, IBDRY[0], IBDRY[1], opt_values_ibdry)
                opt_values_ibdry = opt_ibdry
            
                counter += 1

    AIR_P = predictor(opt_values_air, AIR[0], AIR[1], AIR[2], AIR[3])
    CCL_P = predictor(opt_values_ccl, CCL[0], CCL[1], CCL[2], CCL[3])
    AMZN_P = predictor(opt_values_amzn, AMZN[0], AMZN[1], AMZN[2], AMZN[3])
    WMT_P = predictor(opt_values_wmt, WMT[0], WMT[1], WMT[2], WMT[3])
    CSIQ_P = predictor(opt_values_csiq, CSIQ[0], CSIQ[1], CSIQ[2], CSIQ[3])
    IBDRY_P = predictor(opt_values_ibdry, IBDRY[0], IBDRY[1], IBDRY[2], IBDRY[3])
    
    AIR = AIR_P[0]
    CCL = CCL_P[0]
    AMZN = AMZN_P[0]
    WMT = WMT_P[0]
    CSIQ = CSIQ_P[0]
    IBDRY = IBDRY_P[0]  
    
    fig = make_subplots(subplot_titles=("AIR", "CCL", "AMZN", "WMT", "CSIQ", "IBDRY"), rows=3, cols=2)

    fig.add_trace(
        go.Scatter(x=AIR.index, y=AIR['Adj Close'], mode='lines', name='AIR'), 
        row=1, col=1
        )

    fig.add_trace(
        go.Scatter(x=CCL.index, y=CCL['Adj Close'], mode='lines', name='CCL'),
        row=1, col=2
        )
    
    fig.add_trace(
        go.Scatter(x=AMZN.index, y=AMZN['Adj Close'], mode='lines', name='AMZN'),
        row=2, col=1
        )

    fig.add_trace(
        go.Scatter(x=WMT.index, y=WMT['Adj Close'], mode='lines', name='WMT'),
        row=2, col=2
        )

    fig.add_trace(
        go.Scatter(x=CSIQ.index, y=CSIQ['Adj Close'], mode='lines', name='CSIQ'),
        row=3, col=1
        )

    fig.add_trace(
        go.Scatter(x=IBDRY.index, y=IBDRY['Adj Close'], mode='lines', name='IBDRY'),
        row=3, col=2
        )

    fig.update_layout(height=1500, width=1300, title_text="Forecast with Support-Vector Machine (SVM)")
    
    
    ##PARTE DI CONFRONTO-----------------------------------------------------------------------
    
    def simple_bck_1(ticker, compound):
   
        df = web.DataReader(ticker, data_source='yahoo', start=dt(2010, 1, 1), end=dt.now())   
        df = df['Adj Close']
        df = df.groupby(pd.Grouper(freq='M')).mean()
        df = df.to_frame()
        
        df_ret = compound
        df_ret = df_ret.dropna()
        df_ret.columns = ["Compound"]
        
        df = df.join(df_ret)
   
        return df

    def simple_bck_2(stock):             #N.B. STOCK DEVE ESSERE UN DF CON DUE COLONNE: ADJ CLOSE E COMPOUND, L'output è il df.tail(10)
        array = np.array([])
        date = stock.index[len(stock.index)-11:len(stock.index)]
        
        primo = np.exp(stock.iloc[len(stock.index)-11,1]) + stock.iloc[len(stock.index)-12,0] 
        array = np.append(array, primo)                                                 #Aprile
        
        count = 0
        for i in range(len(stock.index)-10,len(stock.index)):
            
            var = np.exp(stock.iloc[i,1]) + array[count] 
            array = np.append(array, var)
            
            count += 1

        df = pd.DataFrame(data = array, index = date)
        df.columns = ['Simple ret']
    
        return df
    
    
    stock1 = simple_bck_1('AIR.PA',AIR)
    stock2 = simple_bck_1('CCL',CCL)
    stock3 = simple_bck_1('AMZN',AMZN)
    stock4 = simple_bck_1('WMT',WMT)
    stock5 = simple_bck_1('CSIQ',CSIQ)
    stock6 = simple_bck_1('IBDRY',IBDRY)
    
    stock1_fin = simple_bck_2(stock1)
    stock2_fin = simple_bck_2(stock2)
    stock3_fin = simple_bck_2(stock3)
    stock4_fin = simple_bck_2(stock4)
    stock5_fin = simple_bck_2(stock5)
    stock6_fin = simple_bck_2(stock6)
    
    fig4 = make_subplots(subplot_titles=("AIR", "CCL", "AMZN", "WMT", "CSIQ", "IBDRY"), rows=3, cols=2)    
    
    fig4.add_trace(go.Scatter(x=stock1.index, y=stock1['Adj Close'], mode='lines', name='AIR'),            row=1, col=1)       
    fig4.add_trace(go.Scatter(x=stock1_fin.index, y=stock1_fin['Simple ret'], mode='lines', name='AIR'),   row=1, col=1)
    
    fig4.add_trace(go.Scatter(x=stock2.index, y=stock2['Adj Close'], mode='lines', name='CCL'),            row=1, col=2)       
    fig4.add_trace(go.Scatter(x=stock2_fin.index, y=stock2_fin['Simple ret'], mode='lines', name='CCL'),   row=1, col=2) 
    
    fig4.add_trace(go.Scatter(x=stock3.index, y=stock3['Adj Close'], mode='lines', name='AMZN'),            row=2, col=1)       
    fig4.add_trace(go.Scatter(x=stock3_fin.index, y=stock3_fin['Simple ret'], mode='lines', name='AMZN'),   row=2, col=1) 
    
    fig4.add_trace(go.Scatter(x=stock4.index, y=stock4['Adj Close'], mode='lines', name='WMT'),            row=2, col=2)       
    fig4.add_trace(go.Scatter(x=stock4_fin.index, y=stock4_fin['Simple ret'], mode='lines', name='WMT'),   row=2, col=2) 
    
    fig4.add_trace(go.Scatter(x=stock5.index, y=stock5['Adj Close'], mode='lines', name='CSIQ'),            row=3, col=1)       
    fig4.add_trace(go.Scatter(x=stock5_fin.index, y=stock5_fin['Simple ret'], mode='lines', name='CSIQ'),   row=3, col=1) 
    
    fig4.add_trace(go.Scatter(x=stock6.index, y=stock6['Adj Close'], mode='lines', name='IBDRY'),            row=3, col=2)       
    fig4.add_trace(go.Scatter(x=stock6_fin.index, y=stock6_fin['Simple ret'], mode='lines', name='IBDRY'),   row=3, col=2) 
        

    fig4.update_layout(height=1500, width=1300, title_text="Real simple return vs Forecast simple return", showlegend=False)
    
    ##FINE PARTE DI CONFRONTO------------------------------------------------------------------
    
        
    ##PARTE DI MARKOWITZ##---------------------------------------------------------------------
    data2 = pd.concat([AIR, CCL, AMZN, WMT, CSIQ, IBDRY], axis=1)
    data2.columns = ['AIR','CCL','AMZN','WMT','CSIQ','IBDRY']
    data2 = data2.dropna()
    
    log_ret = data2   
    
    np.random.seed(42)
    num_ports = 6000
    all_weights = np.zeros((num_ports, len(data2.columns)))
    ret_arr = np.zeros(num_ports)
    vol_arr = np.zeros(num_ports)
    sharpe_arr = np.zeros(num_ports)

    for x in range(num_ports):
        # Weights
        weights = np.array(np.random.random(6))
        weights = weights/np.sum(weights)
    
        # Save weights
        all_weights[x,:] = weights
    
        # Expected return
        ret_arr[x] = np.sum( (log_ret.mean() * weights * 12))
    
        # Expected volatility
        vol_arr[x] = np.sqrt(np.dot(weights.T, np.dot(log_ret.cov()*12, weights)))
    
        # Sharpe Ratio
        sharpe_arr[x] = (ret_arr[x] - 0.001 )/vol_arr[x]


    max_sr_ret = ret_arr[sharpe_arr.argmax()]
    max_sr_vol = vol_arr[sharpe_arr.argmax()]
    
    fig2 = go.Figure()

    fig2.add_trace(go.Scatter(
        x=vol_arr, y=ret_arr,
        name='portfolios',
        mode='markers',
        marker = dict(
            color=sharpe_arr, 
            colorscale='Viridis')
        ))

    fig2.add_trace(go.Scatter(
        x=[max_sr_vol], y=[max_sr_ret],
        name='best sharpe-ratio port.',
        marker_color='red'
        ))
    
    till = dt.today()
    
    # Set options common to all traces with fig.update_traces
    fig2.update_traces(mode='markers')
    fig2.update_layout(title='Markowitz model with forecast (of lasts 10 months) calculated from Jan 2010 till {}'.format(till.strftime("%d %b %Y")), height=700)
    
    pos_best_port = sharpe_arr.argmax()
    pesi = all_weights[pos_best_port,:]
    
    p_air = pesi[0]
    p_ccl = pesi[1]
    p_amzn = pesi[2]
    p_wmt = pesi[3]
    p_csiq = pesi[4]
    p_ibdry = pesi[5]
    ##FINE PARTE MARKOWITZ-----------------------------------------------------------------------------
    
    ##CALCOLO SIMPLE RETURN PORTAFOGLIO----------------------------------------------------------------
    def simple_ret(stock):
        stock_array = np.array(stock['Adj Close'])
        simple_ret = (stock_array[len(stock_array)-1]/stock_array[0])-1
        return simple_ret
    
    air_simp_ret = simple_ret(air_simple)
    ccl_simp_ret = simple_ret(ccl_simple)
    amzn_simp_ret = simple_ret(amzn_simple)
    wmt_simp_ret = simple_ret(wmt_simple)
    csiq_simp_ret = simple_ret(csiq_simple)
    ibdry_simp_ret = simple_ret(ibdry_simple)
    
    port_simple_ret = (air_simp_ret*p_air)+(ccl_simp_ret*p_ccl)+(amzn_simp_ret*p_amzn)+(wmt_simp_ret*p_wmt)+(csiq_simp_ret*p_csiq)+(ibdry_simp_ret*p_ibdry)

    ##FINE CALCOLO SIMPLE RETURN PORTAFOGLIO-------------------------------------------------------------
    
    #CALCOLO ANDAMENTO PORTAFOGLIO-----------------------------------------------------------------------
    def simple_ret_1M(stock,i):
        stock_array = np.array(stock['Adj Close'])
        simple_ret_1M = (stock_array[i+1]/stock_array[0])-1
        return simple_ret_1M
    
    portfolio = np.array([9900])
    
    for i in range(len(air_simple)-1):
        val = portfolio[0]*(((simple_ret_1M(air_simple, i))*pesi[0] + (simple_ret_1M(ccl_simple, i))*pesi[1] + (simple_ret_1M(amzn_simple, i))*pesi[2] + (simple_ret_1M(wmt_simple, i))*pesi[3] + (simple_ret_1M(csiq_simple, i))*pesi[4] + (simple_ret_1M(ibdry_simple, i))*pesi[5])+1)                                               
        portfolio = np.append(portfolio, val)
    
    inizio = dt.today() - pd.DateOffset(months=10)
    fine = dt.today() + pd.DateOffset(months=1)
    mydates_month = pd.date_range(start=inizio, end=fine, freq='M')

    ts1 = pd.Series(portfolio, index=mydates_month)
    ts1 = ts1.to_frame()
    ts1.columns = ['Value']
    
    fig3 = go.Figure()
    fig3.add_trace(
        go.Scatter(x=ts1.index, y=ts1['Value'], mode='lines', name='Portfolio')),
    fig3.update_layout(title='Portfolio')
    ##FINE CALCOLO ANDAMENTO PORTAFOGLIO---------------------------------------------------------------
    
    ##CALCOLO GROSS, NET AND NET-TAXED RETURN----------------------------------------------------------
    grossret = np.round(portfolio[10],2)
    netret = np.round(portfolio[10] - 9900, 2)
    netrettax = np.round(netret-(netret*0.26), 2)

    
    return fig, fig2, fig3, fig4, 'Expected return of last 10 months: {}%'.format(np.round(max_sr_ret*100, 3)), 'Expected return of last 10 months from today: {}%'.format(np.round(max_sr_ret*100, 3)), 'Expected volatility: {}%'.format(np.round(max_sr_vol*100, 3)), 'AIR: {}%'.format(np.round(p_air*100, 4)), 'CCL: {}%'.format(np.round(p_ccl*100, 4)), 'AMZN: {}%'.format(np.round(p_amzn*100, 4)), 'WMT: {}%'.format(np.round(p_wmt*100, 4)), 'CSIQ: {}%'.format(np.round(p_csiq*100, 4)), 'IBDRY: {}%'.format(np.round(p_ibdry*100, 4)), 'Effective return: {}%'.format(np.round(port_simple_ret*100, 4)), 'Gross return: {}€'.format(grossret), 'Net return: {}€'.format(netret), 'Net return (taxed): {}€'.format(netrettax)                                       
    

@app.callback(Output(component_id='beta', component_property='children'),	
              [Input(component_id='beta-stock', component_property='value'),
               Input(component_id='date-picker-range', component_property='start_date'),		
               Input(component_id='date-picker-range', component_property='end_date')])

def update_beta_value(ticker, start, end):
    
    STOCKdf = web.DataReader(ticker, data_source='yahoo', start=start, end=end)
    STOCKdf = STOCKdf.groupby(pd.Grouper(freq='M')).mean()
    STOCK = np.log(STOCKdf["Adj Close"]/STOCKdf["Adj Close"].shift(1))
    STOCK = STOCK.dropna()
    STOCK.name = ticker

    SP500df = web.DataReader('^GSPC', data_source='yahoo', start=start, end=end)
    SP500df = SP500df.groupby(pd.Grouper(freq='M')).mean()
    SP500 = np.log(SP500df["Adj Close"]/SP500df["Adj Close"].shift(1))
    SP500 = SP500.dropna()
    SP500.name = "SP500"
    
    beta_STOCK = STOCK.cov(SP500) / SP500.var()
    beta_STOCK = np.round(beta_STOCK,3)
    
    return 'Beta: {}'.format(beta_STOCK)


@app.callback(Output(component_id='beta-graph', component_property='figure'),	
              [Input(component_id='beta-stock', component_property='value'),
               Input(component_id='delta-beta', component_property='value')])

def update_beta_graph(ticker, delta):
    
    STOCKdf = web.DataReader(ticker, data_source='yahoo', start=dt(2010, 1, 1), end=dt.now())
    STOCKdf = STOCKdf.groupby(pd.Grouper(freq='M')).mean()
    STOCK = np.log(STOCKdf["Adj Close"]/STOCKdf["Adj Close"].shift(1))
    STOCK = STOCK.dropna()
    STOCK.name = ticker

    SP500df = web.DataReader('^GSPC', data_source='yahoo', start=dt(2010, 1, 1), end=dt.now())
    SP500df = SP500df.groupby(pd.Grouper(freq='M')).mean()
    SP500 = np.log(SP500df["Adj Close"]/SP500df["Adj Close"].shift(1))
    SP500 = SP500.dropna()
    SP500.name = "SP500"
    
    delta_t = delta
    
    def beta_function(stock, market_index):
        beta = stock.cov(market_index) / market_index.var()
        return(beta)
    
    STOCK_betas = pd.Series()
    length_period = SP500.shape[0]
    
    start = delta_t
    
    for i in range(start,length_period):
    
        beta_val_STOCK =  beta_function(STOCK[i-delta_t:i-1], SP500[i-delta_t:i-1])
    
        beta_STOCK = pd.Series([beta_val_STOCK], index=STOCK.index[[i]])
        
        STOCK_betas = pd.concat([STOCK_betas, beta_STOCK], axis=0)
        
    NANs = pd.Series(np.repeat(None, (delta_t)), index=STOCK.index[0:delta_t])
    STOCK_betas = pd.concat([NANs, STOCK_betas], axis=0)
    
    fig = px.line(STOCK_betas, title="Beta serie of: " + ticker)
    fig.update_layout(showlegend = False)
    
    return fig


@app.callback(Output('markowitz-past', 'figure'),
              Output('past-ret', 'children'),
              Output('past-vol', 'children'),
              [Input('fake-input', 'value')])

def markowitz_past(selected_dropdown_value):
    
    AIR = web.DataReader('AIR.PA', data_source='yahoo', start=dt(2010, 1, 1), end=dt.now())
    CCL = web.DataReader('CCL', data_source='yahoo', start=dt(2010, 1, 1), end=dt.now())
    AMZN = web.DataReader('AMZN', data_source='yahoo', start=dt(2010, 1, 1), end=dt.now())
    WMT = web.DataReader('WMT', data_source='yahoo', start=dt(2010, 1, 1), end=dt.now())
    CSIQ = web.DataReader('CSIQ', data_source='yahoo', start=dt(2010, 1, 1), end=dt.now())
    IBDRY = web.DataReader('IBDRY', data_source='yahoo', start=dt(2010, 1, 1), end=dt.now())
    
    AIR = AIR.groupby(pd.Grouper(freq='M')).mean() 
    CCL = CCL.groupby(pd.Grouper(freq='M')).mean()
    AMZN = AMZN.groupby(pd.Grouper(freq='M')).mean()
    WMT = WMT.groupby(pd.Grouper(freq='M')).mean()
    CSIQ = CSIQ.groupby(pd.Grouper(freq='M')).mean()
    IBDRY = IBDRY.groupby(pd.Grouper(freq='M')).mean()
    
    AIR = AIR['Adj Close']
    CCL = CCL['Adj Close']
    AMZN = AMZN['Adj Close']
    WMT = WMT['Adj Close']
    CSIQ = CSIQ['Adj Close']
    IBDRY = IBDRY['Adj Close']
    
    AIR = np.log(AIR/AIR.shift(1))
    CCL = np.log(CCL/CCL.shift(1))  
    AMZN = np.log(AMZN/AMZN.shift(1))
    WMT = np.log(WMT/WMT.shift(1))
    CSIQ = np.log(CSIQ/CSIQ.shift(1))
    IBDRY = np.log(IBDRY/IBDRY.shift(1))
    
    AIR = AIR.dropna()
    CCL = CCL.dropna()
    AMZN = AMZN.dropna()
    WMT = WMT.dropna()
    CSIQ = CSIQ.dropna()
    IBDRY = IBDRY.dropna()
    
    data = pd.concat([AIR, CCL, AMZN, WMT, CSIQ, IBDRY], axis=1)
    data.columns = ['AIR','CCL','AMZN','WMT','CSIQ','IBDRY']
    data = data.dropna()
    
    #dataset_I = data.tail(10)
    log_ret = data.head(len(data.index)-10)           #Dataset che uso per il modello
     
    
    np.random.seed(42)
    num_ports = 20000
    all_weights = np.zeros((num_ports, len(data.columns)))
    ret_arr = np.zeros(num_ports)
    vol_arr = np.zeros(num_ports)
    sharpe_arr = np.zeros(num_ports)

    for x in range(num_ports):
        # Weights
        weights = np.array(np.random.random(6))
        weights = weights/np.sum(weights)
    
        # Save weights
        all_weights[x,:] = weights
    
        # Expected return
        ret_arr[x] = np.sum( (log_ret.mean() * weights * 12))
    
        # Expected volatility
        vol_arr[x] = np.sqrt(np.dot(weights.T, np.dot(log_ret.cov()*12, weights)))
    
        # Sharpe Ratio
        sharpe_arr[x] = (ret_arr[x] - 0.001 )/vol_arr[x]


    max_sr_ret = ret_arr[sharpe_arr.argmax()]
    max_sr_vol = vol_arr[sharpe_arr.argmax()]
    
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=vol_arr, y=ret_arr,
        name='portfolios',
        mode='markers',
        marker = dict(
            color=sharpe_arr, 
            colorscale='Viridis')
        ))

    fig.add_trace(go.Scatter(
        x=[max_sr_vol], y=[max_sr_ret],
        name='best sharpe-ratio port.',
        marker_color='red'
        ))
    
    till = dt.today()-pd.DateOffset(months=10)
    
    # Set options common to all traces with fig.update_traces
    fig.update_traces(mode='markers')
    fig.update_layout(title='Markowitz model without forecast calculated from Jan 2010 till {}'.format(till.strftime("%d %b %Y")) , height=700)
    
    return fig, 'Expected return: {}%'.format(np.round(max_sr_ret*100, 3)), 'Expected volatility: {}%'.format(np.round(max_sr_vol*100, 3))


if __name__ == '__main__':
    app.run_server()