from dash import Dash, html, dcc, dash_table, Input, Output, callback, State, no_update
import plotly.express as px
import pandas as pd
import dash_mantine_components as dmc
from dash_iconify import DashIconify
import dash_bootstrap_components as dbc
from general_utils import *
from graph_utils import *

##############################
# functions to be moved to utils
def get_tiramisu_code(dataset_source_path,current_prog):
    with open(dataset_source_path+current_prog+'_generator.cpp', 'r') as f:
        prog_str = f.read() 
    return prog_str

def get_schedule_text(simplifed_orig_sched_str, merged_decls):
    
    merged_sched_str = ''
    for decl in merged_decls:
        
        if decl[1]=='':
            s =  ''
        else:
            for sdecl in split_to_single_transfs(decl):
                s = ('{'+','.join(sdecl[0])+'}->'+''.join(sdecl[1]))
                s = s.replace('P','Parallelize')
                s = s.replace('I','Interchnage')
                s = s.replace('S','Skew')
                s = s.replace('R','Reverse')
                s = s.replace('T','Tile')
                s = s.replace('U','Unroll')
                s = s.replace('F','Fuse')
                merged_sched_str+=(s+'\n')
    # return transfomations commands as text +?a formatted matrix?
    pass
#     return '''
# Parallelize({comp00,comp01}, L1);
# Tile({comp00}, L1,32,L2,64);
# // Affine transfomations above equivalent to
# [[1,2,2],
#  [1,1,5],
#  [1,3,1]]
#     '''
    return merged_sched_str

def compute_latent_space(programs_dict, model, curr_prog_name, method, dim):
    print('computing projections')
    sched_ids = 'all'
    selected_X = get_embed_by_funcname_schedids(programs_dict, model, curr_prog_name, sched_ids)
    if method == 'tsne':
        tsne = TSNE(n_components=dim, 
                random_state=0, 
                n_jobs=6, 
                verbose=1,
                n_iter=4000, 
                init='pca', 
                perplexity=40
            )
        projections = tsne.fit_transform(selected_X)
    if method == 'pca':
        pca = PCA(n_components=3)
        projections = pca.fit_transform(selected_X)
    print('done computing projections')
    return projections


def get_latent_space_chart(full_scheds_df,programs_dict, model, curr_prog_name, curr_sched_name, method, dim, coloration):
    projections = compute_latent_space(programs_dict, model, curr_prog_name, method, dim)
    if coloration=='target':
        colorscale_label = 'Speedup<br><sup>in base log10</sup>'
        colors=np.log10(full_scheds_df['target'])
    elif coloration=='prediction':
         colorscale_label = 'Predicted Speedup<br><sup>in base log10</sup>'
         colors=np.log10(full_scheds_df['prediction'])
    elif coloration=='APE':
         colorscale_label = 'APE'
         colors = full_scheds_df['APE']
    if dim ==3:
        fig = px.scatter_3d(
                projections, x=0, y=1, z=2,
                color=colors, 
                labels={'color': colorscale_label},
                template='plotly_dark',
                custom_data=[full_scheds_df['target'],full_scheds_df['prediction'],full_scheds_df['sched_name'],full_scheds_df['sched_str']]
            )
        fig.layout.scene = dict(annotations=[dict(
            x=projections[int(curr_sched_name)][0],
            y=projections[int(curr_sched_name)][1],
            z=projections[int(curr_sched_name)][2],
            ax=0,
            ay=-40,
            text="Schedule "+str(curr_sched_name).zfill(4),
            arrowhead=1,
            xanchor="auto",
            yanchor="auto",
            font_color=kindofteal,
            arrowcolor = '#34c79b'
        )])
        
    else:
        fig = px.scatter(
                projections, x=0, y=1,
                color=colors, labels={'color': colorscale_label},
                template='plotly_dark',
                custom_data=[full_scheds_df['target'],full_scheds_df['prediction'],full_scheds_df['sched_name'],full_scheds_df['sched_str']]
            )
        fig.add_annotation(
            x=projections[int(curr_sched_name)][0],
            y=projections[int(curr_sched_name)][1],
            ax=0,
            ay=-40,
            text="Schedule "+str(curr_sched_name).zfill(4),
            arrowhead=1,
            arrowsize=1,
            arrowwidth=2,
            xanchor="auto",
            yanchor="auto",
            font_color=kindofteal,
            arrowcolor = '#34c79b'
        )
    fig.update_traces(
        hovertemplate="<br>".join([
            "Speedup: %{customdata[0]}",
            "Prediction: %{customdata[1]}",
            "Name: %{customdata[2]}",
            "Sched: %{customdata[3]}"
        ])
        )
        


    fig.update_layout(
        # title=go.layout.Title(
        #     text="Projected Embedding Space",
        #     # xref="paper",

        # ),
        xaxis=go.layout.XAxis(
            title=go.layout.xaxis.Title(
                text="X"
            )
        ),
        yaxis=go.layout.YAxis(
            title=go.layout.yaxis.Title(
                text="Y"
            )
        ),
    )
    

    fig.update_traces(marker_size=3 if dim==3 else 5)
    return fig

def get_ast_fig(program_json, schedule_json, initial_schedule_json, source_code, sched_str_new, decorate=False):
    
    shapes,positions = get_children_shapes(initial_schedule_json['tree_structure'], 0, 0)
    annots = make_fig_hoverable(program_json, positions, source_code)
    
  #   merged_decls =[(['comp00', 'comp01'], 'F(L2)'),
  # (['comp00'], 'T3(L0,L1,L2,32,32,32)'),
  # (['comp00', 'comp01'], 'I(L1,L2)R(L0)S(L1,L2,1,2)P(L0)')]
    
    
    decorations_annots = []
    decorations_shapes = []
    if decorate:
        # print(sched_str_new)
        decorations_annots, decorations_shapes = get_sched_decorations(program_json,schedule_json,sched_str_new,positions)
    
    fig = go.Figure()
    fig.update_layout(annotations=annots+decorations_annots)
    # Add shapes
    fig.update_layout(
        shapes=shapes+decorations_shapes
    )
    fig.update_yaxes(scaleanchor = "x", 
                     range=[-15, 0.2], #### hardcoded
                     scaleratio = 1, 
                     showgrid=False, visible =  False, showticklabels= False)
    # # Update axes properties
    fig.update_xaxes(showgrid=False, visible =  False, showticklabels= False
    #     range=[0, 1],
    #     # zeroline=True,
    #     # domain=[0.25,0.5],
    #     # fixedrange=True
    )

    # fig.update_yaxes(
    #     # range=[0, 1],
    #     # zeroline=False,
    # )
    # fig.update_traces(textposition='bottom center')
    fig.update_layout(template='plotly_dark',
                      dragmode = 'pan',
                      margin=dict(
        l=10,
        r=10,
        b=10,
        t=10,
        pad=4
    ))
    

    return fig

def get_sched_contribs_plot(model, programs_dict, function_name, schedule_index):
    contribs_dict,prediction = get_indev_sched_contribs(model, programs_dict, function_name, schedule_index)    
    base = 1
    x=[]
    y=[]
    text=[]
    
    for k,v in sorted([(k,v) for k,v in contribs_dict.items()], key=lambda x:abs(np.log10(x[1]))):
        x.append(v-base)
        y.append(k)
        text.append('<b>'+str(round(v,2))+'</b><i>X<i>')
        
    fig = go.Figure(go.Bar( x=x,
                            y=y,
                            orientation='h', 
                            base=base, 
                            text = text,
                            marker=dict(color = np.log10([i+base for i in x]),
                            colorscale='plasma')
                          )
                   )
    
    # fig.update_layout(
    #     font=dict(
    #         size=18,  # Set the font size here
    #         color="white"
    #     )
    # )
    fig.update_layout(
        title=go.layout.Title(
            text='Final Predicted Speedup <b>'+str(round(prediction,2))+'</b><i>X<i>',
            # xref="paper",
            x=0.5,
        )
    )
    fig.update_layout(yaxis = dict(tickfont = dict(size=10)))
    fig.update_xaxes(type="log")
    fig.update_xaxes(title="<sup>Contributions to the Final Predicted Speedup</sup>")
    fig.add_vline(x=1, line_width=3, line_dash="solid", line_color="darkgrey")
    fig.update_layout(template='plotly_dark')
    fig.update_traces(textposition=['auto'],textfont_size=16)
    return fig

def get_err_corr_fig(full_scheds_df,schedule_index):
    selected_df = full_scheds_df.sort_values(['target'],ascending=False).reset_index()
    row = selected_df.query('sched_name==@schedule_index')
    selected_sched_x = float(row['target'])
    # selected_sched_x = selected_df.query('sched_name==@schedule_index').index[0]
    selected_sched_y = float(row['prediction'])


    fig = go.Figure(go.Scatter(x=selected_df['target'], y=selected_df['target'],opacity=1,
                        mode='lines', name='Ideal Prediction',marker = dict(
            color= 'grey',
                            
                ) ))


    fig.add_trace(go.Scatter(
        x=selected_df['target'], 
        y=selected_df['prediction'],
        # color=np.log10(selected_df['target']),
        mode='markers', 
        name='Predicted Speedup', 
        marker = dict(
            color= selected_df['APE'],
                colorscale= 'plasma',
                size= 6,
                )
    ))
    fig.update_layout(template='plotly_dark')
    fig.add_annotation(
                x=np.log10(selected_sched_x),
                y=np.log10(selected_sched_y),
                ax=0,
                ay=-40,
                text="Schedule "+str(schedule_index).zfill(4),
                arrowhead=1,
                arrowsize=1,
                arrowwidth=2,
                xanchor="auto",
                yanchor="auto",
                font_color=kindofteal,
                arrowcolor = kindofteal
            )

    fig.update_yaxes(type="log", title="<sup>Actual Speedup</sup>")
    fig.update_xaxes(type="log", title="<sup>Predicted Speedup</sup>")
    return fig

def get_curr_prog_footer_cpn(filtered_progs_df,curr_prog_name):
    row = filtered_progs_df.query('name==@curr_prog_name')
    mape = str(round(float(row['MAPE']),2))+'%'
    return dmc.Table(id = 'prog_select_table', children=[
        html.Thead(html.Tr([ html.Th("Selected Program"), html.Th("Schedules Count"), html.Th("Average Error"),]))]
        + [html.Tbody([ html.Tr([html.Td(dmc.Badge(curr_prog_name,size='lg', variant="filled", color='teal')), html.Td(row['sched_count']), html.Td(mape)])])])
def get_curr_sched_footer_cpn(filtered_scheds_df,curr_sched_name):
    row = filtered_scheds_df.query('sched_name==@curr_sched_name')
    target = str(round(float(row['target']),2))+'X'
    pred = str(round(float(row['prediction']),2))+'X'
    ape = str(round(float(row['APE']),2))+'%'
    sched_str = row['sched_str'] if len(str(row['sched_str']))<20 else str(row['sched_str'].iloc[0])[:17]+'...'
    name = str(curr_sched_name).zfill(4)
    # print(filtered_scheds_df)
    return dmc.Table(id = 'sched_select_table', children=[
        html.Thead(html.Tr([html.Th("Selected Schedule"), html.Th("Schedule Code"), html.Th("Speedup"), html.Th("Prediction"), html.Th("Error")]))]
        + [html.Tbody([ html.Tr([html.Td(dmc.Badge(name,size='lg', variant="filled", color='teal')), html.Td(sched_str), html.Td(target), html.Td(pred), html.Td(ape)])])])

def get_full_scheds_df(full_df,curr_prog_name):
    return full_df.query('name==@curr_prog_name', engine="python").reset_index()[['sched_name','sched_str','prediction','target','APE']].round({'prediction':2}).sort_values(['sched_name'])

def get_schedules_datatable_cpn(filtered_scheds_df,curr_sched_name):
    if curr_sched_name=='auto':
        page_current = 0
        selected_rows = [0]
    else:
        page_current =  int(filtered_scheds_df.query('sched_name==@curr_sched_name').index[0]/tables_page_size)
        selected_rows = [filtered_scheds_df.query('sched_name==@curr_sched_name').index[0]]
    return dash_table.DataTable(data=filtered_scheds_df.to_dict('records'), 
                            page_size=tables_page_size, 
                            page_current =  page_current,
                            selected_rows = selected_rows,
                            id='scheds_table', 
                            row_selectable='single',
                            style_header={
                                'backgroundColor': 'rgb(30, 30, 30)',
                                'color': 'white'
                            },
                            style_data={
                                'backgroundColor': 'rgb(50, 50, 50)',
                                'color': 'white'
                            },
                        #   style_cell={},
                            style_as_list_view=True,
                            style_cell={
                            'overflow': 'hidden',
                            'textOverflow': 'ellipsis',
                            'maxWidth': 500,
                            'textAlign': 'left'
                        },
                        tooltip_data=[
                            {
                                column: {'value': str(value), 'type': 'markdown'}
                                for column, value in row.items()
                            } for row in filtered_scheds_df.to_dict('records')
                        ],

                            )   

def get_programs_datatable_cpn(filtered_progs_df, curr_prog_name):
    if curr_prog_name=='auto':
        page_current = 0
        selected_rows = [0]
    else:
        page_current =  int(filtered_progs_df.query('name==@curr_prog_name').index[0]/tables_page_size)
        selected_rows = [filtered_progs_df.query('name==@curr_prog_name').index[0]]
    return dash_table.DataTable(data=filtered_progs_df.to_dict('records'), 
                                    page_size=tables_page_size,
                                    page_current =  page_current,
                                    selected_rows = selected_rows,
                                    id='progs_table', 
                                    row_selectable='single',
                                    style_header={
                                        'backgroundColor': 'rgb(30, 30, 30)',
                                        'color': 'white'
                                    },
                                    style_data={
                                        'backgroundColor': 'rgb(50, 50, 50)',
                                        'color': 'white'
                                    },
                                    style_cell={'textAlign': 'left'},
                                    style_as_list_view=True,

                                    )
def get_src_code_prism(current_source_code):
    return dmc.Prism(language='cpp',
            withLineNumbers=True,
            style={'height': '100%'},
            noCopy = True,
            id='src_code_prism',
            children=[current_source_code+('\n '*max(0,30-current_source_code.count('\n')))+'.'])
def get_src_code_prism_for_modal(current_source_code):
    return dmc.Prism(language='cpp',
            withLineNumbers=True,
            style={'height': '100%'},
            noCopy = False,
            id='src_code_prism_modal',
            children=[current_source_code+('\n '*max(0,30-current_source_code.count('\n')))+'.'])

def get_sched_code_prism(current_sched_code):
    return dmc.Prism(language='cpp',
            withLineNumbers=True,
            style={'height': '100%'},
            noCopy = True,
            children=[current_sched_code + ('\n '*max(0,10-current_sched_code.count('\n')))+'.'])

##############################
###### Constant Globals


full_df = pd.read_csv('./dataset_batch780844-838143_val_tiny_filtered.df.csv').sort_values(['name','sched_name']).fillna('-').round(2)
# full_df = pd.read_csv('./dataset_batch799000-803000_val_tiny.df.csv').sort_values(['name','sched_name']).fillna('-')
full_progs_df = full_df.groupby('name').agg({'sched_str':'count', 'memory_use':'mean', 'APE':'mean'}).reset_index().rename(columns={'sched_str':'sched_count', 'APE':'MAPE'}).round(2)
# dataset_source_path = './dataset_batch799000-803000_val_generators/'
dataset_source_path = './dataset_batch780844-838143_val_tiny_filtered_generators/'
dataset_file = './dataset_batch780844-838143_val_tiny.pkl'
model = Model_Recursive_LSTM_v2(776)
print('loading model')
model.load_state_dict(torch.load('MAPE_base_visml_proj_26.9.pkl',map_location='cpu'))
model.eval()
print('loading data')
with open(dataset_file, 'rb') as f:
    programs_dict = pickle.load(f)  
print('loaded')

#### Temoporary dummies
model_versions = ['./models/Rec-LSTM-Mcomps-20.8-1.2.pkl','./models/Rec-LSTM-Scomps-13.8-0.9.pkl', './models/Flat-LSTM-Scomps-13.8-1.1.pkl'] #dummies for now
datasets_versions = ['./datasets/batch799000-803000_val.pkl','./datasets/batch803000-807000_val.pkl','./datasets/batch807000-811000_val.pkl']

# dummy_df = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/gapminder2007.csv')
# dummy_plot = px.histogram(dummy_df, x='continent', y='lifeExp', histfunc='avg')

# dummy_plot.layout['height'] = 500


##############################
#### Global Variables Initialization
curr_prog_name = 'function795487'
curr_sched_name = 141


### updated from drawer
filtered_progs_df = full_progs_df
full_scheds_df = get_full_scheds_df(full_df,curr_prog_name)
filtered_scheds_df = full_scheds_df


### needs explicit update
program_json = programs_dict[curr_prog_name]['program_annotation']
schedule_json = programs_dict[curr_prog_name]['schedules_list'][curr_sched_name]
initial_schedule_json = programs_dict[curr_prog_name]['schedules_list'][0]

base_sched_str = str(filtered_scheds_df.query('sched_name==@curr_sched_name')['sched_str'].iloc[0])
simplified_base_sched_str = simplify_sched_str(base_sched_str)
new_sched_str = get_schedule_str_new(program_json,schedule_json)
merged_new_sched_str = merge_all_decls(new_sched_str, get_orig_declarations(new_sched_str), program_json, schedule_json) #lists of tuples eg [...(['comp00', 'comp01'], 'I(L1,L2)R(L1)P(L0)')...]
current_source_code = get_tiramisu_code(dataset_source_path,curr_prog_name)
current_sched_code = get_schedule_text(simplified_base_sched_str, merged_new_sched_str)



##############################
####style variables
badges_gradient={"from": "#2186f4", "to": kindofteal, "deg": 45}
badges_size = 'sm'
badge_color = 'grape'
chips_color= 'teal'
select_color='teal'

##############################

def update_globals():
    global program_json
    global schedule_json
    global initial_schedule_json
    global base_sched_str
    global simplified_base_sched_str
    global new_sched_str
    global merged_new_sched_str
    global current_source_code
    global current_sched_code

    program_json = programs_dict[curr_prog_name]['program_annotation']
    schedule_json = programs_dict[curr_prog_name]['schedules_list'][curr_sched_name]
    initial_schedule_json = programs_dict[curr_prog_name]['schedules_list'][0]

    base_sched_str = str(filtered_scheds_df.query('sched_name==@curr_sched_name')['sched_str'].iloc[0])
    simplified_base_sched_str = simplify_sched_str(base_sched_str)
    new_sched_str = get_schedule_str_new(program_json,schedule_json)
    merged_new_sched_str = merge_all_decls(new_sched_str, get_orig_declarations(new_sched_str), program_json, schedule_json) #lists of tuples eg [...(['comp00', 'comp01'], 'I(L1,L2)R(L1)P(L0)')...]
    current_source_code = get_tiramisu_code(dataset_source_path,curr_prog_name)
    current_sched_code = get_schedule_text(simplified_base_sched_str, merged_new_sched_str)

##############################
#### Components

src_sched_cpn = dmc.Stack(align="flex-start", justify="center", children=[
    html.Div(get_sched_code_prism(current_sched_code), id='src_sched_prism_wrapper'),
    dmc.Badge("Schedule",size=badges_size, variant="light", color=badge_color,gradient=badges_gradient, className='title-badge')
    ])
# print(('A'*max(0,30-current_sched_code.count('\n'))))
src_code_cpn = dmc.Stack(align="flex-start", justify="center", children=[
    dmc.ActionIcon(size="lg",variant="filled",id="src_code_zoom_btn", n_clicks=0,className='zoom-btn',children=[
            DashIconify(icon="ic:round-zoom-out-map", width=20),
        ]), 
    html.Div(get_src_code_prism(current_source_code), id='src_code_prism_wrapper'),
    dmc.Badge("Raw Source Code",size=badges_size, variant="light", color=badge_color,gradient=badges_gradient, className='title-badge'),
     
    ])
src_sched_zoom_btn = ''
err_chart_cpn = dmc.Stack(align="flex-start", justify="center", children=[
    dcc.Graph(id = 'err_chart_graph_cpn',figure=get_err_corr_fig(full_scheds_df,curr_sched_name),responsive=False),
    dmc.Badge("Predicted VS Actual",size=badges_size, variant="light", color=badge_color,gradient=badges_gradient, className='title-badge')
])

latent_chart_cpn = dmc.Stack(align="stretch", justify="center", children=[
    dmc.LoadingOverlay(children=[
        dcc.Graph(id='latent_space_graph_cpn',figure=get_latent_space_chart(full_scheds_df,programs_dict, model, curr_prog_name, curr_sched_name, 'tsne', 3, 'target'), responsive=False),
    ]),
    dmc.Group(position='apart', className='title-badge', children=[
        dmc.Badge("Local Latent Space",size=badges_size, variant="light", color=badge_color,gradient=badges_gradient), 
        dmc.Group(position='left', align='flex-end', spacing='xs',  children=[
            dmc.Select(size='xxs', style={'width':30},id="latent_dim_select"       , value=3, data=[{"value": 3, "label": "3D"},{"value": 2, "label": "2D"}],),
            dmc.Select(size='xxs', style={'width':50},id="latent_method_select"    , value='tsne', data=[{"value": "tsne", "label": "T-SNE"},{"value": "pca", "label": "PCA"},],),
            dmc.Select(size='xxs', style={'width':80},id="latent_coloration_select", value='target', data=[{"value": "APE", "label": "Error"},{"value": "target", "label": "Speedup"},{"value": "prediction", "label": "Prediction"}],),
            ]),
        ]),
    ])


ast_fig_cpn = dmc.Stack(align="stretch", justify="center", children=[
    dcc.Graph(id = 'ast_fig_graph_cpn' ,figure=get_ast_fig(program_json, schedule_json, initial_schedule_json, current_source_code, new_sched_str, decorate=True), responsive=False, 
                        # config={'displayModeBar': 'True'}
                        ),
    dmc.Group(position='apart', children=[dmc.Badge("Program's AST", size=badges_size, variant="light", color=badge_color,gradient=badges_gradient), 
                                          dmc.Chip('Show Schedule', id='decorate_ast_chip', size='xs', color=chips_color, checked=True)],  className='title-badge'), 
])

# ast_fig_wrapper = dmc.Stack(align="flex-start", justify="center", children=[
#     html.Div(dmc.Chip('Show Schedule', id='decorate_ast_chip',mb=-60),style={'z-index': '10'}),
#     ast_fig_cpn
# ])
contrib_chart_cpn = dmc.Stack(align="flex-start", justify="center", children=[
    dcc.Graph(id='contrib_chart_graph_cpn',figure=get_sched_contribs_plot(model, programs_dict, curr_prog_name, curr_sched_name), responsive=False),
    dmc.Badge("Transformations Contributions",size=badges_size, variant="light", color=badge_color,gradient=badges_gradient, className='title-badge')
])


prog_filter_reset_btn = dmc.ActionIcon(DashIconify(icon="fluent:arrow-reset-24-filled"),
            size="lg",
            variant="filled",
            id="prog_filter_reset_btn",
            n_clicks=0,
            mt=24
        )
prog_filter_apply_btn = dmc.ActionIcon(DashIconify(icon="material-symbols:filter-alt"),
            size="lg",
            variant="filled",
            id="prog_filter_apply_btn",
            n_clicks=0,
            mt=24
        )

sched_filter_reset_btn = dmc.ActionIcon(DashIconify(icon="fluent:arrow-reset-24-filled"),
            size="lg",
            variant="filled",
            id="sched_filter_reset_btn",
            n_clicks=0,
            mt=24
        )
sched_filter_apply_btn = dmc.ActionIcon(DashIconify(icon="material-symbols:filter-alt"),
            size="lg",
            variant="filled",
            id="sched_filter_apply_btn",
            n_clicks=0,
            mt=24
        )
load_model_btn = dmc.ActionIcon(DashIconify(icon="material-symbols:folder-open-outline-rounded"),size="lg",variant="filled",n_clicks=0,mt=14)
ds_model_btn = dmc.ActionIcon(DashIconify(icon="material-symbols:folder-open-outline-rounded"),size="lg",variant="filled",n_clicks=0,mt=14)

tables_page_size = 5
programs_datatable_cpn = get_programs_datatable_cpn(filtered_progs_df, curr_prog_name)
schedules_datatable_cpn = get_schedules_datatable_cpn(filtered_scheds_df,curr_sched_name)

dp_select_cpn = dmc.Drawer(id="dp_select_drawer",zIndex=10000,position='bottom',overlayBlur=4,overlayOpacity = 0.5,size="55%", children=[
    dmc.Center(children=[
        dmc.Stack(align="stretch",justify="center",children=[
            dmc.Divider(variant="solid",label='Model & Dataset Selection',size=1),
            dmc.SimpleGrid(cols=2, children=[
                dmc.Group(children=[
                    dmc.Select(label="Model version",id="model_ver_select",data=model_versions,value =model_versions[0], style={"width": 500, "marginBottom": 10},),
                        load_model_btn,
                        ]),
                dmc.Group(children=[
                    dmc.Select(label="Datasets",id="ds_select",data=datasets_versions,value =datasets_versions[0], style={"width": 500, "marginBottom": 10},),
                        load_model_btn,
                        ]),
            ]),

            dmc.Divider(variant="solid",label='Program & Schedule Selection',size=1),
            dmc.SimpleGrid([
                dmc.Group(children=[
                    dmc.TextInput(id='filter_prog_query_input', label="Filter programs by query:",placeholder='e.g: sched_cout>200 and APE<20', style={"width": 400}),
                    prog_filter_reset_btn,
                    prog_filter_apply_btn
                    ]),
                # 'p',
                dmc.Group(children=[
                    dmc.TextInput(id='filter_sched_query_input',label="Filter schedules by query:",placeholder='e.g: sched_str.str.contains("P(L0)") and prediction<2', style={"width": 400},),
                    sched_filter_reset_btn,
                    sched_filter_apply_btn
                    ]),
                ], cols=2),
            dmc.SimpleGrid(children=[
                html.Div(programs_datatable_cpn, id='programs_datatable_cpn_wrapper'),
                html.Div(schedules_datatable_cpn, id='schedules_datatable_cpn_wrapper')
                ], cols=2),
])])])





curr_prog_footer = dmc.Center(id='curr_prog_footer_wrapper', children=[get_curr_prog_footer_cpn(filtered_progs_df,curr_prog_name)])
curr_sched_footer = dmc.Center(id='curr_sched_footer_wrapper', children=[get_curr_sched_footer_cpn(filtered_scheds_df,curr_sched_name)])
pullup_btn =  dmc.Center(dmc.ActionIcon(
            DashIconify(icon="material-symbols:keyboard-double-arrow-up-rounded", height=75),
            size="xl",
            variant="transparent",
            id="pullup_btn",
            n_clicks=0,
        ))
footer_content_cpn = dmc.SimpleGrid([curr_prog_footer,pullup_btn,curr_sched_footer],cols=3)





##############################
external_stylesheets = ['./style.css']
app = Dash(__name__, external_stylesheets=external_stylesheets)





app.layout = dmc.MantineProvider(theme={"colorScheme": "dark"},
    children=[
    html.Div(id='parent', children=[
    html.Div(id = "div1",children=[footer_content_cpn]),
    html.Div(id = "div2",children=[contrib_chart_cpn]),
    html.Div(id = "div3",children=[err_chart_cpn]),
    html.Div(id = "div4",children=[src_sched_cpn]),
    html.Div(id = "div5",children=[latent_chart_cpn]),
    html.Div(id = "div6",children=[ast_fig_cpn]),
    html.Div(id = "div7",children=[src_code_cpn]),
    ########## other non attached components
    dp_select_cpn,
    dcc.Store(id='curr_prog_name_store', data={'name':curr_prog_name}),
    dcc.Store(id='temp_selected_prog', data={'name':curr_prog_name}), # used while in drawer before submitting
    dcc.Store(id='curr_sched_name_store', data={'name':curr_sched_name}),
    dmc.Modal(
            id="src_code_modal",
            overflow="inside",
            centered=True,
            size="55%",
            zIndex=10000,
            children=[get_src_code_prism_for_modal(current_source_code)],
        ),
])])



##############################
@callback(
    Output("dp_select_drawer", "opened"),
    Input("pullup_btn", "n_clicks"),
    prevent_initial_call=True,
)
def open_drawer(n_clicks):
    return True

@callback(
    Output("curr_prog_name_store", "data"),
    Output("curr_sched_name_store", "data"),
    Input("dp_select_drawer", "opened"),
    State('progs_table', 'selected_rows'),
    State('scheds_table', 'selected_rows'),
    prevent_initial_call=True,
)
def closed_drawer(opened,prog_rows,sched_rows):
    global filtered_scheds_df
    global filtered_progs_df
    global curr_prog_name
    global curr_sched_name
    if opened: # if the drawer has just opened, don't do anything
        return no_update, no_update
    else: # if the drawer has just closed, update the names
        prog_row = prog_rows[0]
        sched_row = sched_rows[0]
        curr_prog_name = str(filtered_progs_df['name'].iloc[prog_row])
        curr_sched_name = int(filtered_scheds_df['sched_name'].iloc[sched_row])
        print('submission of ', {'name':curr_prog_name}, {'name':curr_sched_name})
        update_globals()
        return {'name':curr_prog_name}, {'name':curr_sched_name}

@callback(
    Output("temp_selected_prog", "data"),
    Input('progs_table', 'selected_rows'),
    prevent_initial_call=True,
    )
def program_selected(selected_rows):
    # global curr_prog_name
    # global curr_sched_name
    selected_row = selected_rows[0]
    temp_selected_prog = str(filtered_progs_df['name'].iloc[selected_row])
    # curr_sched_name = 0
    print('temp selection of', temp_selected_prog)
    return {'name':temp_selected_prog}

@callback(
    Output('schedules_datatable_cpn_wrapper', 'children'),
    State('filter_sched_query_input', 'value'),
    Input("temp_selected_prog", "data"),
    Input('sched_filter_apply_btn','n_clicks'),
    prevent_initial_call=True,
    ) #update the scheds table and the global filtered df and the global full scheds df, called either by the filter function or temp_selecting a prog from the prog table
def update_sched_table(query_str, data, n_clicks):
    global filtered_scheds_df # TODO should I update it here or on drawer closed ? Yes have to so that can get the id using selected rows on drawer closed
    # temp_full_scheds_df = get_full_scheds_df(full_df,data['name']) #improvement, check if callback caused by filtering, if yes no need to call this line, use the global full_scheds_df
    global full_scheds_df
    full_scheds_df = get_full_scheds_df(full_df,data['name']) #improvement, check if callback caused by filtering, if yes no need to call this line, use the global full_scheds_df
    if query_str =='':
        filtered_scheds_df = full_scheds_df
    else:
        filtered_scheds_df = full_scheds_df.query(query_str, engine='python')
    print('updated filtered_scheds_df with "',data['name'],query_str, '" number of results', len(filtered_scheds_df) )
    return [get_schedules_datatable_cpn(filtered_scheds_df, 'auto')]

@callback(
    Output('programs_datatable_cpn_wrapper', 'children'),
    State('filter_prog_query_input', 'value'),
    Input('prog_filter_apply_btn','n_clicks'),
    prevent_initial_call=True,
    ) #update the progs table and the global filtered df, called  by the filter function 
def update_prog_table(query_str, n_clicks):
    global filtered_progs_df 
    if query_str =='':
        filtered_progs_df = full_progs_df
    else:
        filtered_progs_df = full_progs_df.query(query_str, engine='python')
    print('updated filtered_progs_df with "',query_str, '" number of results', len(filtered_progs_df) )
    return [get_programs_datatable_cpn(filtered_progs_df, 'auto')]

@callback(
    Output('filter_sched_query_input', 'value'),
    Output('sched_filter_apply_btn','n_clicks'),
    Input("sched_filter_reset_btn", "n_clicks"),
    prevent_initial_call=True,
    ) 
def reset_sched_filters(n_clicks):
    return '', -1

@callback(
    Output('filter_prog_query_input', 'value'),
    Output('prog_filter_apply_btn','n_clicks'),
    Input("prog_filter_reset_btn", "n_clicks"),
    prevent_initial_call=True,
    ) 
def reset_progs_filters(n_clicks):
    print('haddd')
    return '', -1

@callback(
    Output('ast_fig_graph_cpn', 'figure'),
    Input("curr_prog_name_store", "data"),
    Input("curr_sched_name_store", "data"),
    Input('decorate_ast_chip', 'checked'),
    prevent_initial_call=True,
)
def update_ast_fig(prog_name_data,sched_name_data,checked):
    # print(checked)
    return get_ast_fig(program_json, schedule_json, initial_schedule_json, current_source_code, new_sched_str, decorate=checked)


@callback(
    Output('src_code_prism_wrapper', 'children'),
    Input("curr_prog_name_store", "data"),
    prevent_initial_call=True,
)
def update_code_prism(prog_name_data):
    return get_src_code_prism(current_source_code)

@callback(
    Output('src_sched_prism_wrapper', 'children'),
    Input("curr_prog_name_store", "data"),
    Input("curr_sched_name_store", "data"),
    prevent_initial_call=True,
)
def update_sched_prism(prog_name_data,sched_name_data):
    return get_sched_code_prism(current_sched_code)


@callback(
    Output('err_chart_graph_cpn', 'figure'),
    Input("curr_prog_name_store", "data"),
    Input("curr_sched_name_store", "data"),
    prevent_initial_call=True,
)
def update_err_chart(prog_name_data,sched_name_data):
    return get_err_corr_fig(full_scheds_df,curr_sched_name)

@callback(
    Output('contrib_chart_graph_cpn', 'figure'),
    Input("curr_prog_name_store", "data"),
    Input("curr_sched_name_store", "data"),
    prevent_initial_call=True,
)
def update_contrib_chart(prog_name_data,sched_name_data):
    return get_sched_contribs_plot(model, programs_dict, curr_prog_name, curr_sched_name)

@callback(
    Output('latent_space_graph_cpn', 'figure'),
    Input("curr_prog_name_store", "data"),
    Input("curr_sched_name_store", "data"),
    Input('latent_dim_select', 'value'),
    Input('latent_method_select', 'value'),
    Input('latent_coloration_select', 'value'),
    prevent_initial_call=True,
)
def update_latent_space_chart(prog_name_data,sched_name_data,dim,method,coloration):
    return get_latent_space_chart(full_scheds_df,programs_dict, model, curr_prog_name, curr_sched_name, method, dim, coloration)

@callback(
    Output('curr_prog_footer_wrapper', 'children'),
    Output('curr_sched_footer_wrapper', 'children'),
    Input("curr_prog_name_store", "data"),
    Input("curr_sched_name_store", "data"),
    prevent_initial_call=True,
)
def update_footer(prog_name_data,sched_name_data):
    return get_curr_prog_footer_cpn(filtered_progs_df,curr_prog_name), get_curr_sched_footer_cpn(filtered_scheds_df,curr_sched_name)

@callback(
    Output(f"src_code_modal", "opened"),
    Output(f"src_code_modal", "children"),
    Input(f"src_code_zoom_btn", "n_clicks"),
    State(f"src_code_modal", "opened"),
    prevent_initial_call=True,
)
def toggle_modal(n_clicks, opened):
    return not opened, [get_src_code_prism_for_modal(current_source_code)]


if __name__ == '__main__':
    app.run_server(debug=True)



#in footer cut sched id when too long or make td scrollable
#in sched tables, use new sched str instead?
#round numbers in tables

#done make and 'update func' that recomputes all the global variables whenever they have to
#done for each section create a callback sec_update() triggered by some dcc.store keys, these keys are are modifed by the whatever update global variables

#add a page header
#fliter out progs that have< 30 scheds otherwise will crash cuz of perplexity 
#unify colors/ add more colors / make sched arrow text color teal #38d9a9


#load df withou speedup clip, with prog that have many sched, with right schedstr
#for error chart, can add a chart where only same scheds are represented
#concatenate the transformatino tags with padding and project to 2d space and color by error?
# for the error chart, try using spline and adding spacing between datapoints on the y axis


# fix the function799714 sched 100 ast issue


# todo make the tables in the drawer fixed size and prevent them from moving
#change color of the ast, badges, sched and func name in the footer


#function815015 25
#function837125 19
#function814323 18


#function786945 101 sched_count>500 and MAPE>20 | target>5 not unroll

#function821423 0965 sched_count>500 and MAPE>20 | last page


#function834908 558
# function795487 141