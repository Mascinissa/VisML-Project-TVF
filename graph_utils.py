from os import environ
from pprint import pprint
import pickle
import numpy as np
import torch 
import pandas as pd
import seaborn as sns
from torch import optim
import time
from tqdm.notebook import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy
import random
import plotly.graph_objects as go
import sys
import re
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
import json
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import plotly.express as px

class NbAccessException(Exception):
    pass
class LoopsDepthException(Exception):
    pass

from general_utils import *

    

def get_dp_representation(programs_dict, function_name, schedule_index):
    program_json = programs_dict[function_name]['program_annotation']
    prog_tree, comps_repr_templates_list, loops_repr_templates_list, comps_placeholders_indices_dict, loops_placeholders_indices_dict = get_representation_template(programs_dict[function_name], max_depth = 5)
    schedule_json = programs_dict[function_name]['schedules_list'][schedule_index]
    comps_tensor, loops_tensor = get_schedule_representation(program_json, schedule_json, comps_repr_templates_list, loops_repr_templates_list, comps_placeholders_indices_dict, loops_placeholders_indices_dict, 5)
    return prog_tree, comps_tensor, loops_tensor
def get_sched_json(programs_dict, function_name, schedule_index):
    return programs_dict[function_name]['schedules_list'][schedule_index]

# get the embed vector of a datapoint diven it's index in a df
def get_embed_by_df_index(programs_dict, model, df, i):
    function_name = df['name'].iloc[i]
    sched_name = df['sched_name'].iloc[i]
    sched_index = int(sched_name)
    reprsnt = get_dp_representation(programs_dict, function_name, sched_index)
    return model.get_last_embed(reprsnt)[0][0]
    
def get_embed_by_funcname_schedids(programs_dict, model, function_name, sched_ids):
    comps_list=[]
    loops_list=[]
    if sched_ids=='all':
        sched_ids = list(range(len(programs_dict[function_name]['schedules_list'])))
    for i in sched_ids:
        tree, comps, loops =  get_dp_representation(programs_dict, function_name, i)
        comps_list.append(comps)
        loops_list.append(loops)
    reprsnt_tensor= tree, torch.cat(comps_list,0), torch.cat(loops_list,0)
    # print(model(reprsnt_tensor)[0])
    return model.get_last_embed(reprsnt_tensor).squeeze().detach().numpy()
    
def get_embed_by_df_indices(programs_dict, model, df, indices):
    comps_list=[]
    loops_list=[]
    for i in indices:
        function_name = df['name'].iloc[i]
        sched_name = df['sched_name'].iloc[i]
        sched_index = int(sched_name)
        tree, comps, loops =  get_dp_representation(programs_dict, function_name, sched_index)
        comps_list.append(comps)
        loops_list.append(loops)
    reprsnt_tensor= tree, torch.cat(comps_list,0), torch.cat(loops_list,0)
    return model.get_last_embed(reprsnt_tensor).squeeze().detach().numpy()

################################################
### AST fig utils

def generate_rounded_rectangle_path(height, width, corner_radius, x=0, y=0):
    path = f"M{x + corner_radius} {y}"
    path += f"H{x + width - corner_radius}"
    path += f"C{x + width} {y} {x + width} {y + corner_radius} {x + width} {y + corner_radius}"
    path += f"V{y + height - corner_radius}"
    path += f"C{x + width} {y + height} {x + width - corner_radius} {y + height} {x + width - corner_radius} {y + height}"
    path += f"H{x + corner_radius}"
    path += f"C{x} {y + height} {x} {y + height - corner_radius} {x} {y + height - corner_radius}"
    path += f"V{y + corner_radius}"
    path += f"C{x} {y} {x + corner_radius} {y} {x + corner_radius} {y}"
    path += "Z"
    return path

def generate_rounded_bracket_path(x_top=0, y_top=0, x_bottom=0, y_bottom=0, x_vertical=0, corner_radius=0.5):
    assert y_top>y_bottom
    
    if x_top<x_vertical:
        path = f"M{x_top} {y_top}"
        path += f"H{x_vertical - corner_radius}"
        path += f"C{x_vertical} {y_top} {x_vertical} {y_top - corner_radius} {x_vertical} {y_top - corner_radius}"
        path += f"V{y_bottom + corner_radius}"
        path += f"C{x_vertical} {y_bottom} {x_vertical - corner_radius} {y_bottom} {x_vertical - corner_radius} {y_bottom}"
        path += f"H{x_bottom}"
    else:
        path = f"M{x_top} {y_top}"
        path += f"H{x_vertical + corner_radius}"
        path += f"C{x_vertical} {y_top} {x_vertical} {y_top - corner_radius} {x_vertical} {y_top - corner_radius}"
        path += f"V{y_bottom + corner_radius}"
        path += f"C{x_vertical} {y_bottom} {x_vertical + corner_radius} {y_bottom} {x_vertical + corner_radius} {y_bottom}"
        path += f"H{x_bottom}"
    # path += f"C{x} {y + height} {x} {y + height - corner_radius} {x} {y + height - corner_radius}"
    # path += f"V{y + corner_radius}"
    # path += f"C{x} {y} {x + corner_radius} {y} {x + corner_radius} {y}"
    # path += "Z"
    return path

def generate_circle_path(radius, cx, cy):
    path = ""
    angle_step = 2 * math.pi / 360  # Angle step size (1 degree)

    for angle in range(0, 361):
        x = cx + radius * math.cos(angle * angle_step)
        y = cy + radius * math.sin(angle * angle_step)
        if angle == 0:
            path += f"M{x} {y}"  # Move to the first point
        else:
            path += f"L{x} {y}"  # Line to the next point

    path += "Z"  # Close the path
    return path


def get_rect_lower_left(up_center_x,up_center_y):
    return up_center_x-comp_w/2, up_center_y-comp_h
def get_circ_center(up_center_x,up_center_y):
    return up_center_x, up_center_y-loop_r
def get_circ_lower_center(up_center_x,up_center_y):
    return up_center_x, up_center_y-2*loop_r

def get_subtree_span_h(node, all_subtrees_span_h):
    nb_comps = len(node['computations_list'])
    nb_loops = len(node['child_list'])
    if nb_loops==0:# leaf loop
        self_span = nb_comps*comp_w + abs_gap_h*(nb_comps-1)
    else:
        sum_child_spans = 0
        for child in node['child_list']:
            span, dict_ = get_subtree_span_h(child, all_subtrees_span_h)
            all_subtrees_span_h.update(all_subtrees_span_h)
            sum_child_spans+=span
        self_span = nb_comps*comp_w + sum_child_spans + abs_gap_h*(nb_comps+nb_loops-1)
    all_subtrees_span_h[node['loop_name']] = self_span
    return self_span, all_subtrees_span_h


def get_children_shapes_subroutine(node, self_x, self_y, all_subtrees_span_h, all_positions_dict):
    all_positions_dict[node['loop_name']] = (self_x, self_y)
    #compute total number of childs
    nb_comps = len(node['computations_list'])
    nb_loops = len(node['child_list'])
    #compute position of each child based on parent position
    # the computed positions are upper center, they will be converted later depending on the shape
    
    self_span = all_subtrees_span_h[node['loop_name']]
    pos_list = []
    curr_pos_x = self_x - self_span/2 #as upper left
    for loop_child in node['child_list']:
        child_span = all_subtrees_span_h[loop_child['loop_name']]
        x = curr_pos_x+child_span/2  
        y = self_y-gap_v
        curr_pos_x += (child_span + abs_gap_h)
        pos_list.append((x,y))
    for comp_child in node['computations_list']:
        child_span = comp_w
        x = curr_pos_x+child_span/2 
        y = self_y-gap_v
        curr_pos_x += (child_span + abs_gap_h)
        pos_list.append((x,y))
        all_positions_dict[comp_child] = (x,y)
        
    #get the shape_dict of each child (loops first then comps, but doesn't matter since the ds doesn't have the case of having both)
    shapes_list = []
    #draw self
    x_eff,y_eff = get_circ_center(self_x,self_y)
    shapes_list.append(dict(
                            type="path",
                            path=generate_circle_path(loop_r, x_eff, y_eff),
                            fillcolor=loop_fillcolor,
                            line_color=loop_line)
                        )
    for comp_child in node['computations_list']: #draw comp
        x,y = all_positions_dict[comp_child]
        x_eff,y_eff = get_rect_lower_left(x,y)
        shapes_list.append(dict(
                                type="path",
                                path=generate_rounded_rectangle_path(comp_h, comp_w, comp_r, x_eff, y_eff),
                                fillcolor=comp_fillcolor,
                                line_color=comp_line)
                            )
    
    #for each child loop call the func
    for i,loop_child in enumerate(node['child_list']):
        list_, dict_ = get_children_shapes_subroutine(loop_child, pos_list[i][0], pos_list[i][1], all_subtrees_span_h, all_positions_dict)
        all_positions_dict.update(dict_)
        shapes_list.extend(list_)
    
    #draw lines from self to childs
    for loop_child in node['child_list']:
        c_x,c_y = all_positions_dict[loop_child['loop_name']]
        eff_x, eff_y = get_circ_lower_center(self_x,self_y)
        shapes_list.append(dict(type="line", 
                                x0=eff_x, y0=eff_y, x1=c_x, y1=c_y, line_width=line_width, line_color=line_color)
                            )
    for comp_child in node['computations_list']:
        c_x,c_y = all_positions_dict[comp_child]
        eff_x, eff_y = get_circ_lower_center(self_x,self_y)
        shapes_list.append(dict(type="line", 
                                x0=eff_x, y0=eff_y, x1=c_x, y1=c_y, line_width=line_width, line_color=line_color)
                            )
    return shapes_list, all_positions_dict


def get_children_shapes(root_node, root_x, root_y):

    all_positions_dict = {}
    all_subtrees_span_h = {}
    get_subtree_span_h(root_node, all_subtrees_span_h)
    shapes_list, all_positions_dict = get_children_shapes_subroutine(root_node, root_x, root_y, all_subtrees_span_h, all_positions_dict)
    return shapes_list, all_positions_dict

def split_to_single_transfs(decl):
    res =[]
    for transf in [i+')' for i in decl[1].split(')')][:-1]:
        res.append((decl[0], transf))
    return res

def make_fig_hoverable(program_json, positions, source_code):
    x = []
    y = []
    annots = []
    for name,pos in positions.items():
        if name in program_json['computations']:
            x.append(pos[0]) 
            y.append(pos[1])
            expression = re.findall(name+'.set_expression\((.*)\);',source_code)
            if not expression: #ie set_expression not found
                expression = re.findall(name+'\", \{.*},(.*)\);',source_code)
            expression = expression[0]
            annots.append(go.layout.Annotation(
                x=pos[0],
                y=pos[1],
                xref="x",
                yref="y",
                text='<b>'+name+'</b>',
                align='center',
                showarrow=False,
                yanchor='top',
                hovertext = f'<b>Expression</b>: {expression}',
                # width = 50,
                # height = 3,
                # bordercolor='red',
                borderpad= 20,####################hardcoded
                yshift=-20,     ####################hardcoded               
                textangle=-90))
        else: #it's a loop
            x.append(pos[0]) 
            y.append(pos[1]) 
            annots.append(go.layout.Annotation(
                x=pos[0],
                y=pos[1],
                xref="x",
                yref="y",
                text='<b>'+name+'</b>',
                # font={'size':16},
                align='center',
                showarrow=False,
                yanchor='top',
                hovertext = f"<b>Lower Bound</b>: {program_json['iterators'][name]['lower_bound']}<br><b>Upper Bound</b>: {program_json['iterators'][name]['upper_bound']}",
                # width = 50,
                # height = 3,
                # bordercolor='red',
                borderpad= 20,####################hardcoded
                yshift=0,       ####################hardcoded              
                textangle=0))
    return annots


def get_sched_decorations(program_json,schedule_json,sched_str_new,positions):
# dict of 'l1c-1':['scheds on l1'] 
# 'l2l3c-1':['scheds on l1l3']
    orig_decls = get_orig_declarations(sched_str_new)
    merged_decls = merge_all_decls(sched_str_new, orig_decls, program_json, schedule_json)
    # print(merged_decls)
    # print(sched_str_new)
    decorations_annots = []
    decorations_shapes  = []
    per_level_dic ={}
    single_decls = []
    for decl in merged_decls:
        single_decls.extend(split_to_single_transfs(decl))
    # print(single_decls)
    for comps,transf in single_decls:
        levels = re.findall('L\d', transf)
        if len(levels)==3:
            levels = [levels[0], levels[-1]]
        levels = ','.join(levels)
        # int_levels = re.findall('\d',levels)
        if transf[0]=='U' or transf[0]=='F':
            # print(transf)
            tup = tuple((levels+','+','.join(comps)).split(','))
        else:
            last_comp = sorted(comps)[-1]
            tup = tuple((levels+','+last_comp).split(','))
        per_level_dic[tup] = per_level_dic.get(tup,[])+[transf]
    sorted_keys = sorted(list(per_level_dic.keys()), key=lambda x:len(x))
    # print(per_level_dic)
    verical_sep_x = 4 #### hardcoded
    # print(per_level_dic)
    for key in sorted_keys:
        if per_level_dic[key][0][0]=='F':
            # print(per_level_dic)
            # print(key)
            comp2 = key[-1]
            comp1 = key[-2]
            level = int(key[0][-1])
            loop_name1 = program_json['computations'][comp1]['iterators'][level]
            loop_name2 = program_json['computations'][comp2]['iterators'][level]
            pos1=positions[loop_name1]
            pos2=positions[loop_name2]
            decorations_shapes.append(dict(
                                type="path",
                                path=generate_rounded_bracket_path(pos1[0], pos1[1], pos2[0], pos2[1]-2, verical_sep_x, 1), ####################hardcoded
                                # fillcolor='blue',
                                line_color=kindofteal))####################hardcoded
            decorations_annots.append(go.layout.Annotation(
                x=verical_sep_x,####################hardcoded
                y=(pos1[1]+pos2[1])/2,####################hardcoded
                xref="x",
                yref="y",
                text=''+','.join(per_level_dic[key])+'',
                # font={'size':16},
                align='center',
                showarrow=False,
                yanchor='top',
                xanchor='left',
                font_color=kindofteal,
                # hovertext = f"<b>Lower Bound</b>: {program_json['iterators'][loop_name]['lower_bound']}<br><b>Upper Bound</b>: {program_json['iterators'][name]['upper_bound']}:",
                # width = 50,
                # height = 3,
                # bordercolor='red',
                # borderpad= 20,
                yshift=-10,     ####################hardcoded                
                xshift=0,     ####################hardcoded                
                textangle=0))
            # print('<b>'+','.join(per_level_dic[key])+'</b>')
            verical_sep_x+=2
            pass
        elif per_level_dic[key][0][0]=='U': # if unrolling, show transform on computation
                # print(per_level_dic)
                # loop_name = program_json['computations'][key[-1]]['iterators'][level]
                for comp_name in key[1:]:
                    pos=positions[comp_name]
                    decorations_annots.append(go.layout.Annotation(
                        x=pos[0],
                        y=pos[1],
                        xref="x",
                        yref="y",
                        text=''+','.join(per_level_dic[key])+'',
                        # font={'size':16},
                        align='center',
                        showarrow=False,
                        yanchor='top',
                        font_color=kindofteal,
                        # hovertext = f"<b>Lower Bound</b>: {program_json['iterators'][loop_name]['lower_bound']}<br><b>Upper Bound</b>: {program_json['iterators'][name]['upper_bound']}:",
                        # width = 50,
                        # height = 3,
                        # bordercolor='red',
                        # borderpad= 20,
                        yshift=-45,     ####################hardcoded                
                        xshift=35,     ####################hardcoded                
                        textangle=-90))
        elif len(key)==2: #single param  
            level = int(key[0][1])
            loop_name = program_json['computations'][key[-1]]['iterators'][level]
            pos=positions[loop_name]
            decorations_annots.append(go.layout.Annotation(
                x=pos[0],
                y=pos[1],
                xref="x",
                yref="y",
                text=''+','.join(per_level_dic[key])+'',
                # font={'size':16},
                align='center',
                showarrow=False,
                yanchor='top',
                xanchor='left' if verical_sep_x>0 else 'right',
                font_color=kindofteal,
                # hovertext = f"<b>Lower Bound</b>: {program_json['iterators'][loop_name]['lower_bound']}<br><b>Upper Bound</b>: {program_json['iterators'][name]['upper_bound']}:",
                # width = 50,
                # height = 3,
                # bordercolor='red',
                # borderpad= 20,
                yshift=-20,     ####################hardcoded                
                xshift=27,     ####################hardcoded                
                textangle=0))
            # print('<b>'+','.join(per_level_dic[key])+'</b>')
            # verical_sep_x+=1 ####################hardcoded
            
        else: # two or more
            # get name of firs and last loop
            level1 = int(key[0][1])
            level2 = int(key[-2][1])
            loop_name1 = program_json['computations'][key[-1]]['iterators'][level1]
            loop_name2 = program_json['computations'][key[-1]]['iterators'][level2]
            pos1=positions[loop_name1]
            pos2=positions[loop_name2]
            decorations_shapes.append(dict(
                                type="path",
                                path=generate_rounded_bracket_path(pos1[0], pos1[1], pos2[0], pos2[1]-2, verical_sep_x, 2), ####################hardcoded
                                # fillcolor='blue',
                                line_color=kindofteal))####################hardcoded
            decorations_annots.append(go.layout.Annotation(
                x=verical_sep_x,####################hardcoded
                y=(pos1[1]+pos2[1])/2,####################hardcoded
                xref="x",
                yref="y",
                text=''+','.join(per_level_dic[key])+'',
                # font={'size':16},
                align='center',
                showarrow=False,
                yanchor='top',
                xanchor='left',
                font_color=kindofteal,
                # hovertext = f"<b>Lower Bound</b>: {program_json['iterators'][loop_name]['lower_bound']}<br><b>Upper Bound</b>: {program_json['iterators'][name]['upper_bound']}:",
                # width = 50,
                # height = 3,
                # bordercolor='red',
                # borderpad= 20,
                yshift=-10,     ####################hardcoded                
                xshift=0,     ####################hardcoded                
                textangle=0))
            # print('<b>'+','.join(per_level_dic[key])+'</b>')
            verical_sep_x+=2 ####################hardcoded
            verical_sep_x*=-1 ####################hardcoded
            # draw bracket (gen a shape)
            # add textas annot
    return decorations_annots, decorations_shapes

comp_w = 2
comp_h = 5
comp_r = 0.5
loop_r = 1
abs_gap_h = 1 #the gap between the childs borders
# gap_h = max([comp_w, loop_r*2])+abs_gap_h #because it's considering upper center coords
gap_v = loop_r*2+1 #because it's considering upper center coords
comp_fillcolor = "#99057B"
comp_line = "#6B335F"
loop_fillcolor = "#fc6a03"
loop_line = "#d16002"
line_color = "#b8bdc3"
line_width = 2

kindofteal = '#38d9a9'




##############################
###### sched contrib



def merge_decls(decl1,decl2, program_json, fusions):
    comps1 = sorted(decl1[0], key= lambda x: program_json["computations"][x]["absolute_order"])
    comps2 = sorted(decl2[0], key=lambda x: program_json["computations"][x]["absolute_order"])
    sched1 = decl1[1]
    sched2 = decl2[1]
    
    deepest_lvl = get_deepest_shared_loop(comps1[-1],comps2[-1],program_json,fusions)
    
    index_last_par = 0
    
    for i in range(min(len(sched1),len(sched2))):
        c1 = sched1[i]
        c2 = sched2[i]
        if c1!=c2:
            break
        elif c1==')':
            index_last_par = i
        elif c1=='L':
            if int(sched1[i+1])>deepest_lvl or int(sched2[i+1])>deepest_lvl:
                break
    merge_sched = sched1[:index_last_par]
    if merge_sched == '':
        leftover1 = decl1
        leftover2 = decl2
        merge_res = ([],'')
        return merge_res, leftover1, leftover2
    else:
        merge_sched+=')'
        merge_comps = comps1+comps2
        leftover1_sched = sched1[len(merge_sched):]   
        leftover2_sched = sched2[len(merge_sched):]
        leftover1_comps = comps1 if leftover1_sched else []
        leftover2_comps = comps2 if leftover2_sched else []
        merge_res = (merge_comps,merge_sched)
        leftover1 = (leftover1_comps,leftover1_sched)
        leftover2 = (leftover2_comps,leftover2_sched)
        return merge_res, leftover1, leftover2

def get_orig_declarations(schedule_str_new):
    delarations_str =[ i for i in schedule_str_new.split('{')][1:]
    declarations = []
    for decl_str in delarations_str:
        comps = decl_str[:decl_str.index('}')].split(',')
        sched = decl_str[decl_str.index(':')+1:]
        declarations.append((comps,sched))
    return declarations

    
def merge_all_decls(schedule_str_new, orig_declarations, program_json, schedule_json):
    declarations = orig_declarations[:]
    fusions = schedule_json['fusions']
    has_changed = True
    while has_changed:
        has_changed = False
        for i in range(len(declarations)):
            decl1 = declarations[i]
            for j in range(i+1,len(declarations)):
                decl2 = declarations[j]
                # print(decl1,'??????111????????',decl2)
                merge_res, leftover1, leftover2 = merge_decls(decl1,decl2, program_json,fusions)
                # print(merge_res[1]!='')
                if merge_res[1]!='': #merge possible
                    # print(declarations)
                    # print(decl1,'??????222????????',decl2)

                    declarations.remove(decl1)
                    # print(declarations,'---------',decl2)

                    declarations.remove(decl2)
                    if leftover1[1]!='':
                        declarations.append(leftover1)
                    if leftover2[1]!='':
                        declarations.append(leftover2)
                    declarations.append(merge_res)
                    has_changed = True
                    break # j loop
            if has_changed:
                break # i loop
    return declarations
    
def get_deepest_shared_loop(comp1,comp2, program_json,fusions): #originaly designed for any number of comps, but tricked it two to ease the fusion handling #before schedules, returns depth, assumes each loop has different name
    if fusions:#if fusion is enabled, there can be only one fusion
        if (fusions[0][0]==comp1 and fusions[0][1]==comp2) or (fusions[0][1]==comp1 and fusions[0][0]==comp2):
            return fusions[0][2] # if comps fused, just return the fusion level cuz it can't be deeper than that
    comps_list= [comp1,comp2]
    nests_list = []
    for comp in comps_list:
        nests_list.append(program_json['computations'][comp]['iterators'])
    min_len = min([len(i) for i in nests_list])
    # print(nests_list)
    for i in range(min_len):
        for j in range(len(comps_list)-1):
            # print(nests_list[j][i],nests_list[j+1][i])
            if nests_list[j][i]!=nests_list[j+1][i]:
                return i-1
        
    return min_len-1

def split_to_single_transfs(decl):
    res =[]
    for transf in [i+')' for i in decl[1].split(')')][:-1]:
        res.append((decl[0], transf))
    return res

def decl_to_decl_str(decl):
    if decl[1]=='':
        return ''
    return ('{'+','.join(decl[0])+'}:'+''.join(decl[1]))

def get_indev_sched_contribs(model, programs_dict, function_name, schedule_index):
        
    program_json = programs_dict[function_name]['program_annotation']
    schedule_json = programs_dict[function_name]['schedules_list'][schedule_index]
    
    sched_str_new = get_schedule_str_new(program_json, schedule_json)
    orig_decls = get_orig_declarations(sched_str_new)
    merged_decls = merge_all_decls(sched_str_new, orig_decls, program_json, schedule_json)
    
    prog_tree, comps_repr_templates_list, loops_repr_templates_list, comps_placeholders_indices_dict, loops_placeholders_indices_dict = get_representation_template(programs_dict[function_name], max_depth = 5)

    comps_tensor, loops_tensor = get_schedule_representation(program_json, schedule_json, comps_repr_templates_list, loops_repr_templates_list, comps_placeholders_indices_dict, loops_placeholders_indices_dict, 5)
    # prog_tree, comps_tensor, loops_tensor = get_dp_representation(programs_dict, function_name, schedule_index)
    orig_pred = float(model((prog_tree, comps_tensor, loops_tensor)))

    # print('hola ',len(merged_decls), merged_decls, sched_str_new)
    if schedule_index==0: #Base schedule, no transformatins applied 
        return {'No Transformation': 1}, orig_pred
    
    # raw_changes_dic = {} #dict of disabled_transf('{comp,comp..}:transf'):change in pred (orig - new)
    alt_preds_dict = {} #dict of disabled_transf('{comp,comp..}:transf'):new pred after disabling
    
    #assume that all affine are gathered in one decl
    
    for decl in merged_decls:
        split_decls = split_to_single_transfs(decl)
        for i,s_decl in enumerate(split_decls):
            transf_to_disable = s_decl[1]
            affected_comps = s_decl[0]

            if transf_to_disable[0] in ['I', 'R', 'S']: # affine transformations
                altered_sched_json = disable_affine_transf(schedule_json, affected_comps, i)
            else:
                altered_sched_json = disable_non_affine_transf(schedule_json, affected_comps, transf_to_disable)

            alt_comps_tensor, alt_loops_tensor = get_schedule_representation(program_json, altered_sched_json, comps_repr_templates_list, loops_repr_templates_list, comps_placeholders_indices_dict, loops_placeholders_indices_dict, 5)
            alt_pred = float(model((prog_tree, alt_comps_tensor, alt_loops_tensor)))
            
            s_decl_str = decl_to_decl_str(s_decl)
            alt_preds_dict[s_decl_str] = alt_pred
            
    ratio_contrib_dict = {s_decl_str:orig_pred/alt_preds_dict[s_decl_str] for s_decl_str in alt_preds_dict} #dict of disabled_transf('{comp,comp..}:transf'):orig_pred/new_pred   
    
    normalization_factor = (orig_pred/math.prod([ratio_contrib_dict[s] for s in ratio_contrib_dict]))**(1/len(ratio_contrib_dict))
    
    normalized_changes_dict = {s:ratio_contrib_dict[s]*normalization_factor for s in ratio_contrib_dict}
    
    return normalized_changes_dict,orig_pred


def disable_non_affine_transf(sched_json, comps_involved, transf_to_disable):
    res_sched_json = copy.deepcopy(sched_json)
    if transf_to_disable[0]=='F':
        res_sched_json['fusions'] = None
    elif transf_to_disable[0]=='P':
        for comp in comps_involved:
            res_sched_json[comp]['parallelized_dim'] = None
    elif transf_to_disable[0]=='T':
        for comp in comps_involved:
            res_sched_json[comp]['tiling'] = None
    elif transf_to_disable[0]=='U':
        for comp in comps_involved:
            res_sched_json[comp]['unrolling_factor'] = None
    else:
        print('error')
        return 'error in disable_non_affine_transf'
    return res_sched_json


def disable_affine_transf(sched_json, comps_involved, transf_index):
    res_sched_json = copy.deepcopy(sched_json)
    for comp in comps_involved:
        comp_dict = res_sched_json[comp]
        comp_dict['transformation_matrices'].pop(transf_index+1)#+1 because identity is always there
        mats_dim = int(math.sqrt(len(comp_dict['transformation_matrix'])))
        new_final_mat = np.identity(mats_dim)
        for mat in comp_dict['transformation_matrices'][::-1]:
            # print(np.array(mat).reshape(mats_dim,mats_dim))
            new_final_mat = new_final_mat@(np.array(mat).reshape(mats_dim,mats_dim).astype(int))
        comp_dict['transformation_matrix'] = [str(i) for i in new_final_mat.flatten().astype(int).tolist()]
    return res_sched_json
