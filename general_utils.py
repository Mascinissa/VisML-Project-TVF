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
    
    
train_device = torch.device('cpu')

def get_tree_footprint(tree):
    footprint='<L'+str(int(tree['loop_index']))+'>'
    if tree['has_comps']:
        footprint+='['
        for idx in tree['computations_indices']:
            footprint+='C'+str(int(idx))
        footprint+=']'
    for child in tree['child_list']:
        footprint+= get_tree_footprint(child)
    footprint+='</L'+str(int(tree['loop_index']))+'>'
    return footprint
class Model_Recursive_LSTM_v2(nn.Module):
    def __init__(self, input_size, comp_embed_layer_sizes=[600, 350, 200, 180], drops=[0.225, 0.225, 0.225, 0.225], output_size=1):
        super().__init__()
        embedding_size = comp_embed_layer_sizes[-1]
        regression_layer_sizes = [embedding_size] + comp_embed_layer_sizes[-2:]
        concat_layer_sizes = [embedding_size*2+20] + comp_embed_layer_sizes[-2:]
        comp_embed_layer_sizes = [input_size] + comp_embed_layer_sizes
        self.comp_embedding_layers = nn.ModuleList()
        self.comp_embedding_dropouts= nn.ModuleList()
        self.regression_layers = nn.ModuleList()
        self.regression_dropouts= nn.ModuleList()
        self.concat_layers = nn.ModuleList()
        self.concat_dropouts= nn.ModuleList()
        for i in range(len(comp_embed_layer_sizes)-1):
            self.comp_embedding_layers.append(nn.Linear(comp_embed_layer_sizes[i], comp_embed_layer_sizes[i+1], bias=True))
#             nn.init.xavier_uniform_(self.comp_embedding_layers[i].weight)
            nn.init.zeros_(self.comp_embedding_layers[i].weight)
            self.comp_embedding_dropouts.append(nn.Dropout(drops[i]))
        for i in range(len(regression_layer_sizes)-1):
            self.regression_layers.append(nn.Linear(regression_layer_sizes[i], regression_layer_sizes[i+1], bias=True))
#             nn.init.xavier_uniform_(self.regression_layers[i].weight)
            nn.init.zeros_(self.regression_layers[i].weight)
            self.regression_dropouts.append(nn.Dropout(drops[i]))
        for i in range(len(concat_layer_sizes)-1):
            self.concat_layers.append(nn.Linear(concat_layer_sizes[i], concat_layer_sizes[i+1], bias=True))
#             nn.init.xavier_uniform_(self.concat_layers[i].weight)
            nn.init.zeros_(self.concat_layers[i].weight)
            self.concat_dropouts.append(nn.Dropout(drops[i]))
        self.predict = nn.Linear(regression_layer_sizes[-1], output_size, bias=True)
#         nn.init.xavier_uniform_(self.predict.weight)
        nn.init.zeros_(self.predict.weight)
        self.ELU=nn.ELU()
        self.no_comps_tensor = nn.Parameter(nn.init.xavier_uniform_(torch.zeros(1, embedding_size)))
        self.no_nodes_tensor = nn.Parameter(nn.init.xavier_uniform_(torch.zeros(1, embedding_size)))
        self.comps_lstm = nn.LSTM(comp_embed_layer_sizes[-1], embedding_size, batch_first=True)
        self.nodes_lstm = nn.LSTM(comp_embed_layer_sizes[-1], embedding_size, batch_first=True)
        
    def get_hidden_state(self, node, comps_embeddings, loops_tensor):
        nodes_list = []
        for n in node['child_list']:
            nodes_list.append(self.get_hidden_state(n, comps_embeddings,loops_tensor))
        if (nodes_list != []):
            nodes_tensor = torch.cat(nodes_list, 1) 
            lstm_out, (nodes_h_n, nodes_c_n) = self.nodes_lstm(nodes_tensor)
            nodes_h_n = nodes_h_n.permute(1,0,2)
        else:       
            nodes_h_n = torch.unsqueeze(self.no_nodes_tensor, 0).expand(comps_embeddings.shape[0], -1, -1)
        if (node['has_comps']):
            selected_comps_tensor = torch.index_select(comps_embeddings, 1, node['computations_indices'])
            lstm_out, (comps_h_n, comps_c_n) = self.comps_lstm(selected_comps_tensor) 
            comps_h_n = comps_h_n.permute(1,0,2)
        else:
            comps_h_n = torch.unsqueeze(self.no_comps_tensor, 0).expand(comps_embeddings.shape[0], -1, -1)
        selected_loop_tensor = torch.index_select(loops_tensor,1,node['loop_index'])
        x = torch.cat((nodes_h_n, comps_h_n, selected_loop_tensor),2)
        for i in range(len(self.concat_layers)):
            x = self.concat_layers[i](x)
            x = self.concat_dropouts[i](self.ELU(x))
        return x  

    def forward(self, tree_tensors):
        tree, comps_tensor, loops_tensor = tree_tensors
        #computation embbedding layer
        x = comps_tensor
        for i in range(len(self.comp_embedding_layers)):
            x = self.comp_embedding_layers[i](x)
            x = self.comp_embedding_dropouts[i](self.ELU(x))  
        comps_embeddings = x
        #recursive loop embbeding layer
        prog_embedding = self.get_hidden_state(tree, comps_embeddings, loops_tensor)
        #regression layer
        x = prog_embedding
        for i in range(len(self.regression_layers)):
            x = self.regression_layers[i](x)
            x = self.regression_dropouts[i](self.ELU(x))
        out = self.predict(x)
            
        return self.ELU(out[:,0,0])
    
    def get_last_embed(self, tree_tensors):
        tree, comps_tensor, loops_tensor = tree_tensors
        #computation embbedding layer
        x = comps_tensor
        for i in range(len(self.comp_embedding_layers)):
            x = self.comp_embedding_layers[i](x)
            x = self.comp_embedding_dropouts[i](self.ELU(x))  
        comps_embeddings = x
        #recursive loop embbeding layer
        prog_embedding = self.get_hidden_state(tree, comps_embeddings, loops_tensor)
        #regression layer
        x = prog_embedding
        for i in range(len(self.regression_layers)):
            x = self.regression_layers[i](x)
            x = self.regression_dropouts[i](self.ELU(x))
        
            
        return x

    
    
    
    
    
    
    
def get_representation_template(program_dict, max_depth):
    max_accesses = 15
    min_accesses = 1
#     max_depth = 5 
    
    comps_repr_templates_list = []
    comps_indices_dict = dict()
    comps_placeholders_indices_dict = dict()
    
    program_json = program_dict['program_annotation']
    computations_dict = program_json['computations']
    ordered_comp_list = sorted(list(computations_dict.keys()), key = lambda x: computations_dict[x]['absolute_order'])
    
    for comp_index, comp_name in enumerate(ordered_comp_list):
        comp_dict = computations_dict[comp_name]
        if len(comp_dict['accesses'])>max_accesses:
            raise NbAccessException
        if len(comp_dict['accesses'])<min_accesses:
            raise NbAccessException
        if len(comp_dict['iterators'])>max_depth:
            raise LoopsDepthException
            
        comp_repr_template = []
        # Is this computation a reduction 
        comp_repr_template.append(+comp_dict['comp_is_reduction'])


#         iterators representation + tiling and interchage
        iterators_repr = []
        for iter_i,iterator_name in enumerate(comp_dict['iterators']):
            iterator_dict = program_json['iterators'][iterator_name]
            iterators_repr.extend([iterator_dict['lower_bound'], iterator_dict['upper_bound']])
            
            # transformations placeholders
            c_code = 'C'+str(comp_index)
            l_code= c_code+'-L'+str(iter_i)
            iterators_repr.extend([l_code+'Parallelized',
                                   l_code+'Tiled', l_code+'TileFactor',
                                   l_code+'Fused']) #unrolling is skipped since it is only applied on innermost loop

        # Adding padding
        iterator_repr_size = int(len(iterators_repr)/len(comp_dict['iterators']))
        iterators_repr.extend([0]*iterator_repr_size*(max_depth-len(comp_dict['iterators']))) # adding iterators padding 

        # Adding unrolling placeholder since unrolling can only be applied to the innermost loop 
        iterators_repr.extend([c_code+'-Unrolled', c_code+'-UnrollFactor'])
        
        # Adding transformation matrix place holder
        iterators_repr.append(c_code+'-TransformationMatrixStart')
        iterators_repr.extend(['M']*((max_depth+1)**2-2))
        iterators_repr.append(c_code+'-TransformationMatrixEnd')
    
        # Adding the iterators representation to computation vector
        comp_repr_template.extend(iterators_repr)     

        #  Write access representation to computation vector
        padded_write_matrix = pad_access_matrix(isl_to_write_matrix(comp_dict['write_access_relation']), max_depth)
        write_access_repr = [comp_dict['write_buffer_id']+1] + padded_write_matrix.flatten().tolist() # buffer_id + flattened access matrix 
        
        # Adding write access representation to computation vector
        comp_repr_template.extend(write_access_repr)

        # Read Access representation 
        read_accesses_repr=[]
        for read_access_dict in comp_dict['accesses']:
            read_access_matrix = pad_access_matrix(read_access_dict['access_matrix'], max_depth)
            read_access_repr = [+read_access_dict['access_is_reduction']]+ [read_access_dict['buffer_id']+1] + read_access_matrix.flatten().tolist() # buffer_id + flattened access matrix 
            read_accesses_repr.extend(read_access_repr)

        access_repr_len = (max_depth+1)*(max_depth + 2) + 1 +1 # access matrix size +1 for buffer id +1 for is_access_reduction
        read_accesses_repr.extend([0]*access_repr_len*(max_accesses-len(comp_dict['accesses']))) #adding accesses padding

    
        comp_repr_template.extend(read_accesses_repr)

        # Adding Operations count to computation vector
        comp_repr_template.append(comp_dict['number_of_additions'])
        comp_repr_template.append(comp_dict['number_of_subtraction'])
        comp_repr_template.append(comp_dict['number_of_multiplication'])
        comp_repr_template.append(comp_dict['number_of_division'])
        

        # adding log(x+1) of the representation
#         log_rep = list(np.log1p(comp_representation))
#         comp_representation.extend(log_rep)
        
        comps_repr_templates_list.append(comp_repr_template)
        comps_indices_dict[comp_name] = comp_index
        for j, element in enumerate(comp_repr_template):
            if isinstance(element, str):
                comps_placeholders_indices_dict[element] = (comp_index,j)
    

        
    #building loop representation template
    
    loops_repr_templates_list = []
    loops_indices_dict = dict()
    loops_placeholders_indices_dict = dict()
#     assert len(program_json['iterators'])==len(set(program_json['iterators'])) #just to make sure that loop names are not duplicates, but this can't happen because it's a dict
    for loop_index, loop_name in enumerate(program_json['iterators']): # !! is the order in this list fix? can't we get new indices during schedule repr !!! should we use loop name in plchldrs instead of index ? !! #Edit: now it's using the name, so this issue shouldn't occure
        loop_repr_template=[]
        l_code = 'L'+loop_name
        # upper and lower bound
        loop_repr_template.extend([program_json['iterators'][loop_name]['lower_bound'],program_json['iterators'][loop_name]['upper_bound']])   
        loop_repr_template.extend([l_code+'Parallelized',
                                   l_code+'Tiled', l_code+'TileFactor',
                                   l_code+'Fused',
                                   l_code+'Unrolled', l_code+'UnrollFactor'])
        loop_repr_template.extend([l_code+'TransfMatRowStart']+['M']*(max_depth-2+1)+[l_code+'TransfMatRowEnd']) #+1 for the frame
        loop_repr_template.extend([l_code+'TransfMatColStart']+['M']*(max_depth-2+1)+[l_code+'TransfMatColEnd'])
        # adding log(x+1) of the loop representation
        loops_repr_templates_list.append(loop_repr_template)    
        loops_indices_dict[loop_name]=loop_index
        
        for j, element in enumerate(loop_repr_template):
            if isinstance(element, str):
                loops_placeholders_indices_dict[element] = (loop_index,j)
    
            
     
    def update_tree_atributes(node):     
        node['loop_index'] = torch.tensor(loops_indices_dict[node['loop_name']]).to(train_device)
        if node['computations_list']!=[]:
            node['computations_indices'] = torch.tensor([comps_indices_dict[comp_name] for comp_name in node['computations_list']]).to(train_device)
            node['has_comps'] = True
        else:
            node['has_comps'] = False
        for child_node in node['child_list']:
            update_tree_atributes(child_node)
        return node
    
    # getting the original tree structure 
    no_sched_json = program_dict['schedules_list'][0]
    assert 'fusions' not in no_sched_json or no_sched_json['fusions']==None
    orig_tree_structure = no_sched_json['tree_structure']
    tree_annotation = copy.deepcopy(orig_tree_structure) #to avoid altering the original tree from the json
    prog_tree = update_tree_atributes(tree_annotation) 
    
#     loops_tensor = torch.unsqueeze(torch.FloatTensor(loops_repr_templates_list),0)#.to(device)
#     computations_tensor = torch.unsqueeze(torch.FloatTensor(comps_repr_templates_list),0)#.to(device)     

    return prog_tree, comps_repr_templates_list, loops_repr_templates_list, comps_placeholders_indices_dict, loops_placeholders_indices_dict


def get_schedule_representation(program_json, schedule_json, comps_repr_templates_list, loops_repr_templates_list, comps_placeholders_indices_dict, loops_placeholders_indices_dict, max_depth):

    comps_repr = copy.deepcopy(comps_repr_templates_list)
    loops_repr = copy.deepcopy(loops_repr_templates_list)
    
    computations_dict = program_json['computations']
    ordered_comp_list = sorted(list(computations_dict.keys()), key = lambda x: computations_dict[x]['absolute_order'])
    
    padded_tranf_mat_per_comp = dict()
    
    for comp_index, comp_name in enumerate(ordered_comp_list):
        comp_dict =  program_json['computations'][comp_name]
        comp_schedule_dict=schedule_json[comp_name]
        c_code = 'C'+str(comp_index)
        
        
        #Fusion representation
        fused_levels = []
        if 'fusions' in schedule_json and schedule_json['fusions']:
            for fusion in schedule_json['fusions']:#check if comp is involved in fusions 
                 # fusions format [compname1, compname2, loop depth]
                if comp_name in fusion:
                    fused_levels.append(fusion[2])
                
            
        for iter_i,iterator_name in enumerate(comp_dict['iterators']):
            
            ### Updating the computations representation template 
            l_code= c_code+'-L'+str(iter_i)
            
             # Parallelization representation
            parallelized = 0
            if iterator_name == comp_schedule_dict['parallelized_dim']:
                parallelized = 1 # parallelized true
            p_index = comps_placeholders_indices_dict[l_code+'Parallelized']
            comps_repr[p_index[0]][p_index[1]]=parallelized
            
            # Tiling representation 
            tiled = 0
            tile_factor = 0
            if comp_schedule_dict['tiling'] and (iterator_name in comp_schedule_dict['tiling']['tiling_dims']):
                tiled = 1 #tiled: true
                tile_factor_index = comp_schedule_dict['tiling']['tiling_dims'].index(iterator_name)
                tile_factor = int(comp_schedule_dict['tiling']['tiling_factors'][tile_factor_index]) #tile factor
            p_index = comps_placeholders_indices_dict[l_code+'Tiled']
            comps_repr[p_index[0]][p_index[1]] = tiled
            p_index = comps_placeholders_indices_dict[l_code+'TileFactor']
            comps_repr[p_index[0]][p_index[1]] = tile_factor
            
            # Fusion representation
            fused = 0
            if iter_i in fused_levels:
                fused=1
            p_index = comps_placeholders_indices_dict[l_code+'Fused']
            comps_repr[p_index[0]][p_index[1]] = fused
            

         # Unrolling Representation 
        unrolled = 0
        unroll_factor = 0
        if comp_schedule_dict['unrolling_factor']: #Unrolling is always aplied to the innermost loop 
            unrolled=1 #unrolled True
            unroll_factor = int(comp_schedule_dict['unrolling_factor']) #unroll factor
        p_index = comps_placeholders_indices_dict[c_code+'-Unrolled']
        comps_repr[p_index[0]][p_index[1]] = unrolled
        p_index = comps_placeholders_indices_dict[c_code+'-UnrollFactor']
        comps_repr[p_index[0]][p_index[1]] = unroll_factor
        
        # Adding the transformation matrix
        # get the matrix start and end indices 
        mat_start = comps_placeholders_indices_dict[c_code+'-TransformationMatrixStart']
        mat_end = comps_placeholders_indices_dict[c_code+'-TransformationMatrixEnd']
        nb_mat_elements = mat_end[1] - mat_start[1] + 1
        max_depth = int(np.sqrt(nb_mat_elements))-1 # temporarily hack to get max_depth to use it in padding
        padded_matrix = get_padded_transformation_matrix(program_json, schedule_json, comp_name, max_depth)
    #     print(nb_mat_elements, padded_matrix, max_depth)
        assert len(padded_matrix.flatten().tolist()) == nb_mat_elements
    #     print(nb_mat_elements)
        comps_repr[mat_start[0]][mat_start[1]:mat_end[1]+1] = padded_matrix.flatten().tolist() 
        
        padded_tranf_mat_per_comp[comp_name] = padded_matrix #saving it for later to be used in loop repr
        
#     # transforming the schedule_json in order to have loops as key instead of computations, this dict helps building the loop vectors
    loop_schedules_dict = dict()
    for loop_name in program_json['iterators']:
        loop_schedules_dict[loop_name]=dict()
        loop_schedules_dict[loop_name]['TransformationMatrixCol']=[]
        loop_schedules_dict[loop_name]['TransformationMatrixRow']=[]
        loop_schedules_dict[loop_name]['tiled']=0
        loop_schedules_dict[loop_name]['tile_factor']=0
        loop_schedules_dict[loop_name]['unrolled']=0
        loop_schedules_dict[loop_name]['unroll_factor']=0
        loop_schedules_dict[loop_name]['parallelized']=0
        loop_schedules_dict[loop_name]['fused']=0
        
    for comp_index, comp_name in enumerate(ordered_comp_list):
        comp_schedule_dict = schedule_json[comp_name]
        if comp_schedule_dict['tiling']:
            for tiled_loop_index,tiled_loop in enumerate(comp_schedule_dict['tiling']['tiling_dims']):
                loop_schedules_dict[tiled_loop]['tiled']=1
                assert loop_schedules_dict[tiled_loop]['tile_factor']==0 or loop_schedules_dict[tiled_loop]['tile_factor']==int(comp_schedule_dict['tiling']['tiling_factors'][tiled_loop_index]) #just checking that it hasn't been updated with a different value
                loop_schedules_dict[tiled_loop]['tile_factor']=int(comp_schedule_dict['tiling']['tiling_factors'][tiled_loop_index])
        if comp_schedule_dict['unrolling_factor']:
            comp_innermost_loop=computations_dict[comp_name]['iterators'][-1] 
            loop_schedules_dict[comp_innermost_loop]['unrolled']=1
            assert loop_schedules_dict[comp_innermost_loop]['unroll_factor']==0 or loop_schedules_dict[comp_innermost_loop]['unroll_factor']==int(comp_schedule_dict['unrolling_factor'])  #just checking that it hasn't been updated with a different value
            loop_schedules_dict[comp_innermost_loop]['unroll_factor']=int(comp_schedule_dict['unrolling_factor'])
        if comp_schedule_dict['parallelized_dim']:
            loop_schedules_dict[comp_schedule_dict['parallelized_dim']]['parallelized'] = 1
        
        # get the rows and cols transformation matrix for each iterator
        assert padded_tranf_mat_per_comp[comp_name].shape == (max_depth+1,max_depth+1) # make sure that the padding frame is applied, otherwise need to remove the +1 from iter_i+1 in the next few lines 
        for iter_i, loop_name in enumerate(computations_dict[comp_name]['iterators']):
            if len(loop_schedules_dict[loop_name]['TransformationMatrixCol'])>0:#if not empty
                assert (loop_schedules_dict[loop_name]['TransformationMatrixCol'] == padded_tranf_mat_per_comp[comp_name][:,iter_i+1]).all() #chck if the iterator what affected by a different matrix, that shouldn't happen
            else:
                loop_schedules_dict[loop_name]['TransformationMatrixCol'] = padded_tranf_mat_per_comp[comp_name][:,iter_i+1] #+1 for the padding frame
            if len(loop_schedules_dict[loop_name]['TransformationMatrixRow'])>0:#if not empty
                assert (loop_schedules_dict[loop_name]['TransformationMatrixRow'] == padded_tranf_mat_per_comp[comp_name][iter_i+1,:]).all() #chck if the iterator what affected by a different matrix, that shouldn't happen
            else:
                loop_schedules_dict[loop_name]['TransformationMatrixRow'] = padded_tranf_mat_per_comp[comp_name][iter_i+1,:]#+1 for the padding frame
    
    #update the fusions in loops dict 
    if 'fusions' in schedule_json and schedule_json['fusions']:
        for fusion in schedule_json['fusions']:
            fused_loop1 = computations_dict[fusion[0]]['iterators'][fusion[2]]
            fused_loop2 = computations_dict[fusion[1]]['iterators'][fusion[2]]
            loop_schedules_dict[fused_loop1]['fused']=1
            loop_schedules_dict[fused_loop2]['fused']=1
        
# Updating the loop representation templates
    for loop_name in program_json['iterators']:
        l_code = 'L'+loop_name
        
        p_index = loops_placeholders_indices_dict[l_code+'Parallelized']
        loops_repr[p_index[0]][p_index[1]] = loop_schedules_dict[loop_name]['parallelized']
        
        p_index = loops_placeholders_indices_dict[l_code+'Tiled']
        loops_repr[p_index[0]][p_index[1]] = loop_schedules_dict[loop_name]['tiled']
        p_index = loops_placeholders_indices_dict[l_code+'TileFactor']
        loops_repr[p_index[0]][p_index[1]] = loop_schedules_dict[loop_name]['tile_factor']
        
        p_index = loops_placeholders_indices_dict[l_code+'Unrolled']
        loops_repr[p_index[0]][p_index[1]] = loop_schedules_dict[loop_name]['unrolled']
        p_index = loops_placeholders_indices_dict[l_code+'UnrollFactor']
        loops_repr[p_index[0]][p_index[1]] = loop_schedules_dict[loop_name]['unroll_factor']
        
        p_index = loops_placeholders_indices_dict[l_code+'Fused']
        loops_repr[p_index[0]][p_index[1]] = loop_schedules_dict[loop_name]['fused']
        
        row_start = loops_placeholders_indices_dict[l_code+'TransfMatRowStart']
        row_end = loops_placeholders_indices_dict[l_code+'TransfMatRowEnd']
        nb_row_elements = row_end[1] - row_start[1] + 1
        assert len(loop_schedules_dict[loop_name]['TransformationMatrixRow']) == nb_row_elements
        loops_repr[row_start[0]][row_start[1]:row_end[1]+1] = loop_schedules_dict[loop_name]['TransformationMatrixRow']
        
        col_start = loops_placeholders_indices_dict[l_code+'TransfMatColStart']
        col_end = loops_placeholders_indices_dict[l_code+'TransfMatColEnd']
        nb_col_elements = col_end[1] - col_start[1] + 1
        assert len(loop_schedules_dict[loop_name]['TransformationMatrixCol']) == nb_col_elements
        loops_repr[col_start[0]][col_start[1]:col_end[1]+1] = loop_schedules_dict[loop_name]['TransformationMatrixCol']
    
    loops_tensor = torch.unsqueeze(torch.FloatTensor(loops_repr),0)#.to(device)
    computations_tensor = torch.unsqueeze(torch.FloatTensor(comps_repr),0)#.to(device)     

    return computations_tensor, loops_tensor


global_dioph_sols_dict = dict()
def get_padded_transformation_matrix(program_json, schedule_json, comp_name, max_depth=None):
    comp_name = list(program_json['computations'].keys())[0] # for single comp programs, there is only one computation
    comp_dict =  program_json['computations'][comp_name]
    comp_schedule_dict=schedule_json[comp_name]
    nb_iterators = len(comp_dict['iterators'])
    loop_nest = comp_dict['iterators'][:]
    
    if 'transformation_matrix' in comp_schedule_dict: # if the program is explored using matrices
        if comp_schedule_dict['transformation_matrix']!=[]: #if matrix applied, else set it to identity
            assert np.sqrt(len(comp_schedule_dict['transformation_matrix']))==nb_iterators
            final_mat = np.array(list(map(int,comp_schedule_dict['transformation_matrix']))).reshape(nb_iterators,nb_iterators)
        else:
            final_mat = np.zeros((nb_iterators,nb_iterators),int)
            np.fill_diagonal(final_mat,1)
        # just for checking
        comparison_matrix = np.zeros((nb_iterators,nb_iterators),int)
        np.fill_diagonal(comparison_matrix,1)
        for mat in comp_schedule_dict['transformation_matrices'][::-1]:
            comparison_matrix = comparison_matrix@np.array(list(map(int,mat))).reshape(nb_iterators,nb_iterators)
        assert (comparison_matrix==final_mat).all()
    else: # if the program is explored using tags
        interchange_matrix = np.zeros((nb_iterators,nb_iterators),int)
        np.fill_diagonal(interchange_matrix,1)
        if comp_schedule_dict['interchange_dims']:
            first_iter_index = loop_nest.index(comp_schedule_dict['interchange_dims'][0])
            second_iter_index = loop_nest.index(comp_schedule_dict['interchange_dims'][1])
            interchange_matrix[first_iter_index,first_iter_index]=0 #zeroing the diagonal elements
            interchange_matrix[second_iter_index,second_iter_index]=0 #zeroing the diagonal elements
            interchange_matrix[first_iter_index, second_iter_index]=1
            interchange_matrix[second_iter_index, first_iter_index]=1
            loop_nest[first_iter_index], loop_nest[second_iter_index] = loop_nest[second_iter_index], loop_nest[first_iter_index] # swapping iterators in loop nest

        skewing_matrix = np.zeros((nb_iterators,nb_iterators),int)
        np.fill_diagonal(skewing_matrix,1)
        if comp_schedule_dict['skewing']:
            first_iter_index = loop_nest.index(comp_schedule_dict['skewing']['skewed_dims'][0])
            second_iter_index = loop_nest.index(comp_schedule_dict['skewing']['skewed_dims'][1])
            first_factor = int(comp_schedule_dict['skewing']['skewing_factors'][0])
            second_factor = int(comp_schedule_dict['skewing']['skewing_factors'][1])
            # the skewing sub matrix should be in the form of 
            # [[fact1, fact2],
            #  [a,   , b    ]]
            # and we need to find a and b to make to matix det==1
    #         a, b = symbols('a b')
    #         sol = diophantine(first_factor*b - second_factor*a - 1) # solve the diophantine equation to keep a determinant of 1 in the matrix, 
    #         a, b = list(sol)[0] # since we know that there should at least (or only?) one solution 
    #         free_symbol = list(a.free_symbols)[0] # since we know that there should be only one free symbol
    #         a = int(a.subs({free_symbol:0})) #substitue the free symbol with 0 to get the initial solution
    #         b = int(b.subs({free_symbol:0}))
#             sol = simple_linear_diophantine_r(first_factor,second_factor)
            if (first_factor,second_factor) in global_dioph_sols_dict:
                a, b = global_dioph_sols_dict[(first_factor,second_factor)]
            else: 
                a, b = linear_diophantine_default(first_factor,second_factor)
            skewing_matrix[first_iter_index,first_iter_index] = first_factor # update the matrix
            skewing_matrix[first_iter_index,second_iter_index] = second_factor
            skewing_matrix[second_iter_index,first_iter_index] = a
            skewing_matrix[second_iter_index,second_iter_index] = b

        #multiply the mats 
        final_mat = skewing_matrix@interchange_matrix # Right order is skew_mat * interchange_mat
    
    padded_mat = final_mat
    
    
    #pad matrix if max_depth defined
    if max_depth!=None:
        padded_mat = np.c_[np.ones(padded_mat.shape[0]), padded_mat] # adding tags for marking the used rows
        padded_mat = np.r_[[np.ones(padded_mat.shape[1])], padded_mat] # adding tags for marking the used columns
        padded_mat = np.pad(padded_mat, [(0,max_depth-nb_iterators),(0,max_depth-nb_iterators)], mode='constant', constant_values=0)
    
    return padded_mat



    
    
def get_datapoint_attributes(func_name, program_dict, schedule_index, tree_footprint):
    schedule_json = program_dict['schedules_list'][schedule_index]
    sched_id = str(schedule_index).zfill(4)
    sched_str = sched_json_to_sched_str(schedule_json)
    exec_time = np.min(schedule_json['execution_times'])
    memory_use = program_dict['program_annotation']['memory_size']
    node_name = program_dict['node_name'] if 'node_name' in program_dict else 'unknown'
    speedup = program_dict['initial_execution_time']/exec_time 

    return (func_name, sched_id, sched_str, exec_time, memory_use, node_name, tree_footprint, speedup)

def sched_json_to_sched_str(sched_json): 
    
    if 'sched_str' in sched_json:
        return sched_json['sched_str']
    
    orig_loop_nest = []
    orig_loop_nest.append(sched_json['tree_structure']['loop_name'])
    child_list = sched_json['tree_structure']['child_list']
    while len(child_list)>0:
        child_loop = child_list[0]
        orig_loop_nest.append(child_loop['loop_name'])
        child_list = child_loop['child_list']
        
    comp_name = [n for n in sched_json.keys() if not n in ['unfuse_iterators','tree_structure','execution_times']][0]
    schedule = sched_json[comp_name]
    transf_loop_nest = orig_loop_nest
    sched_str = ''
    
    if 'Transformation Matrix' in schedule:
        if schedule['Transformation Matrix']:
            sched_str+='M('+','.join(schedule['Transformation Matrix'])+')'
    elif "transformation_matrix" in schedule:
        if schedule['transformation_matrix']:
            sched_str+='M('+','.join(schedule['transformation_matrix'])+')'
    if schedule['interchange_dims']:
        first_dim_index = transf_loop_nest.index(schedule['interchange_dims'][0])
        second_dim_index = transf_loop_nest.index(schedule['interchange_dims'][1])
        sched_str+='I(L'+str(first_dim_index)+',L'+str(second_dim_index)+')'
        transf_loop_nest[first_dim_index], transf_loop_nest[second_dim_index] = transf_loop_nest[second_dim_index], transf_loop_nest[first_dim_index]
    if schedule['skewing']:
        first_dim_index = transf_loop_nest.index(schedule['skewing']['skewed_dims'][0])
        second_dim_index = transf_loop_nest.index(schedule['skewing']['skewed_dims'][1])
        first_factor = schedule['skewing']['skewing_factors'][0]
        second_factor = schedule['skewing']['skewing_factors'][1]
        sched_str+='S(L'+str(first_dim_index)+',L'+str(second_dim_index)+','+str(first_factor)+','+str(second_factor)+')'
    if schedule['parallelized_dim']:
        dim_index = transf_loop_nest.index(schedule['parallelized_dim'])
        sched_str+='P(L'+str(dim_index)+')'
    if schedule['tiling']:
        if schedule['tiling']['tiling_depth']==2:
            first_dim = schedule['tiling']['tiling_dims'][0]
            second_dim = schedule['tiling']['tiling_dims'][1]
            first_dim_index = transf_loop_nest.index(first_dim)
            second_dim_index = transf_loop_nest.index(second_dim)
            first_factor = schedule['tiling']['tiling_factors'][0]
            second_factor = schedule['tiling']['tiling_factors'][1]
            sched_str+='T2(L'+str(first_dim_index)+',L'+str(second_dim_index)+','+str(first_factor)+','+str(second_factor)+')'
            i = transf_loop_nest.index(first_dim)
            transf_loop_nest[i:i+1]=first_dim+'_outer', second_dim+'_outer'
            i = transf_loop_nest.index(second_dim)
            transf_loop_nest[i:i+1]=first_dim+'_inner', second_dim+'_inner'
        else: #tiling depth == 3
            first_dim = schedule['tiling']['tiling_dims'][0]
            second_dim = schedule['tiling']['tiling_dims'][1]
            third_dim = schedule['tiling']['tiling_dims'][2]
            first_dim_index = transf_loop_nest.index(first_dim)
            second_dim_index = transf_loop_nest.index(second_dim)
            third_dim_index = transf_loop_nest.index(third_dim)
            first_factor = schedule['tiling']['tiling_factors'][0]
            second_factor = schedule['tiling']['tiling_factors'][1]
            third_factor = schedule['tiling']['tiling_factors'][2]
            sched_str+='T3(L'+str(first_dim_index)+',L'+str(second_dim_index)+',L'+str(third_dim_index)+','+str(first_factor)+','+str(second_factor)+','+str(third_factor)+')'
            i = transf_loop_nest.index(first_dim)
            transf_loop_nest[i:i+1]=first_dim+'_outer', second_dim+'_outer', third_dim+'_outer'
            i = transf_loop_nest.index(second_dim)
            transf_loop_nest[i:i+1]=first_dim+'_inner', second_dim+'_inner', third_dim+'_inner'
            transf_loop_nest.remove(third_dim)
    if schedule['unrolling_factor']:
        dim_index = len(transf_loop_nest)-1
        dim_name =transf_loop_nest[-1]
        sched_str+='U(L'+str(dim_index)+','+schedule['unrolling_factor']+')'
        transf_loop_nest[dim_index:dim_index+1] = dim_name+'_Uouter', dim_name+'_Uinner'
    
    
    return sched_str
    
def pad_access_matrix(access_matrix, max_depth):
    access_matrix = np.array(access_matrix)
    access_matrix = np.c_[np.ones(access_matrix.shape[0]), access_matrix] # adding tags for marking the used rows
    access_matrix = np.r_[[np.ones(access_matrix.shape[1])], access_matrix] # adding tags for marking the used columns
    padded_access_matrix = np.zeros((max_depth + 1, max_depth + 2))
    padded_access_matrix[:access_matrix.shape[0],:access_matrix.shape[1]-1] = access_matrix[:,:-1] #adding padding to the access matrix before the last column
    padded_access_matrix[:access_matrix.shape[0],-1] = access_matrix[:,-1] #appending the last columns
    
    return padded_access_matrix

def isl_to_write_matrix(isl_map): # for now this function only support reductions
    comp_iterators_str = re.findall(r'\[(.*)\]\s*->', isl_map)[0]
    buffer_iterators_str = re.findall(r'->\s*\w*\[(.*)\]', isl_map)[0]
    buffer_iterators_str=re.sub(r"\w+'\s=","",buffer_iterators_str)
    comp_iter_names = re.findall(r'(?:\s*(\w+))+', comp_iterators_str)
    buf_iter_names = re.findall(r'(?:\s*(\w+))+', buffer_iterators_str)
    matrix = np.zeros([len(buf_iter_names),len(comp_iter_names)+1])
    for i,buf_iter in enumerate(buf_iter_names):
        for j,comp_iter in enumerate(comp_iter_names):
            if buf_iter==comp_iter:
                matrix[i,j]=1
                break
    return matrix

def isl_to_write_dims(isl_map): # return the buffer iterator that defines the write buffer
    buffer_iterators_str = re.findall(r'->\s*\w*\[(.*)\]', isl_map)[0]
    buffer_iterators_str = re.sub(r"\w+'\s=","",buffer_iterators_str)
    buf_iter_names = re.findall(r'(?:\s*(\w+))+', buffer_iterators_str)
    return buf_iter_names

def get_results_df(dataset, batches_list, indices, model, log=False):   
    df = pd.DataFrame()
    model.eval()
    torch.set_grad_enabled(False)
    all_outputs=[]
    all_labels=[]
    prog_names=[]
    sched_names=[]
    exec_times=[]
    sched_strs=[]
    memory_uses=[]
    node_names=[]
    tree_footprints = []

    for k, (inputs, labels) in tqdm(list(enumerate(batches_list))):
        original_device = labels.device
        inputs=(inputs[0], inputs[1].to(train_device), inputs[2].to(train_device))
        labels=labels.to(train_device)
        outputs = model(inputs)
        assert outputs.shape == labels.shape
        all_outputs.append(outputs)
        all_labels.append(labels)
#         assert len(outputs)==len(dataset.batched_schedule_names[indices[k]])
#         assert len(outputs)==len(dataset.batched_program_names[indices[k]])
#         for j, sched_name in enumerate(dataset.batched_schedule_names[indices[k]]):
#             sched_names.append(sched_name)
#             prog_names.append(dataset.batched_program_names[indices[k]][j])
#             exec_times.append(dataset.batched_exec_time[indices[k]][j])
        assert len(outputs)==len(dataset.batched_datapoint_attributes[indices[k]])
        zipped_attributes = list(zip(*dataset.batched_datapoint_attributes[indices[k]]))
        prog_names.extend(zipped_attributes[0])
        sched_names.extend(zipped_attributes[1])
        sched_strs.extend(zipped_attributes[2])
        exec_times.extend(zipped_attributes[3])
        memory_uses.extend(zipped_attributes[4])
        node_names.extend(zipped_attributes[5])
        tree_footprints.extend(zipped_attributes[6])
        inputs=(inputs[0], inputs[1].to(original_device), inputs[2].to(original_device))
        labels=labels.to(original_device)
    preds = torch.cat(all_outputs)
    targets = torch.cat(all_labels)
    preds = preds.cpu().detach().numpy().reshape((-1,))
    preds = np.around(preds,decimals=6)
    targets = np.around(targets.cpu().detach().numpy().reshape((-1,)),decimals=6)
                                            
    assert preds.shape == targets.shape 
    df['name'] = prog_names
    df['tree_struct'] = tree_footprints
    df['sched_name'] = sched_names
    df['sched_str'] = sched_strs
    df['exec_time'] = exec_times
    df['memory_use'] = list(map(float,memory_uses))
    df['node_name'] = node_names
    df['prediction'] = np.array(preds)
    df['target'] = np.array(targets)
#     df['abs_diff'] = np.abs(preds - targets)
    df['APE'] = np.abs(df.target - df.prediction)/df.target * 100
    df['sched_str'] = df['sched_str'].apply(lambda x: simplify_sched_str(x))
        
    return df

def simplify_sched_str(sched_str): #checks if the the same matrix is applied multiple computations, then merge the M() parts into a single 
#     print('before ')
    if sched_str.count('M')==1:
        return sched_str
    comps = re.findall('C\d+', sched_str)
    comps = set(comps)
    
    mats = set(re.findall(r'M\({[\dC\,]+},([\d\,\-]+)',sched_str))
    comps_per_mat = {mat:[] for mat in mats}
    new_mats_str = ''
    for mat in comps_per_mat:
        for mat_part in re.findall('M\({[C\d\,]+},'+mat,sched_str):
            comps_per_mat[mat].extend(re.findall('C\d+',mat_part))
        new_mats_str+='M({'+','.join(sorted(comps_per_mat[mat]))+'},'+mat+')'
    return re.sub('(M\({[\dC\,]+},[\d\,\-]+\))+',new_mats_str,sched_str)




def access_is_stencil(access):
    return np.any(access['access_matrix'], axis=0)[-1]
def linear_diophantine_default(f_i,f_j):
    found = False
    gamma = 0
    sigma = 1
    if ((f_j == 1) or (f_i == 1)):
        gamma = f_i - 1
        sigma = 1
    else:
        if((f_j == -1) and (f_i > 1)):
            gamma = 1
            sigma = 0       
        else:     
            i =0
            while((i < 100) and (not found)):     
                if (((sigma * f_i ) % abs(f_j)) ==  1):
                            found = True
                else:
                    sigma+=1
                    i+=1
            if(not found):
                print('Error cannof find solution to diophantine equation')
                return
            gamma = ((sigma * f_i) - 1 ) / f_j
    
    return gamma, sigma








# Maximum sequence of transformations (reversal, interchange and skewing) allowed. Currently set to 5 
MAX_NUM_TRANSFORMATIONS = 4

# Maximum size of the tags vector representing each transformation
MAX_TAGS = 8

def get_transformations_list(
    program_json, schedule_json, max_depth,
):
    
    computations_dict = program_json["computations"]
    ordered_comp_list = sorted(
        list(computations_dict.keys()),
        key=lambda x: computations_dict[x]["absolute_order"],
    )
    
    transformations_list = []
    for comp_index, comp_name in enumerate(ordered_comp_list):
        comp_dict = program_json["computations"][comp_name]
        comp_schedule_dict = schedule_json[comp_name]
        
        # Check which transformations (interchange, reversal and skweing) were applied and add the padded vector representation to their corresponding position
        padded_tags = get_transformation_tags(
            program_json, schedule_json, comp_name, max_depth
        )
        transformations_list.append(padded_tags)

    return transformations_list

def get_transformation_tags(
    program_json, schedule_json, comp_name, max_depth=None
):
    # Extract information about the computation and the transformations that were applied from the json input
    comp_dict = program_json["computations"][comp_name]
    comp_schedule_dict = schedule_json[comp_name]
    nb_iterators = len(comp_dict["iterators"])
    loop_nest = comp_dict["iterators"][:]
    
    # Create an identity vector that represents that no transformation was applied
    identity = np.zeros((nb_iterators, nb_iterators), int)
    np.fill_diagonal(identity, 1)
    identity_tags = np.zeros((1,MAX_TAGS), dtype=np.int32)
    
    tag_factors = []
    
    # If the transformations are represented using matrices
    if "transformation_matrices" in comp_schedule_dict:
        
        if comp_schedule_dict["transformation_matrices"] != []:
            if ("transformation_matrix" in comp_schedule_dict) and (
                comp_schedule_dict["transformation_matrix"]
            ):
                # transformation_matrix represents all of the applied transformations in a single compact matrix
                final_transformation_matrix = np.array(
                    list(map(int, comp_schedule_dict["transformation_matrix"]))
                ).reshape(nb_iterators, nb_iterators)
            else:
                final_transformation_matrix = identity.copy()

            
            tag_factors = []
            # For each transformation in the schedule
            for matrix in comp_schedule_dict["transformation_matrices"][::-1]:
                # Check whether the size of the matrix is coherent
                assert np.sqrt(len(matrix)) == nb_iterators
                transformation_matrix = np.array(list(map(int, matrix))).reshape(
                    nb_iterators, nb_iterators
                )
                tags_vector = identity_tags.copy()
                # Calculate the residual to check whether this transformation is the identity matrix
                residual = np.abs(identity - transformation_matrix)
                
                mask_line = residual.sum(axis=1) > 0
                mask_col  = residual.sum(axis=0) > 0
                
                non_zeros_positions = np.argwhere(residual>0)

                for index in non_zeros_positions:
                    if (abs(index[0] - index[1]) > 1) and nb_iterators != abs(transformation_matrix).sum():

                        raise RandomMatrix
                    else:
                        if (index[1] < index[0]):
                            
                            first_factor = transformation_matrix[index[0]-1][index[1]]
                            second_factor = transformation_matrix[index[0]-1][index[1]+1]

                            if ( index[1] < index[0] and first_factor == 1 and second_factor == 0 ):

                                raise RandomMatrix
                
                # If that's the case (Identity matrix), skip it to avoid adding it in the representation
                if (transformation_matrix == identity).all():
                    continue
                    
                # If the transformation isn't the identity, fill the tag_vector with the corresponsonding transformation info
                elif residual.sum() != 0 and nb_iterators == transformation_matrix.sum():
                    # If the sum of the elements in the matrix is equal to the number of iterators, then interchange has been applied
                    tags_vector[0][0] = 1
                    
                    # Extract the interchange parameters (loop indecies) through the created mask
                    dims = np.arange((nb_iterators),dtype=int)[mask_line]
                    
                    assert dims.shape[0] == 2
                    first_iter_index, second_iter_index=dims
                    
                    # Add the interchange parameters to the tag vector
                    tags_vector[0][1], tags_vector[0][2] = first_iter_index, second_iter_index
                
                elif nb_iterators - 2 == np.trace(transformation_matrix) and transformation_matrix.sum() - np.trace(transformation_matrix) == 0:  
                    # If the sum of the elements in the matrix is equal to the number of iterators -2, then loop reversal has been applied
                    tags_vector[0][0] = 2      
                    
                    # Extract the reversed loop index
                    dims = np.arange((nb_iterators),dtype=int)[mask_line]
#                     print(transformation_matrix)
                    assert dims.shape[0] == 1
                    index=dims[0]
                    
                    # Add the reversal parameter to the tags vector
                    tags_vector[0][3] = index
                else:
                    # If the matrix isn't one of the first three cases (identity, reversal, interchange) then we know it represents skewing
                    tags_vector[0][0] = 3
                    dims_line = np.arange((nb_iterators),dtype=int)[mask_line]
                    dims_col = np.arange((nb_iterators),dtype=int)[mask_col]
                    
                    dims_line = [dims_line[0], dims_line[0]+1]
                    
                    dims_col = [dims_col[0], dims_col[0]+1]
                    
                    # Extract which loops have been skewed 
                    first_iter_index, second_iter_index = dims_line
                    
                    # Add the skewed loops indecies to the tags vector
                    tags_vector[0][4], tags_vector[0][5] = first_iter_index, second_iter_index

                    
                    # Extract the skewing factors 
                    first_factor = transformation_matrix[first_iter_index, first_iter_index]
                    second_factor = transformation_matrix[first_iter_index, second_iter_index]
                    a = transformation_matrix[second_iter_index, first_iter_index]
                    b = transformation_matrix[second_iter_index, second_iter_index]
                    
        
                    if ((a, b) != linear_diophantine_default(first_factor, second_factor)):
                        raise RandomMatrix
                    
                    # Add the skewing factors to the tags vector
                    tags_vector[0][6], tags_vector[0][7] = first_factor, second_factor
                    # if(first_factor != 1):
                    #     print("transformation matrix")
                    #     print(transformation_matrix)
                    #     print("tags_vector")
                    #     print(tags_vector)
                
                transformation_matrix = np.c_[
                    np.ones(transformation_matrix.shape[0]), transformation_matrix
                ]
                transformation_matrix = np.r_[
                    [np.ones(transformation_matrix.shape[1])], transformation_matrix
                ]
                transformation_matrix = np.pad(
                    transformation_matrix,
                    [
                        (0, max_depth + 1 - transformation_matrix.shape[0]),
                        (0, max_depth + 1 - transformation_matrix.shape[1]),
                    ],
                    mode="constant",
                    constant_values=0,
                )
                tag_factors.append(tags_vector)
            
            # We use MAX_NUM_TRANSFORMATIONS+1 instead of MAX_NUM_TRANSFORMATIONS to include the matrix that represents the whole sequence
            if len(tag_factors) > (MAX_NUM_TRANSFORMATIONS):
                # If the number of transformations is greater than the maximum allowed, raise an exception
                raise NbTranformationException
    
    
            # If no tranformation was found (comp_schedule_dict["transformation_matrices"] only contains the identity) then we use the idendity tags
            final_tags = (
                np.concatenate(tag_factors, axis=0)
                if tag_factors
                else identity_tags.copy()
            )
        else:
            # If comp_schedule_dict["transformation_matrices"] is empty we also use the identity tags 
            final_transformation_matrix = identity.copy()
            final_tags = identity_tags.copy()
        # To make sure the data is coherent, we want to check whether the sequence of transformations is equal to the final transformation matrix (represented by final_transformation_matrix) 
        # In the polyhedral model, the multiplication of the sequence of transformation matrices results in one matrix that represents the whole combination
        comparison_matrix = identity.copy()
        for mat in comp_schedule_dict["transformation_matrices"][::-1]:
            comparison_matrix = comparison_matrix @ np.array(
                list(map(int, mat))
            ).reshape(nb_iterators, nb_iterators)
        
        assert (comparison_matrix == final_transformation_matrix).all()
    else:
        # If the non-polyhedral representation is used, only interchange and skewing are applied
        # The skewing and interchange parameters are present in the json
        # We directly add them to the tags vector 
        if comp_schedule_dict["interchange_dims"]:
            first_iter_index = loop_nest.index(
                comp_schedule_dict["interchange_dims"][0]
            )
            second_iter_index = loop_nest.index(
                comp_schedule_dict["interchange_dims"][1]
            )
            loop_nest[first_iter_index], loop_nest[second_iter_index] = (
                loop_nest[second_iter_index],
                loop_nest[first_iter_index],
            )
            tags_vector = identity_tags.copy()
            tags_vector[0][0] = 1
            tags_vector[0][1], tags_vector[0][2] = first_iter_index, second_iter_index
            tag_factors.append(tags_vector)

        
        if comp_schedule_dict["skewing"]:
            first_iter_index = loop_nest.index(
                comp_schedule_dict["skewing"]["skewed_dims"][0]
            )
            second_iter_index = loop_nest.index(
                comp_schedule_dict["skewing"]["skewed_dims"][1]
            )
            first_factor = int(comp_schedule_dict["skewing"]["skewing_factors"][0])
            second_factor = int(comp_schedule_dict["skewing"]["skewing_factors"][1])

            tags_vector = identity_tags.copy()
            tags_vector[0][0] = 3
            tags_vector[0][4], tags_vector[0][5] = first_iter_index, second_iter_index
            tags_vector[0][6], tags_vector[0][7] = first_factor, second_factor
            tag_factors.append(tags_vector)
        # If neither skewing or interchange were applied, we use the identity tags vector
        final_tags = (
            np.concatenate(tag_factors, axis=0)
            if tag_factors
            else identity_tags.copy()
        )
        
    return final_tags




# returns a string representation of a schedule and the transformations applied in it
def get_schedule_str_new(program_json, sched_json):
    comp_name = [
        n
        for n in sched_json.keys()
        if not n in [ "tree_structure", "execution_times", "fusions", "sched_str", "legality_check", "exploration_method"]
    ]
    sched_str = ""
    if ("fusions" in sched_json and sched_json["fusions"]):
        for fusion in sched_json["fusions"]:
            sched_str += "{"
            for name in comp_name:
                if name in fusion:
                    sched_str += name + ","
            
            sched_str = sched_str[:-1]
            sched_str += "}:F(L"+str(fusion[-1])+")"
   
    comp_transformations = get_transformations_list(
            program_json,
            sched_json,
            5
        )
    
    computations_dict = program_json["computations"]
    ordered_comp_list = sorted(
                list(comp_name),
                key=lambda x: computations_dict[x]["absolute_order"],
            )
    for comp_index, name in enumerate(ordered_comp_list):
        transf_loop_nest = program_json["computations"][name]["iterators"].copy()
        schedule = sched_json[name]
        sched_str += '{' + name + '}:'
        


        schedule["transformations_list"] = comp_transformations[comp_index].tolist()[::-1]
        for transformation in schedule["transformations_list"]:

            if (transformation[0] == 1):
                sched_str += "I(L" + str(transformation[1]) + ",L" + str(transformation[2]) + ")"
                
            elif (transformation[0] == 2):
                sched_str += "R(L" + str(transformation[3])+ ")"
            elif (transformation[0] == 3):
                sched_str += "S(L" + str(transformation[4]) + ",L" + str(transformation[5]) + "," + str(transformation[6]) + "," + str(transformation[7]) + ")"
                
        if schedule["parallelized_dim"]:
            
            dim_index = transf_loop_nest.index(schedule["parallelized_dim"])
            sched_str += "P(L" + str(dim_index) + ")"

        if schedule["tiling"]:
            if schedule["tiling"]["tiling_depth"] == 2:
                first_dim = schedule["tiling"]["tiling_dims"][0]
                second_dim = schedule["tiling"]["tiling_dims"][1]
                
                first_dim_index = transf_loop_nest.index(first_dim)
                second_dim_index = transf_loop_nest.index(second_dim)
                first_factor = schedule["tiling"]["tiling_factors"][0]
                second_factor = schedule["tiling"]["tiling_factors"][1]
                sched_str += (
                    "T2(L"
                    + str(first_dim_index)
                    + ",L"
                    + str(second_dim_index)
                    + ","
                    + str(first_factor)
                    + ","
                    + str(second_factor)
                    + ")"
                )
                i = transf_loop_nest.index(first_dim)
                transf_loop_nest[i : i + 1] = first_dim + "_outer", second_dim + "_outer"
                i = transf_loop_nest.index(second_dim)
                transf_loop_nest[i : i + 1] = first_dim + "_inner", second_dim + "_inner"
            else:
                first_dim = schedule["tiling"]["tiling_dims"][0]
                second_dim = schedule["tiling"]["tiling_dims"][1]
                third_dim = schedule["tiling"]["tiling_dims"][2]
                first_dim_index = transf_loop_nest.index(first_dim)
                second_dim_index = transf_loop_nest.index(second_dim)
                third_dim_index = transf_loop_nest.index(third_dim)
                first_factor = schedule["tiling"]["tiling_factors"][0]
                second_factor = schedule["tiling"]["tiling_factors"][1]
                third_factor = schedule["tiling"]["tiling_factors"][2]
                sched_str += (
                    "T3(L"
                    + str(first_dim_index)
                    + ",L"
                    + str(second_dim_index)
                    + ",L"
                    + str(third_dim_index)
                    + ","
                    + str(first_factor)
                    + ","
                    + str(second_factor)
                    + ","
                    + str(third_factor)
                    + ")"
                )
                i = transf_loop_nest.index(first_dim)
                transf_loop_nest[i : i + 1] = (
                    first_dim + "_outer",
                    second_dim + "_outer",
                    third_dim + "_outer",
                )
                i = transf_loop_nest.index(second_dim)
                transf_loop_nest[i : i + 1] = (
                    first_dim + "_inner",
                    second_dim + "_inner",
                    third_dim + "_inner",
                )
                transf_loop_nest.remove(third_dim)

        if schedule["unrolling_factor"]:
            dim_index = len(transf_loop_nest) - 1
            dim_name = transf_loop_nest[-1]
            sched_str += "U(L" + str(dim_index) + "," + schedule["unrolling_factor"] + ")"
            transf_loop_nest[dim_index : dim_index + 1] = (
                dim_name + "_Uouter",
                dim_name + "_Uinner",
            )
    return sched_str
