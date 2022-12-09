#!/usr/bin/env python
# coding: utf-8

# In[1]:


import datetime
begin_time = datetime.datetime.now()
import os
import glob
import string
import chemparse
import re
import numpy as np
import pandas as pd
from collections import namedtuple
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout
import pydot
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.ticker as ticker
from mycolorpy import colorlist as mcp
import math

m_h = 1.007825
m_c = 12.
m_n = 14.003074
m_o = 15.994915
m_s = 31.972071
masses = (m_h, m_c, m_n, m_o, m_s)
valence_h = 1
valence_c = 4
valence_n = 3
valence_o = 2
valence_s = 2


# In[28]:


get_ipython().system('pip show networkx')


# In[22]:


def mol2net_layout(trafos = ['H2','O','CO','NH3'], nComponents = 1):

    ##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #trafos          =       ['H2','O','CO','NH3']
    ##trafoWeights    =       [ 1,   5,   1,   2]
    trafoWeights    =       [1]*len(trafos)
    columnNames     =       ['molecular_formula','abundance_int']             ##  for  molFormula  &  abundance/intensity
    sample          =       'Ice composition H2O_CH3OH_NH3'
    #nComponents     =       2
    only_uniques    =       'yes'
    color_type      =       sample
    node_size       =       'Degree'
    ##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    ######################################
    ##                              TRAFOS
    ######################################
    cf = re.compile('([A-Z][a-z]?)(\d?\d?\d?)')
    all_elements = []
    for xxxx in trafos:
        for el, count in re.findall(cf, xxxx):
            all_elements += [el]

    atoms = set(all_elements)
    counts = [[""] + trafos]
    for e in atoms:
        count = [e]
        for xxxx in trafos:    
            d = dict(re.findall(cf, xxxx))
            n = d.get(e, 0)
            if n == '': n = 1
            count += [int(n)]
        counts += [count]

    A = np.array(counts)
    A_t = (A.transpose())

    x2 = pd.DataFrame(A_t)
    new_header = x2.iloc[0]
    x2 = x2[1:]
    x2.columns = new_header
    x2 = x2.rename(columns={'': 'Molecular Formula'})
    for i in ['H', 'C', 'N', 'O', 'S']:
        if i in x2.columns:
            x2[i] = x2[i].astype(int)
        else:
            x2[i] = 0    

    h = x2['H'].to_numpy()
    c = x2['C'].to_numpy()
    n = x2['N'].to_numpy()    
    o = x2['O'].to_numpy()
    s = x2['S'].to_numpy()
    elements = np.transpose(np.array([h, c, n, o, s]))
    massExact = np.dot(elements, masses)
    x2['Mass (exact)'] = massExact.round(6)
    weights = pd.DataFrame(trafoWeights,columns = ['trafoWeights'])
    weights['trafos'] = trafos
    x3 = x2.join(weights.set_index(['trafos'], verify_integrity=True ),on=['Molecular Formula'], how='left')

    xTRAFOS = x3
    mdTRAFOS = xTRAFOS['Mass (exact)'].round(6).to_numpy()

    ######################################
    ##                          DETECTIONS
    ######################################
    path = os.getcwd()
    files = glob.glob(os.path.join("*.tsv"))
    d = {}
    for f in files:
        key = f
        name = f.replace(".tsv", "")
        x = globals()[f"orig_{name}"] = pd.read_csv(f, sep='\t', skiprows=0)
        if 'id' in x.columns:
            pass
        else:
            x.insert(0, 'id', range(1, 1 + len(x)))
    #    
        try:
            x = x[['id',sample,columnNames[0],columnNames[1]]].rename(columns={columnNames[0]: 'Molecular Formula',columnNames[1]:'Intensity'})
        except:
            x = x[['id',columnNames[0],columnNames[1]]].rename(columns={columnNames[0]: 'Molecular Formula',columnNames[1]:'Intensity'})
    #
        molFormulas = x['Molecular Formula']
        a = []
        for i in molFormulas:
            tmp_a = chemparse.parse_formula(i)
            a.append(tmp_a)
    #
        x_a = pd.DataFrame(a).fillna(0)
        x2 = pd.concat([x,x_a], axis=1)
    #
        if 'H' in x2.columns:
            x2['H'] = x2['H'].astype(int)
        else:
            x2['H'] = 0    
    #
        if 'C' in x2.columns:
            x2['C'] = x2['C'].astype(int)
        else:
            x2['C'] = 0    
    #
        if 'N' in x2.columns:
            x2['N'] = x2['N'].astype(int)
        else:
            x2['N'] = 0    
    #
        if 'O' in x2.columns:
            x2['O'] = x2['O'].astype(int)
        else:
            x2['O'] = 0    
    #
        if 'S' in x2.columns:
            x2['S'] = x2['S'].astype(int)
        else:
            x2['S'] = 0    
    #
        h = x2['H'].to_numpy()
        c = x2['C'].to_numpy()
        o = x2['O'].to_numpy()
        n = x2['N'].to_numpy()    
        s = x2['S'].to_numpy()
        elements = np.transpose(np.array([h, c, n, o, s]))
        massExact = np.dot(elements, masses)
        x2['Mass (exact)'] = massExact.round(6)
        x2['mu'] = mu = ( (h*valence_h + c*valence_c + n*valence_n + o*valence_o + s*valence_s) / 2 ) - (h + c + o + n + s) + 1
        x2['DBE'] = dbe = c - (h/2) + (n/2) + 1
        x2['HC'] = h/c
        x2['NC'] = n/c
        x2['OC'] = o/c
        x2['SC'] = s/c
        x2['FILTER'] =  1 * ( (mu >= 0) )
        atom_list = ['H','C','N','O','S']
        for i in range(len(atom_list)-1):
            x2['Molecular Formula'] = x2['Molecular Formula'].replace({atom_list[i]+'0'}, {atom_list[i]}, regex=True)
            x2['Molecular Formula'] = x2['Molecular Formula'].replace({atom_list[i]+'1'+atom_list[i+1]}, {atom_list[i]}, regex=True)
            x2['Molecular Formula'] = x2['Molecular Formula'].replace({'S'+'0'}, {''}, regex=True)
            x2['Molecular Formula'] = x2['Molecular Formula'].replace({'S'+'1'}, {'S'}, regex=True)
    #
        x2['Mass (frequency)'] = x2['Mass (exact)'].map(dict(x2['Mass (exact)'].value_counts()))
        try:
            if only_uniques == 'yes':
                x3 = x2.drop_duplicates(subset=['Mass (exact)'], keep='first')
            else:
                pass
        except:
            x3 = x2
    #
        xNODES0 = xDET = x3[(x3['FILTER']==1)]
        globals()[f"{name}"] = xDET
        mass2 = mass = xDET['Mass (exact)'].round(6).to_numpy()
        data = xDET[['id','Mass (exact)']]
    #
        ##########################################     x x    
        ##                              MD MATCHES
        ##########################################
        md_matches = namedtuple('md_matches', 'md_matches hits')
        new_list = []
        for md in mdTRAFOS:
            for element in mass:
                if element+md in mass2:
                    tmp_new_list = md_matches(element, element+md)
                    new_list.append(tmp_new_list)
        #
        matches = np.array(new_list).reshape(len(new_list),2)
        matches = matches[np.argsort(matches[:,0])]
        sources = data.rename(columns={'id': 'Source', 'Mass (exact)': 'Mass (source)'})
        targets = data.rename(columns={'id': 'Target', 'Mass (exact)': 'Mass (target)'})
        matches2 = pd.DataFrame({'Mass (source)': matches[:, 0], 'Mass (target)': matches[:, 1]})
        source_match = matches2.merge(sources, how='left', on=['Mass (source)'])
        target_match = source_match.merge(targets, how='left', on=['Mass (target)'])
        target_match['Mass difference'] = mass_diff = np.around(target_match['Mass (target)'] - target_match['Mass (source)'],6)
        xxxx = xTRAFOS[['Mass (exact)','Molecular Formula']].rename(columns={'Mass (exact)': 'Mass difference'})
        x = target_match.merge(xxxx, how='left', on=['Mass difference'])
        x['type'] = type = np.full((len(mass_diff),1), 'Undirected')
        x['Label'] = Label = np.full((len(mass_diff),1), 'x')
        xxxx = xTRAFOS[['Mass (exact)','trafoWeights']]
        x2 = x.join(xxxx.set_index(['Mass (exact)'], verify_integrity=True ),on=['Mass difference'], how='left').rename(columns={'trafoWeights': 'Weight'}).sort_values(by=['Source'])
        x2['Mass difference (frequency)'] = x2['Mass difference'].map(dict(x2['Mass difference'].value_counts()))
        #
        xEDGES0 = x2
        #
        md_freq = x2[['Molecular Formula','Mass difference (frequency)']]
        md_freq = md_freq.drop_duplicates(subset=['Molecular Formula'], keep='first')
        md_freq = md_freq.sort_values(by=['Mass difference (frequency)'], ascending=False)
        print('\n\n\n\n')
        print(name+', '+str(len(xNODES0))+' nodes,', str(len(xEDGES0))+' edges')
        print(md_freq)
        print('\n\n\n\n')
    #
        ##########################################     x x    
        ##                             NETWORK ANA
        ##########################################
        G0 = nx.from_pandas_edgelist(xEDGES0, 'Source', 'Target', create_using=nx.Graph())
        G0cc = list(nx.connected_components(G0))
        d = {name:k for k,comp in enumerate(G0cc) for name in comp}               #dict(enumerate(G0cc))
        df_comp = pd.DataFrame.from_dict(d, orient='index', columns=['component']).rename_axis('id').reset_index()
        nodes2 = xNODES0.merge(df_comp, how='left', on='id')
        nans = nodes2[nodes2.isna().any(axis=1)]
        xNODES0 = nodes2[~nodes2['component'].isnull()]
        #
        df_source = xEDGES0.rename(columns={'Source':'id'})
        edges2 = df_source.merge(df_comp, how='left', on='id')
        edges2 = edges2.rename(columns={'component':'component source','id':'Source'})
        df_target = edges2.rename(columns={'Target':'id'})
        edges3 = df_target.merge(df_comp, how='left', on='id')
        edges3 = edges3.rename(columns={'component':'component target','id':'Target','mass_difference':'mass difference','molecular_formula':'molecular formula'})
        #
        df_comp_freq = pd.DataFrame(xNODES0['component'].value_counts()).rename(columns={'component': 'freq_component'})
        xNODES0['df_comp_freq'] = xNODES0['component'].map(dict(xNODES0['component'].value_counts()))
        df_comp_freq_norm = pd.DataFrame(xNODES0['component'].value_counts(normalize=True)).rename(columns={'component': 'freq_component_norm [0-1]'})
        df_comp_freq_comb = pd.concat([df_comp_freq,df_comp_freq_norm], axis=1)
        print('\n\n\n\n')
        print(name+' (all components)')
        print(df_comp_freq_comb)
        print('\n\n\n\n')
        #
        FILTER_COMPONENTS  =        list(df_comp_freq_comb.index)[:nComponents]
        xEDGES = edges3[(edges3['component source'].isin(FILTER_COMPONENTS))]
        xNODES = xNODES0[(xNODES0['component'].isin(FILTER_COMPONENTS))]
        G = nx.from_pandas_edgelist(xEDGES, 'Source', 'Target', create_using=nx.Graph())
        a = pd.DataFrame(G.nodes(data = True)).rename(columns={0:'id'}).drop([1], axis=1)
        a2 = a.join(xNODES.set_index('id'), on='id')
        a2['id_G0_nodes'] = range(1,1+len(a2))
        xNODES = a2
    #
        degrees = pd.DataFrame(G.degree(), columns=["id", "Degree"])
        degrees['id'] = degrees['id'].astype(int)
        xNODES = xNODES.join(degrees.set_index('id'), on='id') 
        xNODES['Degree counts'] = xNODES['Degree'].map(dict(xNODES['Degree'].value_counts()))
    #
    ##    bc = nx.betweenness_centrality(G, normalized=True)                                                                                                                                                                                             ## takes a while....
    ##    bc2 = pd.DataFrame.from_dict(bc, orient='index', columns=['betCen']).rename_axis('id').reset_index()
    ##    xNODES = xNODES.merge(bc2, how='left', on='id')
    #
        colors = mcp.gen_color(cmap='gist_rainbow',n=len(xNODES[color_type].unique()))                    ##  cmap  AUTUMN tab20b coolwarm gist_rainbow tab20c brg      https://matplotlib.org/stable/tutorials/colors/colormaps.html
        ##colors = ['#15B01A',    'blue', 'orange',     'lightgray',   'red', 'green']
        samples0 = xNODES[color_type].unique().tolist()
        try:
            samples = sorted(samples0, key=lambda x: int("".join([i for i in x if i.isdigit()])))
        except:
            samples = sorted(samples0)
        #
        sample_colors = pd.DataFrame({color_type: samples, 'color': colors})
        xNODES = xNODES.merge(sample_colors, how='left', on=color_type)
        xNODES = xNODES.sort_values(by=['id_G0_nodes'], ascending=True)                              ## re-ordering/re-indexing according to graph G0 (id_G0_nodes)
        node_colors = xNODES['color'].tolist()
    #
        xNODES = xNODES.sort_values(by=['id_G0_nodes'], ascending=True)
        xNODES['node_size'] = (xNODES[node_size] - np.min(xNODES[node_size])) / (np.max(xNODES[node_size]) - np.min(xNODES[node_size]))
        node_sizes = xNODES['node_size'].tolist()
        node_sizes = [i * 1e2 for i in node_sizes]
    #


    ###########################################
    ####                                 LAYOUT
    ###########################################
    plt.figure()
    plt.title(name, wrap=True, fontsize=12)
    nx.draw(G, pos = nx.nx_pydot.graphviz_layout(G), with_labels=False, node_color=node_colors, node_size=20, width=.03, font_size=5)
        #nx.draw(G, pos = nx.nx_pydot.graphviz_layout(G), with_labels=False, node_color=node_colors, node_size=node_sizes, width=.05, alpha=1)
        #nx.draw(G, pos = nx.spring_layout(G, scale=1), with_labels=False, node_color=node_colors, node_size=node_sizes, width=.05, alpha=1)
        #nx.draw(G, pos = nx.spring_layout(G, k=.5/math.sqrt(G.order()), scale=10), with_labels=False, node_color=node_colors, node_size=node_sizes, width=.05, alpha=1)
    #nx.draw(G, pos = nx.nx_pydot.graphviz_layout(G), with_labels=False, node_color=node_colors, node_size=2, width=.01, alpha=1)
    #nx.draw(G, pos = nx.nx_pydot.graphviz_layout(G), with_labels=False, node_color=node_colors, node_size=1, width=.05, alpha=1)

    for i in range(len(sample_colors)):
        plt.plot([], [], sample_colors.values[i][1], marker='o', markersize=10, label=sample_colors.values[i][0])      


    # PRINTING FIGURE 1.

    plt.legend()                      ## loc='upper left'
    plt.text(.6, -1.1,str(G.number_of_nodes())+' nodes, '+str(G.number_of_edges())+' edges, node size ~ '+str(node_size), fontsize=8, wrap=True)
    #plt.text(.6, -1.1,str(G.number_of_nodes())+' nodes, '+str(G.number_of_edges())+' edges', fontsize=8, wrap=True)
    plt.savefig(name+'_Comp'+str(FILTER_COMPONENTS)+'    '+str(color_type)+'    network'+'.png')
    plt.show()


# In[3]:


def mol2net_clust_trafo(SAMPLE_START, SAMPLE_END):
    ##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    trafos          =       ['H2', 'O', 'CO', 'H3N', 'CHN', 'CH3N', 'H2O', 'CO2', 'CH2O', 'CHNO']
    ##trafoWeights    =       [ 1,   5,   1,   2]
    trafoWeights    =       [1]*len(trafos)
    columnNames     =       ['molecular_formula','abundance_int']             ##  for  molFormula  &  abundance/intensity
    sample          =       'Ice composition H2O_CH3OH_NH3'
    nComponents     =       1
    only_uniques    =       'yes'
    color_type      =       sample
    node_size       =       'Degree'
    ##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    ######################################
    ##                              TRAFOS
    ######################################
    cf = re.compile('([A-Z][a-z]?)(\d?\d?\d?)')
    all_elements = []
    for xxxx in trafos:
        for el, count in re.findall(cf, xxxx):
            all_elements += [el]

    atoms = set(all_elements)
    counts = [[""] + trafos]
    for e in atoms:
        count = [e]
        for xxxx in trafos:    
            d = dict(re.findall(cf, xxxx))
            n = d.get(e, 0)
            if n == '': n = 1
            count += [int(n)]
        counts += [count]

    A = np.array(counts)
    A_t = (A.transpose())

    x2 = pd.DataFrame(A_t)
    new_header = x2.iloc[0]
    x2 = x2[1:]
    x2.columns = new_header
    x2 = x2.rename(columns={'': 'Molecular Formula'})
    for i in ['H', 'C', 'N', 'O', 'S']:
        if i in x2.columns:
            x2[i] = x2[i].astype(int)
        else:
            x2[i] = 0    

    h = x2['H'].to_numpy()
    c = x2['C'].to_numpy()
    n = x2['N'].to_numpy()    
    o = x2['O'].to_numpy()
    s = x2['S'].to_numpy()
    elements = np.transpose(np.array([h, c, n, o, s]))
    massExact = np.dot(elements, masses)
    x2['Mass (exact)'] = massExact.round(6)
    weights = pd.DataFrame(trafoWeights,columns = ['trafoWeights'])
    weights['trafos'] = trafos
    x3 = x2.join(weights.set_index(['trafos'], verify_integrity=True ),on=['Molecular Formula'], how='left')

    xTRAFOS = x3
    mdTRAFOS = xTRAFOS['Mass (exact)'].round(6).to_numpy()

    ######################################
    ##                          DETECTIONS
    ######################################
    path = os.getcwd()
    files = glob.glob(os.path.join("*.tsv"))
    d = {}
    for f in files:
        key = f
        name = f.replace(".tsv", "")
        x = globals()[f"orig_{name}"] = pd.read_csv(f, sep='\t', skiprows=0)
        if 'id' in x.columns:
            pass
        else:
            x.insert(0, 'id', range(1, 1 + len(x)))
    #    
        try:
            x = x[['id',sample,columnNames[0],columnNames[1]]].rename(columns={columnNames[0]: 'Molecular Formula',columnNames[1]:'Intensity'})
        except:
            x = x[['id',columnNames[0],columnNames[1]]].rename(columns={columnNames[0]: 'Molecular Formula',columnNames[1]:'Intensity'})
    #
        molFormulas = x['Molecular Formula']
        a = []
        for i in molFormulas:
            tmp_a = chemparse.parse_formula(i)
            a.append(tmp_a)
    #
        x_a = pd.DataFrame(a).fillna(0)
        x2 = pd.concat([x,x_a], axis=1)
    #
        if 'H' in x2.columns:
            x2['H'] = x2['H'].astype(int)
        else:
            x2['H'] = 0    
    #
        if 'C' in x2.columns:
            x2['C'] = x2['C'].astype(int)
        else:
            x2['C'] = 0    
    #
        if 'N' in x2.columns:
            x2['N'] = x2['N'].astype(int)
        else:
            x2['N'] = 0    
    #
        if 'O' in x2.columns:
            x2['O'] = x2['O'].astype(int)
        else:
            x2['O'] = 0    
    #
        if 'S' in x2.columns:
            x2['S'] = x2['S'].astype(int)
        else:
            x2['S'] = 0    
    #
        h = x2['H'].to_numpy()
        c = x2['C'].to_numpy()
        o = x2['O'].to_numpy()
        n = x2['N'].to_numpy()    
        s = x2['S'].to_numpy()
        elements = np.transpose(np.array([h, c, n, o, s]))
        massExact = np.dot(elements, masses)
        x2['Mass (exact)'] = massExact.round(6)
        x2['mu'] = mu = ( (h*valence_h + c*valence_c + n*valence_n + o*valence_o + s*valence_s) / 2 ) - (h + c + o + n + s) + 1
        x2['DBE'] = dbe = c - (h/2) + (n/2) + 1
        x2['HC'] = h/c
        x2['NC'] = n/c
        x2['OC'] = o/c
        x2['SC'] = s/c
        x2['FILTER'] =  1 * ( (mu >= 0) )
        atom_list = ['H','C','N','O','S']
        for i in range(len(atom_list)-1):
            x2['Molecular Formula'] = x2['Molecular Formula'].replace({atom_list[i]+'0'}, {atom_list[i]}, regex=True)
            x2['Molecular Formula'] = x2['Molecular Formula'].replace({atom_list[i]+'1'+atom_list[i+1]}, {atom_list[i]}, regex=True)
            x2['Molecular Formula'] = x2['Molecular Formula'].replace({'S'+'0'}, {''}, regex=True)
            x2['Molecular Formula'] = x2['Molecular Formula'].replace({'S'+'1'}, {'S'}, regex=True)
    #
        x2['Mass (frequency)'] = x2['Mass (exact)'].map(dict(x2['Mass (exact)'].value_counts()))
        try:
            if only_uniques == 'yes':
                x3 = x2.drop_duplicates(subset=['Mass (exact)'], keep='first')
            else:
                pass
        except:
            x3 = x2
    #
        xNODES0 = xDET = x3[(x3['FILTER']==1)]
        globals()[f"{name}"] = xDET
        mass2 = mass = xDET['Mass (exact)'].round(6).to_numpy()
        data = xDET[['id','Mass (exact)']]
    #
        ##########################################     x x    
        ##                              MD MATCHES
        ##########################################
        md_matches = namedtuple('md_matches', 'md_matches hits')
        new_list = []
        for md in mdTRAFOS:
            for element in mass:
                if element+md in mass2:
                    tmp_new_list = md_matches(element, element+md)
                    new_list.append(tmp_new_list)
        #
        matches = np.array(new_list).reshape(len(new_list),2)
        matches = matches[np.argsort(matches[:,0])]
        sources = data.rename(columns={'id': 'Source', 'Mass (exact)': 'Mass (source)'})
        targets = data.rename(columns={'id': 'Target', 'Mass (exact)': 'Mass (target)'})
        matches2 = pd.DataFrame({'Mass (source)': matches[:, 0], 'Mass (target)': matches[:, 1]})
        source_match = matches2.merge(sources, how='left', on=['Mass (source)'])
        target_match = source_match.merge(targets, how='left', on=['Mass (target)'])
        target_match['Mass difference'] = mass_diff = np.around(target_match['Mass (target)'] - target_match['Mass (source)'],6)
        xxxx = xTRAFOS[['Mass (exact)','Molecular Formula']].rename(columns={'Mass (exact)': 'Mass difference'})
        x = target_match.merge(xxxx, how='left', on=['Mass difference'])
        x['type'] = type = np.full((len(mass_diff),1), 'Undirected')
        x['Label'] = Label = np.full((len(mass_diff),1), 'x')
        xxxx = xTRAFOS[['Mass (exact)','trafoWeights']]
        x2 = x.join(xxxx.set_index(['Mass (exact)'], verify_integrity=True ),on=['Mass difference'], how='left').rename(columns={'trafoWeights': 'Weight'}).sort_values(by=['Source'])
        x2['Mass difference (frequency)'] = x2['Mass difference'].map(dict(x2['Mass difference'].value_counts()))
        #
        xEDGES0 = x2
        #
        md_freq = x2[['Molecular Formula','Mass difference (frequency)']]
        md_freq = md_freq.drop_duplicates(subset=['Molecular Formula'], keep='first')
        md_freq = md_freq.sort_values(by=['Mass difference (frequency)'], ascending=False)
        print('\n\n\n\n')
        print(name+', '+str(len(xNODES0))+' nodes,', str(len(xEDGES0))+' edges')
        print(md_freq)
        print('\n\n\n\n')
    #
        ##########################################     x x    
        ##                             NETWORK ANA
        ##########################################
        G0 = nx.from_pandas_edgelist(xEDGES0, 'Source', 'Target', create_using=nx.Graph())
        G0cc = list(nx.connected_components(G0))
        d = {name:k for k,comp in enumerate(G0cc) for name in comp}               #dict(enumerate(G0cc))
        df_comp = pd.DataFrame.from_dict(d, orient='index', columns=['component']).rename_axis('id').reset_index()
        nodes2 = xNODES0.merge(df_comp, how='left', on='id')
        nans = nodes2[nodes2.isna().any(axis=1)]
        xNODES0 = nodes2[~nodes2['component'].isnull()]
        #
        df_source = xEDGES0.rename(columns={'Source':'id'})
        edges2 = df_source.merge(df_comp, how='left', on='id')
        edges2 = edges2.rename(columns={'component':'component source','id':'Source'})
        df_target = edges2.rename(columns={'Target':'id'})
        edges3 = df_target.merge(df_comp, how='left', on='id')
        edges3 = edges3.rename(columns={'component':'component target','id':'Target','mass_difference':'mass difference','molecular_formula':'molecular formula'})
        #
        df_comp_freq = pd.DataFrame(xNODES0['component'].value_counts()).rename(columns={'component': 'freq_component'})
        xNODES0['df_comp_freq'] = xNODES0['component'].map(dict(xNODES0['component'].value_counts()))
        df_comp_freq_norm = pd.DataFrame(xNODES0['component'].value_counts(normalize=True)).rename(columns={'component': 'freq_component_norm [0-1]'})
        df_comp_freq_comb = pd.concat([df_comp_freq,df_comp_freq_norm], axis=1)
        print('\n\n\n\n')
        print(name+' (all components)')
        print(df_comp_freq_comb)
        print('\n\n\n\n')
        #
        FILTER_COMPONENTS  =        list(df_comp_freq_comb.index)[:nComponents]
        xEDGES = edges3[(edges3['component source'].isin(FILTER_COMPONENTS))]
        xNODES = xNODES0[(xNODES0['component'].isin(FILTER_COMPONENTS))]
        G = nx.from_pandas_edgelist(xEDGES, 'Source', 'Target', create_using=nx.Graph())
        a = pd.DataFrame(G.nodes(data = True)).rename(columns={0:'id'}).drop([1], axis=1)
        a2 = a.join(xNODES.set_index('id'), on='id')
        a2['id_G0_nodes'] = range(1,1+len(a2))
        xNODES = a2
    #
        degrees = pd.DataFrame(G.degree(), columns=["id", "Degree"])
        degrees['id'] = degrees['id'].astype(int)
        xNODES = xNODES.join(degrees.set_index('id'), on='id') 
        xNODES['Degree counts'] = xNODES['Degree'].map(dict(xNODES['Degree'].value_counts()))
    #
    ##    bc = nx.betweenness_centrality(G, normalized=True)                                                                                                                                                                                             ## takes a while....
    ##    bc2 = pd.DataFrame.from_dict(bc, orient='index', columns=['betCen']).rename_axis('id').reset_index()
    ##    xNODES = xNODES.merge(bc2, how='left', on='id')
    #
        colors = mcp.gen_color(cmap='gist_rainbow',n=len(xNODES[color_type].unique()))                    ##  cmap  AUTUMN tab20b coolwarm gist_rainbow tab20c brg      https://matplotlib.org/stable/tutorials/colors/colormaps.html
        ##colors = ['#15B01A',    'blue', 'orange',     'lightgray',   'red', 'green']
        samples0 = xNODES[color_type].unique().tolist()
        try:
            samples = sorted(samples0, key=lambda x: int("".join([i for i in x if i.isdigit()])))
        except:
            samples = sorted(samples0)
        #
        sample_colors = pd.DataFrame({color_type: samples, 'color': colors})
        xNODES = xNODES.merge(sample_colors, how='left', on=color_type)
        xNODES = xNODES.sort_values(by=['id_G0_nodes'], ascending=True)                              ## re-ordering/re-indexing according to graph G0 (id_G0_nodes)
        node_colors = xNODES['color'].tolist()
    #
        xNODES = xNODES.sort_values(by=['id_G0_nodes'], ascending=True)
        xNODES['node_size'] = (xNODES[node_size] - np.min(xNODES[node_size])) / (np.max(xNODES[node_size]) - np.min(xNODES[node_size]))
        node_sizes = xNODES['node_size'].tolist()
        node_sizes = [i * 1e2 for i in node_sizes]


    ###########################
    ###     CLUSTER TRANSITIONS                              fig 2: min_set     directions between nh3_poor to medium, etc.. 
    ###         WITH DIRECTIONS
    ###########################

    ###~~~~~~~~~~~~~~~~~~~~~~~

    FROM_TO = 'Ice composition H2O_CH3OH_NH3'

    ###~~~~~~~~~~~~~~~~~~~~~~~

    df_samples = xNODES[['id', FROM_TO]]
    df_edges4 = xEDGES.rename(columns={'Source':'id'})
    df_edges4 = df_edges4.merge(df_samples, how='left', on='id')
    df_edges4 = df_edges4.rename(columns={FROM_TO: FROM_TO+str('_Source'), 'id':'Source'})
    df_edges5 = df_edges4.rename(columns={'Target':'id'})
    df_edges5 = df_edges5.merge(df_samples, how='left', on='id')
    df_edges5 = df_edges5.rename(columns={FROM_TO: FROM_TO+str('_Target'), 'id':'Target'})


    ###~~~~~~~~~~~~~~~~~~~~~~~
    #SAMPLE_START = '3_1_0.2'
    #SAMPLE_END   = '3_1_1'

    ###SAMPLE_START = '3_1_1'
    ###SAMPLE_END   = '10_1_1'

    ###SAMPLE_START = '10_1_1'
    ###SAMPLE_END   = '3_1_5'
    ###~~~~~~~~~~~~~~~~~~~~~~~



    df_edges6 = df_edges5[(df_edges5[FROM_TO+str('_Source')] == SAMPLE_START) & (df_edges5[FROM_TO+str('_Target')] == SAMPLE_END)]
    df_edges6_backward= df_edges5[(df_edges5[FROM_TO+str('_Source')] == SAMPLE_END) & (df_edges5[FROM_TO+str('_Target')] == SAMPLE_START)]
    products6  = df_edges6['Mass (target)'] - df_edges6['Mass (source)']
    products6_backward  = df_edges6_backward['Mass (target)'] - df_edges6_backward['Mass (source)']

    ###trafo_names = ['H2', 'CO', 'O', 'H3N']           ### automatize later
    ###trafo_names = df_edges['molecular_formula'].unique().tolist()       ## doesn't work.. maybe because of order below

    trafo_names = ['H2', 'O', 'CO', 'H3N', 'CHN', 'CH3N', 'H2O', 'CO2', 'CH2O', 'CHNO']

    for x in trafo_names:
        #for x in trafos:
        TRAFO = x
        globals()['df_edges6_%s' % x] = df_edges6[(df_edges6['Molecular Formula'].str.strip() == TRAFO)]
        globals()['products6_%s' % x]  = globals()['df_edges6_%s' % x]['Mass (target)'] - globals()['df_edges6_%s' % x]['Mass (source)']   
        globals()['df_edges6_backward_%s' % x] = df_edges6_backward[(df_edges6_backward['Molecular Formula'].str.strip() == TRAFO)]
        globals()['products6_backward_%s' % x] = globals()['df_edges6_backward_%s' % x]['Mass (target)'] - globals()['df_edges6_backward_%s' % x]['Mass (source)']

    #forward = [products6_H2.shape[0], products6_O.shape[0], products6_CO.shape[0], products6_NH3.shape[0]]
    #backward = [products6_backward_H2.shape[0], products6_backward_O.shape[0], products6_backward_CO.shape[0], products6_backward_NH3.shape[0]]
    #df = pd.DataFrame({'Forward': forward,  'Backward': backward}, index=trafos)

    #df = pd.DataFrame({'Forward+Backward': np.add(forward, backward), 'Forward': forward,  'Backward': backward}, index=trafos)

    forward = [products6_H2.shape[0], products6_O.shape[0], products6_CO.shape[0], products6_H3N.shape[0], products6_CHN.shape[0], products6_CH3N.shape[0], products6_H2O.shape[0], products6_CO2.shape[0], products6_CH2O.shape[0], products6_CHNO.shape[0]]               ##  to be generalized... xx     
    backward = [products6_backward_H2.shape[0], products6_backward_O.shape[0], products6_backward_CO.shape[0], products6_backward_H3N.shape[0], products6_backward_CHN.shape[0], products6_backward_CH3N.shape[0], products6_backward_H2O.shape[0], products6_backward_CO2.shape[0], products6_backward_CH2O.shape[0], products6_backward_CHNO.shape[0]]
    df = pd.DataFrame({'Forward': forward,  'Backward': backward}, index=trafo_names)

    fontsize = 9
    ###plt.figure()
    ###df.plot.bar(rot=0, stacked=True)
    ###df.plot(kind='bar', rot=0,  width=.8, color={"Forward+Backward": "lightgray", "Forward": "black", 'Backward': 'w'},  edgecolor='black')

    df.plot(kind='bar', rot=0,  width=.8, color={"Forward": "black", 'Backward': 'w'},  edgecolor='black', stacked=True, fontsize=fontsize)
    plt.title(SAMPLE_START+r'$\rightarrow$'+SAMPLE_END+'    ('+str(FILTER_COMPONENTS)+', Component ' +str(FILTER_COMPONENTS)+')', fontsize=fontsize+2)
    # what is name2???

    plt.xticks()
    plt.yticks()

    plt.xlabel('Transformations', fontsize=fontsize+2)
    plt.ylabel('Frequency', fontsize=fontsize+2)
    plt.legend(fontsize=fontsize)

    plt.savefig(name+'_Component'+str(FILTER_COMPONENTS)+'_'+SAMPLE_START+SAMPLE_END+'    trafo_bar_chart'+'.png')

    plt.show()


# In[7]:


# FIGURE 1
# (Printing out First 2 Component - w/Ice Composition Coloring)

trafos = ['H2','O','CO','NH3']
nComponents = 2

mol2net_layout(trafos, nComponents)


# In[11]:


# FIGURE II, PART I: '3_1_0.2' --> '3_1_1'

mol2net_clust_trafo('3_1_0.2', '3_1_1')


# In[12]:


# FIGURE 2 (Cluster Transformations) - PART II: '3_1_1' --> '10_1_1'

mol2net_clust_trafo('3_1_1', '10_1_1')


# In[13]:


# FIGURE 2 (Cluster Transformations) - PART III: '10_1_1' --> '3_1_5'

mol2net_clust_trafo('10_1_1', '3_1_5')


# In[14]:


# FIGURE 3A - Splitted Minimal Transformation Set
# PART I: Minimal Set - H2

trafos = ['O','CO','NH3']

mol2net_layout(trafos, 5)


# In[15]:


# FIGURE 3A - Splitted Minimal Transformation Set
# PART II: Minimal Set - O

trafos = ['H2', 'CO','NH3']

mol2net_layout(trafos, 5)


# In[16]:


# FIGURE 3A - Splitted Minimal Transformation Set
# PART III: Minimal Set - CO

trafos = ['H2','O','NH3']

mol2net_layout(trafos, 5)


# In[15]:


# FIGURE 3A - Splitted Minimal Transformation Set
# PART IV: Minimal Set - NH3

trafos = ['H2','O','CO']

mol2net_layout(trafos, 5)


# In[10]:


trafos          =       ['H2','O','CO','NH3']
##trafoWeights    =       [ 1,   5,   1,   2]
trafoWeights    =       [1]*len(trafos)
columnNames     =       ['molecular_formula','abundance_int']             ##  for  molFormula  &  abundance/intensity
sample          =       'Ice composition H2O_CH3OH_NH3'
nComponents     =       5
only_uniques    =       'yes'
color_type      =       sample
node_size       =       'Degree'

######################################
##                              TRAFOS
######################################
cf = re.compile('([A-Z][a-z]?)(\d?\d?\d?)')
all_elements = []
for xxxx in trafos:
    for el, count in re.findall(cf, xxxx):
        all_elements += [el]

atoms = set(all_elements)
counts = [[""] + trafos]
for e in atoms:
    count = [e]
    for xxxx in trafos:    
        d = dict(re.findall(cf, xxxx))
        n = d.get(e, 0)
        if n == '': n = 1
        count += [int(n)]
    counts += [count]

A = np.array(counts)
A_t = (A.transpose())

x2 = pd.DataFrame(A_t)
new_header = x2.iloc[0]
x2 = x2[1:]
x2.columns = new_header
x2 = x2.rename(columns={'': 'Molecular Formula'})
for i in ['H', 'C', 'N', 'O', 'S']:
    if i in x2.columns:
        x2[i] = x2[i].astype(int)
    else:
        x2[i] = 0    

h = x2['H'].to_numpy()
c = x2['C'].to_numpy()
n = x2['N'].to_numpy()    
o = x2['O'].to_numpy()
s = x2['S'].to_numpy()
elements = np.transpose(np.array([h, c, n, o, s]))
massExact = np.dot(elements, masses)
x2['Mass (exact)'] = massExact.round(6)
weights = pd.DataFrame(trafoWeights,columns = ['trafoWeights'])
weights['trafos'] = trafos
x3 = x2.join(weights.set_index(['trafos'], verify_integrity=True ),on=['Molecular Formula'], how='left')

xTRAFOS = x3
mdTRAFOS = xTRAFOS['Mass (exact)'].round(6).to_numpy()

######################################
##                          DETECTIONS
######################################
path = os.getcwd()
files = glob.glob(os.path.join("*.tsv"))
d = {}
for f in files:
    key = f
    name = f.replace(".tsv", "")
    x = globals()[f"orig_{name}"] = pd.read_csv(f, sep='\t', skiprows=0)
    if 'id' in x.columns:
        pass
    else:
        x.insert(0, 'id', range(1, 1 + len(x)))
#    
    try:
        x = x[['id',sample,columnNames[0],columnNames[1]]].rename(columns={columnNames[0]: 'Molecular Formula',columnNames[1]:'Intensity'})
    except:
        x = x[['id',columnNames[0],columnNames[1]]].rename(columns={columnNames[0]: 'Molecular Formula',columnNames[1]:'Intensity'})
#
    molFormulas = x['Molecular Formula']
    a = []
    for i in molFormulas:
        tmp_a = chemparse.parse_formula(i)
        a.append(tmp_a)
#
    x_a = pd.DataFrame(a).fillna(0)
    x2 = pd.concat([x,x_a], axis=1)
#
    if 'H' in x2.columns:
        x2['H'] = x2['H'].astype(int)
    else:
        x2['H'] = 0    
#
    if 'C' in x2.columns:
        x2['C'] = x2['C'].astype(int)
    else:
        x2['C'] = 0    
#
    if 'N' in x2.columns:
        x2['N'] = x2['N'].astype(int)
    else:
        x2['N'] = 0    
#
    if 'O' in x2.columns:
        x2['O'] = x2['O'].astype(int)
    else:
        x2['O'] = 0    
#
    if 'S' in x2.columns:
        x2['S'] = x2['S'].astype(int)
    else:
        x2['S'] = 0    
#
    h = x2['H'].to_numpy()
    c = x2['C'].to_numpy()
    o = x2['O'].to_numpy()
    n = x2['N'].to_numpy()    
    s = x2['S'].to_numpy()
    elements = np.transpose(np.array([h, c, n, o, s]))
    massExact = np.dot(elements, masses)
    x2['Mass (exact)'] = massExact.round(6)
    x2['mu'] = mu = ( (h*valence_h + c*valence_c + n*valence_n + o*valence_o + s*valence_s) / 2 ) - (h + c + o + n + s) + 1
    x2['DBE'] = dbe = c - (h/2) + (n/2) + 1
    x2['HC'] = h/c
    x2['NC'] = n/c
    x2['OC'] = o/c
    x2['SC'] = s/c
    x2['FILTER'] =  1 * ( (mu >= 0) )
    atom_list = ['H','C','N','O','S']
    for i in range(len(atom_list)-1):
        x2['Molecular Formula'] = x2['Molecular Formula'].replace({atom_list[i]+'0'}, {atom_list[i]}, regex=True)
        x2['Molecular Formula'] = x2['Molecular Formula'].replace({atom_list[i]+'1'+atom_list[i+1]}, {atom_list[i]}, regex=True)
        x2['Molecular Formula'] = x2['Molecular Formula'].replace({'S'+'0'}, {''}, regex=True)
        x2['Molecular Formula'] = x2['Molecular Formula'].replace({'S'+'1'}, {'S'}, regex=True)
#
    x2['Mass (frequency)'] = x2['Mass (exact)'].map(dict(x2['Mass (exact)'].value_counts()))
    try:
        if only_uniques == 'yes':
            x3 = x2.drop_duplicates(subset=['Mass (exact)'], keep='first')
        else:
            pass
    except:
        x3 = x2
#
    xNODES0 = xDET = x3[(x3['FILTER']==1)]
    globals()[f"{name}"] = xDET
    mass2 = mass = xDET['Mass (exact)'].round(6).to_numpy()
    data = xDET[['id','Mass (exact)']]
#
    ##########################################     x x    
    ##                              MD MATCHES
    ##########################################
    md_matches = namedtuple('md_matches', 'md_matches hits')
    new_list = []
    for md in mdTRAFOS:
        for element in mass:
            if element+md in mass2:
                tmp_new_list = md_matches(element, element+md)
                new_list.append(tmp_new_list)
    #
    matches = np.array(new_list).reshape(len(new_list),2)
    matches = matches[np.argsort(matches[:,0])]
    sources = data.rename(columns={'id': 'Source', 'Mass (exact)': 'Mass (source)'})
    targets = data.rename(columns={'id': 'Target', 'Mass (exact)': 'Mass (target)'})
    matches2 = pd.DataFrame({'Mass (source)': matches[:, 0], 'Mass (target)': matches[:, 1]})
    source_match = matches2.merge(sources, how='left', on=['Mass (source)'])
    target_match = source_match.merge(targets, how='left', on=['Mass (target)'])
    target_match['Mass difference'] = mass_diff = np.around(target_match['Mass (target)'] - target_match['Mass (source)'],6)
    xxxx = xTRAFOS[['Mass (exact)','Molecular Formula']].rename(columns={'Mass (exact)': 'Mass difference'})
    x = target_match.merge(xxxx, how='left', on=['Mass difference'])
    x['type'] = type = np.full((len(mass_diff),1), 'Undirected')
    x['Label'] = Label = np.full((len(mass_diff),1), 'x')
    xxxx = xTRAFOS[['Mass (exact)','trafoWeights']]
    x2 = x.join(xxxx.set_index(['Mass (exact)'], verify_integrity=True ),on=['Mass difference'], how='left').rename(columns={'trafoWeights': 'Weight'}).sort_values(by=['Source'])
    x2['Mass difference (frequency)'] = x2['Mass difference'].map(dict(x2['Mass difference'].value_counts()))
    #
    xEDGES0 = x2
    #
    md_freq = x2[['Molecular Formula','Mass difference (frequency)']]
    md_freq = md_freq.drop_duplicates(subset=['Molecular Formula'], keep='first')
    md_freq = md_freq.sort_values(by=['Mass difference (frequency)'], ascending=False)
#     print('\n\n\n\n')
#     print(name+', '+str(len(xNODES0))+' nodes,', str(len(xEDGES0))+' edges')
#     print(md_freq)
#     print('\n\n\n\n')
#
    ##########################################     x x    
    ##                             NETWORK ANA
    ##########################################
    G0 = nx.from_pandas_edgelist(xEDGES0, 'Source', 'Target', create_using=nx.Graph())
    G0cc = list(nx.connected_components(G0))
    d = {name:k for k,comp in enumerate(G0cc) for name in comp}               #dict(enumerate(G0cc))
    df_comp = pd.DataFrame.from_dict(d, orient='index', columns=['component']).rename_axis('id').reset_index()
    nodes2 = xNODES0.merge(df_comp, how='left', on='id')
    nans = nodes2[nodes2.isna().any(axis=1)]
    xNODES0 = nodes2[~nodes2['component'].isnull()]
    #
    df_source = xEDGES0.rename(columns={'Source':'id'})
    edges2 = df_source.merge(df_comp, how='left', on='id')
    edges2 = edges2.rename(columns={'component':'component source','id':'Source'})
    df_target = edges2.rename(columns={'Target':'id'})
    edges3 = df_target.merge(df_comp, how='left', on='id')
    edges3 = edges3.rename(columns={'component':'component target','id':'Target','mass_difference':'mass difference','molecular_formula':'molecular formula'})
    #
    df_comp_freq = pd.DataFrame(xNODES0['component'].value_counts()).rename(columns={'component': 'freq_component'})
    xNODES0['df_comp_freq'] = xNODES0['component'].map(dict(xNODES0['component'].value_counts()))
    df_comp_freq_norm = pd.DataFrame(xNODES0['component'].value_counts(normalize=True)).rename(columns={'component': 'freq_component_norm [0-1]'})
    df_comp_freq_comb = pd.concat([df_comp_freq,df_comp_freq_norm], axis=1)
#     print('\n\n\n\n')
#     print(name+' (all components)')
#     print(df_comp_freq_comb)
#     print('\n\n\n\n')
    #
    FILTER_COMPONENTS  =        list(df_comp_freq_comb.index)[:nComponents]
    xEDGES = edges3[(edges3['component source'].isin(FILTER_COMPONENTS))]
    xNODES = xNODES0[(xNODES0['component'].isin(FILTER_COMPONENTS))]
    G = nx.from_pandas_edgelist(xEDGES, 'Source', 'Target', create_using=nx.Graph())
    a = pd.DataFrame(G.nodes(data = True)).rename(columns={0:'id'}).drop([1], axis=1)
    a2 = a.join(xNODES.set_index('id'), on='id')
    a2['id_G0_nodes'] = range(1,1+len(a2))
    xNODES = a2
#
    degrees = pd.DataFrame(G.degree(), columns=["id", "Degree"])
    degrees['id'] = degrees['id'].astype(int)
    xNODES = xNODES.join(degrees.set_index('id'), on='id') 
    xNODES['Degree counts'] = xNODES['Degree'].map(dict(xNODES['Degree'].value_counts()))
#
##    bc = nx.betweenness_centrality(G, normalized=True)                                                                                                                                                                                             ## takes a while....
##    bc2 = pd.DataFrame.from_dict(bc, orient='index', columns=['betCen']).rename_axis('id').reset_index()
##    xNODES = xNODES.merge(bc2, how='left', on='id')
#
    colors = mcp.gen_color(cmap='gist_rainbow',n=len(xNODES[color_type].unique()))                    ##  cmap  AUTUMN tab20b coolwarm gist_rainbow tab20c brg      https://matplotlib.org/stable/tutorials/colors/colormaps.html
    ##colors = ['#15B01A',    'blue', 'orange',     'lightgray',   'red', 'green']
    samples0 = xNODES[color_type].unique().tolist()
    try:
        samples = sorted(samples0, key=lambda x: int("".join([i for i in x if i.isdigit()])))
    except:
        samples = sorted(samples0)
    #
    sample_colors = pd.DataFrame({color_type: samples, 'color': colors})
    xNODES = xNODES.merge(sample_colors, how='left', on=color_type)
    xNODES = xNODES.sort_values(by=['id_G0_nodes'], ascending=True)                              ## re-ordering/re-indexing according to graph G0 (id_G0_nodes)
    node_colors = xNODES['color'].tolist()
#
    xNODES = xNODES.sort_values(by=['id_G0_nodes'], ascending=True)
    xNODES['node_size'] = (xNODES[node_size] - np.min(xNODES[node_size])) / (np.max(xNODES[node_size]) - np.min(xNODES[node_size]))
    node_sizes = xNODES['node_size'].tolist()
    node_sizes = [i * 1e2 for i in node_sizes]



#######################################################
#######                                   component map
#######################################################

"""
fontsize=20
color_type = 'component'
carac = xNODES0[['id', color_type]]
carac = carac.set_index('id')
carac = carac.reindex(G.nodes())
carac[color_type]=pd.Categorical(carac[color_type])
carac[color_type].cat.codes
nodes = G.nodes()
plt.figure()
plt.title(color_type, wrap=True, fontsize=20)
nx.draw(G, pos = nx.nx_pydot.graphviz_layout(G), with_labels=False, node_color=carac[color_type].cat.codes, cmap=plt.cm.jet, node_size=3, width=.1, font_size=10)
#pos = nx.spring_layout(G)
#pos = nx.fruchterman_reingold_layout(G)
pos = nx.nx_pydot.graphviz_layout(G)
ec = nx.draw_networkx_edges(G, pos, alpha=0.2)
#    nc = nx.draw_networkx_nodes(G, pos, nodelist=nodes, node_color=carac[color_type], with_labels=False, node_size=5, cmap=plt.cm.jet) 
nc = nx.draw_networkx_nodes(G, pos, nodelist=nodes, node_color=carac[color_type], node_size=3, cmap=plt.cm.jet) 
##plt.colorbar(nc)

cb = plt.colorbar(nc, orientation='vertical').set_label(label=color_type, size=fontsize+2)
nc.figure.axes[0].tick_params(axis="both", labelsize=21)           ## change the label size
nc.figure.axes[1].tick_params(axis="y", labelsize=21)              ## change the tick label size of colorbar

plt.axis('off')
#plt.savefig(name+', Comp'+str(FILTER_COMPONENTS)+'    '+color_type+'    network'+'.png')
#plt.savefig('Supporting Figure 1 - Network Representation'+'.png')
plt.show()

"""

    

#################################
####          COMPONENT HISTOGRAM
#################################
    
plt.figure()

#pd.DataFrame(xNODES0['component'].value_counts(normalize=True)).sort_index().plot(kind='bar', rot=0, ylabel='', legend=False, color='black', width=0.8)
pd.DataFrame(xNODES0['component'].value_counts(normalize=False)).plot(kind='bar', rot=0, ylabel='', legend=False, color='black', width=0.7)

plt.title('Minimal Set\n [H2], [O], [CO], [H3N]', wrap=True)          

plt.xticks(np.arange(0, len(df_comp_freq), 1), fontsize=5)                      ## ice
##plt.xticks(np.arange(0, len(df_comp_freq), 100), fontsize=15)                      ## paris

plt.yticks(fontsize=10)
plt.yscale('log')

plt.xlabel('Component number', fontsize=10)
plt.ylabel('Frequency (log scale, not normalized)', fontsize=15)
#plt.ylabel('Frequency (log scale)')


#plt.savefig('Supporting Figure 1 - Frequency Histogram'+'.png')
plt.show()

#print(pd.DataFrame(xNODES0['component'].value_counts(normalize=False)).sort_index())
#print(xNODES0)
#print(xNODES[color_type].unique())


# In[13]:


print(xNODES['component'].unique())


# In[7]:


print(df_comp_freq_comb)


# In[128]:


trafos          =       ['H2','O','CO','NH3']
##trafoWeights    =       [ 1,   5,   1,   2]
trafoWeights    =       [1]*len(trafos)
columnNames     =       ['molecular_formula','abundance_int']             ##  for  molFormula  &  abundance/intensity
sample          =       'Ice composition H2O_CH3OH_NH3'
nComponents     =       2
only_uniques    =       'yes'
color_type      =       sample
node_size       =       'Degree'

######################################
##                              TRAFOS
######################################
cf = re.compile('([A-Z][a-z]?)(\d?\d?\d?)')
all_elements = []
for xxxx in trafos:
    for el, count in re.findall(cf, xxxx):
        all_elements += [el]

atoms = set(all_elements)
counts = [[""] + trafos]
for e in atoms:
    count = [e]
    for xxxx in trafos:    
        d = dict(re.findall(cf, xxxx))
        n = d.get(e, 0)
        if n == '': n = 1
        count += [int(n)]
    counts += [count]

A = np.array(counts)
A_t = (A.transpose())

x2 = pd.DataFrame(A_t)
new_header = x2.iloc[0]
x2 = x2[1:]
x2.columns = new_header
x2 = x2.rename(columns={'': 'Molecular Formula'})
for i in ['H', 'C', 'N', 'O', 'S']:
    if i in x2.columns:
        x2[i] = x2[i].astype(int)
    else:
        x2[i] = 0    

h = x2['H'].to_numpy()
c = x2['C'].to_numpy()
n = x2['N'].to_numpy()    
o = x2['O'].to_numpy()
s = x2['S'].to_numpy()
elements = np.transpose(np.array([h, c, n, o, s]))
massExact = np.dot(elements, masses)
x2['Mass (exact)'] = massExact.round(6)
weights = pd.DataFrame(trafoWeights,columns = ['trafoWeights'])
weights['trafos'] = trafos
x3 = x2.join(weights.set_index(['trafos'], verify_integrity=True ),on=['Molecular Formula'], how='left')

xTRAFOS = x3
mdTRAFOS = xTRAFOS['Mass (exact)'].round(6).to_numpy()

######################################
##                          DETECTIONS
######################################
path = os.getcwd()
files = glob.glob(os.path.join("*.tsv"))
d = {}
for f in files:
    key = f
    name = f.replace(".tsv", "")
    x = globals()[f"orig_{name}"] = pd.read_csv(f, sep='\t', skiprows=0)
    if 'id' in x.columns:
        pass
    else:
        x.insert(0, 'id', range(1, 1 + len(x)))
#    
    try:
        x = x[['id',sample,columnNames[0],columnNames[1]]].rename(columns={columnNames[0]: 'Molecular Formula',columnNames[1]:'Intensity'})
    except:
        x = x[['id',columnNames[0],columnNames[1]]].rename(columns={columnNames[0]: 'Molecular Formula',columnNames[1]:'Intensity'})
#
    molFormulas = x['Molecular Formula']
    a = []
    for i in molFormulas:
        tmp_a = chemparse.parse_formula(i)
        a.append(tmp_a)
#
    x_a = pd.DataFrame(a).fillna(0)
    x2 = pd.concat([x,x_a], axis=1)
#
    if 'H' in x2.columns:
        x2['H'] = x2['H'].astype(int)
    else:
        x2['H'] = 0    
#
    if 'C' in x2.columns:
        x2['C'] = x2['C'].astype(int)
    else:
        x2['C'] = 0    
#
    if 'N' in x2.columns:
        x2['N'] = x2['N'].astype(int)
    else:
        x2['N'] = 0    
#
    if 'O' in x2.columns:
        x2['O'] = x2['O'].astype(int)
    else:
        x2['O'] = 0    
#
    if 'S' in x2.columns:
        x2['S'] = x2['S'].astype(int)
    else:
        x2['S'] = 0    
#
    h = x2['H'].to_numpy()
    c = x2['C'].to_numpy()
    o = x2['O'].to_numpy()
    n = x2['N'].to_numpy()    
    s = x2['S'].to_numpy()
    elements = np.transpose(np.array([h, c, n, o, s]))
    massExact = np.dot(elements, masses)
    x2['Mass (exact)'] = massExact.round(6)
    x2['mu'] = mu = ( (h*valence_h + c*valence_c + n*valence_n + o*valence_o + s*valence_s) / 2 ) - (h + c + o + n + s) + 1
    x2['DBE'] = dbe = c - (h/2) + (n/2) + 1
    x2['HC'] = h/c
    x2['NC'] = n/c
    x2['OC'] = o/c
    x2['SC'] = s/c
    x2['FILTER'] =  1 * ( (mu >= 0) )
    atom_list = ['H','C','N','O','S']
    for i in range(len(atom_list)-1):
        x2['Molecular Formula'] = x2['Molecular Formula'].replace({atom_list[i]+'0'}, {atom_list[i]}, regex=True)
        x2['Molecular Formula'] = x2['Molecular Formula'].replace({atom_list[i]+'1'+atom_list[i+1]}, {atom_list[i]}, regex=True)
        x2['Molecular Formula'] = x2['Molecular Formula'].replace({'S'+'0'}, {''}, regex=True)
        x2['Molecular Formula'] = x2['Molecular Formula'].replace({'S'+'1'}, {'S'}, regex=True)
#
    x2['Mass (frequency)'] = x2['Mass (exact)'].map(dict(x2['Mass (exact)'].value_counts()))
    try:
        if only_uniques == 'yes':
            x3 = x2.drop_duplicates(subset=['Mass (exact)'], keep='first')
        else:
            pass
    except:
        x3 = x2
#
    xNODES0 = xDET = x3[(x3['FILTER']==1)]
    globals()[f"{name}"] = xDET
    mass2 = mass = xDET['Mass (exact)'].round(6).to_numpy()
    data = xDET[['id','Mass (exact)']]
#
    ##########################################     x x    
    ##                              MD MATCHES
    ##########################################
    md_matches = namedtuple('md_matches', 'md_matches hits')
    new_list = []
    for md in mdTRAFOS:
        for element in mass:
            if element+md in mass2:
                tmp_new_list = md_matches(element, element+md)
                new_list.append(tmp_new_list)
    #
    matches = np.array(new_list).reshape(len(new_list),2)
    matches = matches[np.argsort(matches[:,0])]
    sources = data.rename(columns={'id': 'Source', 'Mass (exact)': 'Mass (source)'})
    targets = data.rename(columns={'id': 'Target', 'Mass (exact)': 'Mass (target)'})
    matches2 = pd.DataFrame({'Mass (source)': matches[:, 0], 'Mass (target)': matches[:, 1]})
    source_match = matches2.merge(sources, how='left', on=['Mass (source)'])
    target_match = source_match.merge(targets, how='left', on=['Mass (target)'])
    target_match['Mass difference'] = mass_diff = np.around(target_match['Mass (target)'] - target_match['Mass (source)'],6)
    xxxx = xTRAFOS[['Mass (exact)','Molecular Formula']].rename(columns={'Mass (exact)': 'Mass difference'})
    x = target_match.merge(xxxx, how='left', on=['Mass difference'])
    x['type'] = type = np.full((len(mass_diff),1), 'Undirected')
    x['Label'] = Label = np.full((len(mass_diff),1), 'x')
    xxxx = xTRAFOS[['Mass (exact)','trafoWeights']]
    x2 = x.join(xxxx.set_index(['Mass (exact)'], verify_integrity=True ),on=['Mass difference'], how='left').rename(columns={'trafoWeights': 'Weight'}).sort_values(by=['Source'])
    x2['Mass difference (frequency)'] = x2['Mass difference'].map(dict(x2['Mass difference'].value_counts()))
    #
    xEDGES0 = x2
    #
    md_freq = x2[['Molecular Formula','Mass difference (frequency)']]
    md_freq = md_freq.drop_duplicates(subset=['Molecular Formula'], keep='first')
    md_freq = md_freq.sort_values(by=['Mass difference (frequency)'], ascending=False)
#     print('\n\n\n\n')
#     print(name+', '+str(len(xNODES0))+' nodes,', str(len(xEDGES0))+' edges')
#     print(md_freq)
#     print('\n\n\n\n')
#
    ##########################################     x x    
    ##                             NETWORK ANA
    ##########################################
    G0 = nx.from_pandas_edgelist(xEDGES0, 'Source', 'Target', create_using=nx.Graph())
    G0cc = list(nx.connected_components(G0))
    d = {name:k for k,comp in enumerate(G0cc) for name in comp}               #dict(enumerate(G0cc))
    df_comp = pd.DataFrame.from_dict(d, orient='index', columns=['component']).rename_axis('id').reset_index()
    nodes2 = xNODES0.merge(df_comp, how='left', on='id')
    nans = nodes2[nodes2.isna().any(axis=1)]
    xNODES0 = nodes2[~nodes2['component'].isnull()]
    #
    df_source = xEDGES0.rename(columns={'Source':'id'})
    edges2 = df_source.merge(df_comp, how='left', on='id')
    edges2 = edges2.rename(columns={'component':'component source','id':'Source'})
    df_target = edges2.rename(columns={'Target':'id'})
    edges3 = df_target.merge(df_comp, how='left', on='id')
    edges3 = edges3.rename(columns={'component':'component target','id':'Target','mass_difference':'mass difference','molecular_formula':'molecular formula'})
    #
    df_comp_freq = pd.DataFrame(xNODES0['component'].value_counts()).rename(columns={'component': 'freq_component'})
    xNODES0['df_comp_freq'] = xNODES0['component'].map(dict(xNODES0['component'].value_counts()))
    df_comp_freq_norm = pd.DataFrame(xNODES0['component'].value_counts(normalize=True)).rename(columns={'component': 'freq_component_norm [0-1]'})
    df_comp_freq_comb = pd.concat([df_comp_freq,df_comp_freq_norm], axis=1)
#     print('\n\n\n\n')
#     print(name+' (all components)')
#     print(df_comp_freq_comb)
#     print('\n\n\n\n')
    #
    FILTER_COMPONENTS  =        list(df_comp_freq_comb.index)[:nComponents]
    xEDGES = edges3[(edges3['component source'].isin(FILTER_COMPONENTS))]
    xNODES = xNODES0[(xNODES0['component'].isin(FILTER_COMPONENTS))]
    G = nx.from_pandas_edgelist(xEDGES, 'Source', 'Target', create_using=nx.Graph())
    a = pd.DataFrame(G.nodes(data = True)).rename(columns={0:'id'}).drop([1], axis=1)
    a2 = a.join(xNODES.set_index('id'), on='id')
    a2['id_G0_nodes'] = range(1,1+len(a2))
    xNODES = a2
#
    degrees = pd.DataFrame(G.degree(), columns=["id", "Degree"])
    degrees['id'] = degrees['id'].astype(int)
    xNODES = xNODES.join(degrees.set_index('id'), on='id') 
    xNODES['Degree counts'] = xNODES['Degree'].map(dict(xNODES['Degree'].value_counts()))
#
##    bc = nx.betweenness_centrality(G, normalized=True)                                                                                                                                                                                             ## takes a while....
##    bc2 = pd.DataFrame.from_dict(bc, orient='index', columns=['betCen']).rename_axis('id').reset_index()
##    xNODES = xNODES.merge(bc2, how='left', on='id')
#
    colors = mcp.gen_color(cmap='gist_rainbow',n=len(xNODES[color_type].unique()))                    ##  cmap  AUTUMN tab20b coolwarm gist_rainbow tab20c brg      https://matplotlib.org/stable/tutorials/colors/colormaps.html
    ##colors = ['#15B01A',    'blue', 'orange',     'lightgray',   'red', 'green']
    samples0 = xNODES[color_type].unique().tolist()
    try:
        samples = sorted(samples0, key=lambda x: int("".join([i for i in x if i.isdigit()])))
    except:
        samples = sorted(samples0)
    #
    sample_colors = pd.DataFrame({color_type: samples, 'color': colors})
    xNODES = xNODES.merge(sample_colors, how='left', on=color_type)
    xNODES = xNODES.sort_values(by=['id_G0_nodes'], ascending=True)                              ## re-ordering/re-indexing according to graph G0 (id_G0_nodes)
    node_colors = xNODES['color'].tolist()
#
    xNODES = xNODES.sort_values(by=['id_G0_nodes'], ascending=True)
    xNODES['node_size'] = (xNODES[node_size] - np.min(xNODES[node_size])) / (np.max(xNODES[node_size]) - np.min(xNODES[node_size]))
    node_sizes = xNODES['node_size'].tolist()
    node_sizes = [i * 1e2 for i in node_sizes]
    
    
####################################################
####    CONTINOUS COLOR GRADIENT (e.g. element maps)
####################################################
fontsize=20
l = ['H', 'C', 'N', 'O', 'HC', 'NC', 'OC', 'Mass (exact)', 'Intensity', 'DBE']
#put 'S' and 'SC' away, since they are not needed.


#l = ['Intensity']

id_temp = xNODES['id']
int_temp = np.log10(xNODES['Intensity'])

merged_temp = pd.concat([id_temp, int_temp], axis=1)

for i in l:
   color_type = i

   if(i == 'Intensity'):
       carac = merged_temp
   else:
       carac = carac = xNODES[['id', color_type]]

   #print(carac)

   carac = carac.set_index('id')
   carac = carac.reindex(G.nodes())
   carac[color_type]=pd.Categorical(carac[color_type])
   carac[color_type].cat.codes
   nodes = G.nodes()
   plt.figure()
   if(i == 'Intensity'):
       plt.title('lg_abundance_int', wrap=True, fontsize=25)
   else:
       plt.title(color_type, wrap=True, fontsize=25)
##        nx.draw(G, pos = nx.nx_pydot.graphviz_layout(G), with_labels=False, node_color=carac[color_type].cat.codes, cmap=plt.cm.jet, node_size=3, width=.1, font_size=10)
##        pos = nx.spring_layout(G)
   pos = nx.nx_pydot.graphviz_layout(G)
##        pos = nx.fruchterman_reingold_layout(G)
##        pos = nx.nx_pydot.graphviz_layout(G)
   nx.draw(G, pos, with_labels=False, node_color=node_colors, node_size=2, width=.01, alpha=1)
   ec = nx.draw_networkx_edges(G, pos, width=.01, alpha=0.2)
#    nc = nx.draw_networkx_nodes(G, pos, nodelist=nodes, node_color=carac[color_type], with_labels=False, node_size=5, cmap=plt.cm.jet) 
   nc = nx.draw_networkx_nodes(G, pos, nodelist=nodes, node_color=carac[color_type], node_size=3, cmap=plt.cm.jet) 
##    plt.colorbar(nc)
   cb = plt.colorbar(nc, orientation='vertical')
   nc.figure.axes[0].tick_params(axis="both", labelsize=21)           ## change the label size
   nc.figure.axes[1].tick_params(axis="y", labelsize=21)              ## change the tick label size of colorbar
   plt.axis('off')
   plt.savefig('Supporting Figure 4_' + color_type +'.png')
   plt.show()


# In[24]:


# SUPPORTING FIGURE 7 ** "Minimal Set - X", "Element Maps"

#trafos          =       ['O','CO','NH3'] # Minimal Set - H2, 69 components.

#trafos          =       ['H2','CO','NH3'] # Minimal Set - O, 99 components.

#trafos          =       ['H2','O','NH3'] # Minimal Set - CO, 82 components.

trafos          =       ['H2','O','CO'] # Minimal Set - NH3, 73 components.

##trafoWeights    =       [ 1,   5,   1,   2]
trafoWeights    =       [1]*len(trafos)
columnNames     =       ['molecular_formula','abundance_int']             ##  for  molFormula  &  abundance/intensity
sample          =       'Ice composition H2O_CH3OH_NH3'
nComponents     =       73
only_uniques    =       'yes'
color_type      =       sample
node_size       =       'Degree'

######################################
##                              TRAFOS
######################################
cf = re.compile('([A-Z][a-z]?)(\d?\d?\d?)')
all_elements = []
for xxxx in trafos:
    for el, count in re.findall(cf, xxxx):
        all_elements += [el]

atoms = set(all_elements)
counts = [[""] + trafos]
for e in atoms:
    count = [e]
    for xxxx in trafos:    
        d = dict(re.findall(cf, xxxx))
        n = d.get(e, 0)
        if n == '': n = 1
        count += [int(n)]
    counts += [count]

A = np.array(counts)
A_t = (A.transpose())

x2 = pd.DataFrame(A_t)
new_header = x2.iloc[0]
x2 = x2[1:]
x2.columns = new_header
x2 = x2.rename(columns={'': 'Molecular Formula'})
for i in ['H', 'C', 'N', 'O', 'S']:
    if i in x2.columns:
        x2[i] = x2[i].astype(int)
    else:
        x2[i] = 0    

h = x2['H'].to_numpy()
c = x2['C'].to_numpy()
n = x2['N'].to_numpy()    
o = x2['O'].to_numpy()
s = x2['S'].to_numpy()
elements = np.transpose(np.array([h, c, n, o, s]))
massExact = np.dot(elements, masses)
x2['Mass (exact)'] = massExact.round(6)
weights = pd.DataFrame(trafoWeights,columns = ['trafoWeights'])
weights['trafos'] = trafos
x3 = x2.join(weights.set_index(['trafos'], verify_integrity=True ),on=['Molecular Formula'], how='left')

xTRAFOS = x3
mdTRAFOS = xTRAFOS['Mass (exact)'].round(6).to_numpy()

######################################
##                          DETECTIONS
######################################
path = os.getcwd()
files = glob.glob(os.path.join("*.tsv"))
d = {}
for f in files:
    key = f
    name = f.replace(".tsv", "")
    x = globals()[f"orig_{name}"] = pd.read_csv(f, sep='\t', skiprows=0)
    if 'id' in x.columns:
        pass
    else:
        x.insert(0, 'id', range(1, 1 + len(x)))
#    
    try:
        x = x[['id',sample,columnNames[0],columnNames[1]]].rename(columns={columnNames[0]: 'Molecular Formula',columnNames[1]:'Intensity'})
    except:
        x = x[['id',columnNames[0],columnNames[1]]].rename(columns={columnNames[0]: 'Molecular Formula',columnNames[1]:'Intensity'})
#
    molFormulas = x['Molecular Formula']
    a = []
    for i in molFormulas:
        tmp_a = chemparse.parse_formula(i)
        a.append(tmp_a)
#
    x_a = pd.DataFrame(a).fillna(0)
    x2 = pd.concat([x,x_a], axis=1)
#
    if 'H' in x2.columns:
        x2['H'] = x2['H'].astype(int)
    else:
        x2['H'] = 0    
#
    if 'C' in x2.columns:
        x2['C'] = x2['C'].astype(int)
    else:
        x2['C'] = 0    
#
    if 'N' in x2.columns:
        x2['N'] = x2['N'].astype(int)
    else:
        x2['N'] = 0    
#
    if 'O' in x2.columns:
        x2['O'] = x2['O'].astype(int)
    else:
        x2['O'] = 0    
#
    if 'S' in x2.columns:
        x2['S'] = x2['S'].astype(int)
    else:
        x2['S'] = 0    
#
    h = x2['H'].to_numpy()
    c = x2['C'].to_numpy()
    o = x2['O'].to_numpy()
    n = x2['N'].to_numpy()    
    s = x2['S'].to_numpy()
    elements = np.transpose(np.array([h, c, n, o, s]))
    massExact = np.dot(elements, masses)
    x2['Mass (exact)'] = massExact.round(6)
    x2['mu'] = mu = ( (h*valence_h + c*valence_c + n*valence_n + o*valence_o + s*valence_s) / 2 ) - (h + c + o + n + s) + 1
    x2['DBE'] = dbe = c - (h/2) + (n/2) + 1
    x2['HC'] = h/c
    x2['NC'] = n/c
    x2['OC'] = o/c
    x2['SC'] = s/c
    x2['FILTER'] =  1 * ( (mu >= 0) )
    atom_list = ['H','C','N','O','S']
    for i in range(len(atom_list)-1):
        x2['Molecular Formula'] = x2['Molecular Formula'].replace({atom_list[i]+'0'}, {atom_list[i]}, regex=True)
        x2['Molecular Formula'] = x2['Molecular Formula'].replace({atom_list[i]+'1'+atom_list[i+1]}, {atom_list[i]}, regex=True)
        x2['Molecular Formula'] = x2['Molecular Formula'].replace({'S'+'0'}, {''}, regex=True)
        x2['Molecular Formula'] = x2['Molecular Formula'].replace({'S'+'1'}, {'S'}, regex=True)
#
    x2['Mass (frequency)'] = x2['Mass (exact)'].map(dict(x2['Mass (exact)'].value_counts()))
    try:
        if only_uniques == 'yes':
            x3 = x2.drop_duplicates(subset=['Mass (exact)'], keep='first')
        else:
            pass
    except:
        x3 = x2
#
    xNODES0 = xDET = x3[(x3['FILTER']==1)]
    globals()[f"{name}"] = xDET
    mass2 = mass = xDET['Mass (exact)'].round(6).to_numpy()
    data = xDET[['id','Mass (exact)']]
#
    ##########################################     x x    
    ##                              MD MATCHES
    ##########################################
    md_matches = namedtuple('md_matches', 'md_matches hits')
    new_list = []
    for md in mdTRAFOS:
        for element in mass:
            if element+md in mass2:
                tmp_new_list = md_matches(element, element+md)
                new_list.append(tmp_new_list)
    #
    matches = np.array(new_list).reshape(len(new_list),2)
    matches = matches[np.argsort(matches[:,0])]
    sources = data.rename(columns={'id': 'Source', 'Mass (exact)': 'Mass (source)'})
    targets = data.rename(columns={'id': 'Target', 'Mass (exact)': 'Mass (target)'})
    matches2 = pd.DataFrame({'Mass (source)': matches[:, 0], 'Mass (target)': matches[:, 1]})
    source_match = matches2.merge(sources, how='left', on=['Mass (source)'])
    target_match = source_match.merge(targets, how='left', on=['Mass (target)'])
    target_match['Mass difference'] = mass_diff = np.around(target_match['Mass (target)'] - target_match['Mass (source)'],6)
    xxxx = xTRAFOS[['Mass (exact)','Molecular Formula']].rename(columns={'Mass (exact)': 'Mass difference'})
    x = target_match.merge(xxxx, how='left', on=['Mass difference'])
    x['type'] = type = np.full((len(mass_diff),1), 'Undirected')
    x['Label'] = Label = np.full((len(mass_diff),1), 'x')
    xxxx = xTRAFOS[['Mass (exact)','trafoWeights']]
    x2 = x.join(xxxx.set_index(['Mass (exact)'], verify_integrity=True ),on=['Mass difference'], how='left').rename(columns={'trafoWeights': 'Weight'}).sort_values(by=['Source'])
    x2['Mass difference (frequency)'] = x2['Mass difference'].map(dict(x2['Mass difference'].value_counts()))
    #
    xEDGES0 = x2
    #
    md_freq = x2[['Molecular Formula','Mass difference (frequency)']]
    md_freq = md_freq.drop_duplicates(subset=['Molecular Formula'], keep='first')
    md_freq = md_freq.sort_values(by=['Mass difference (frequency)'], ascending=False)
#     print('\n\n\n\n')
#     print(name+', '+str(len(xNODES0))+' nodes,', str(len(xEDGES0))+' edges')
#     print(md_freq)
#     print('\n\n\n\n')
#
    ##########################################     x x    
    ##                             NETWORK ANA
    ##########################################
    G0 = nx.from_pandas_edgelist(xEDGES0, 'Source', 'Target', create_using=nx.Graph())
    G0cc = list(nx.connected_components(G0))
    d = {name:k for k,comp in enumerate(G0cc) for name in comp}               #dict(enumerate(G0cc))
    df_comp = pd.DataFrame.from_dict(d, orient='index', columns=['component']).rename_axis('id').reset_index()
    nodes2 = xNODES0.merge(df_comp, how='left', on='id')
    nans = nodes2[nodes2.isna().any(axis=1)]
    xNODES0 = nodes2[~nodes2['component'].isnull()]
    #
    df_source = xEDGES0.rename(columns={'Source':'id'})
    edges2 = df_source.merge(df_comp, how='left', on='id')
    edges2 = edges2.rename(columns={'component':'component source','id':'Source'})
    df_target = edges2.rename(columns={'Target':'id'})
    edges3 = df_target.merge(df_comp, how='left', on='id')
    edges3 = edges3.rename(columns={'component':'component target','id':'Target','mass_difference':'mass difference','molecular_formula':'molecular formula'})
    #
    df_comp_freq = pd.DataFrame(xNODES0['component'].value_counts()).rename(columns={'component': 'freq_component'})
    xNODES0['df_comp_freq'] = xNODES0['component'].map(dict(xNODES0['component'].value_counts()))
    df_comp_freq_norm = pd.DataFrame(xNODES0['component'].value_counts(normalize=True)).rename(columns={'component': 'freq_component_norm [0-1]'})
    df_comp_freq_comb = pd.concat([df_comp_freq,df_comp_freq_norm], axis=1)
#     print('\n\n\n\n')
#     print(name+' (all components)')
#     print(df_comp_freq_comb)
#     print('\n\n\n\n')
    #
    FILTER_COMPONENTS  =        list(df_comp_freq_comb.index)[:nComponents]
    xEDGES = edges3[(edges3['component source'].isin(FILTER_COMPONENTS))]
    xNODES = xNODES0[(xNODES0['component'].isin(FILTER_COMPONENTS))]
    G = nx.from_pandas_edgelist(xEDGES, 'Source', 'Target', create_using=nx.Graph())
    a = pd.DataFrame(G.nodes(data = True)).rename(columns={0:'id'}).drop([1], axis=1)
    a2 = a.join(xNODES.set_index('id'), on='id')
    a2['id_G0_nodes'] = range(1,1+len(a2))
    xNODES = a2
#
    degrees = pd.DataFrame(G.degree(), columns=["id", "Degree"])
    degrees['id'] = degrees['id'].astype(int)
    xNODES = xNODES.join(degrees.set_index('id'), on='id') 
    xNODES['Degree counts'] = xNODES['Degree'].map(dict(xNODES['Degree'].value_counts()))
#
##    bc = nx.betweenness_centrality(G, normalized=True)                                                                                                                                                                                             ## takes a while....
##    bc2 = pd.DataFrame.from_dict(bc, orient='index', columns=['betCen']).rename_axis('id').reset_index()
##    xNODES = xNODES.merge(bc2, how='left', on='id')
#
    colors = mcp.gen_color(cmap='gist_rainbow',n=len(xNODES[color_type].unique()))                    ##  cmap  AUTUMN tab20b coolwarm gist_rainbow tab20c brg      https://matplotlib.org/stable/tutorials/colors/colormaps.html
    ##colors = ['#15B01A',    'blue', 'orange',     'lightgray',   'red', 'green']
    samples0 = xNODES[color_type].unique().tolist()
    try:
        samples = sorted(samples0, key=lambda x: int("".join([i for i in x if i.isdigit()])))
    except:
        samples = sorted(samples0)
    #
    sample_colors = pd.DataFrame({color_type: samples, 'color': colors})
    xNODES = xNODES.merge(sample_colors, how='left', on=color_type)
    xNODES = xNODES.sort_values(by=['id_G0_nodes'], ascending=True)                              ## re-ordering/re-indexing according to graph G0 (id_G0_nodes)
    node_colors = xNODES['color'].tolist()
#
    xNODES = xNODES.sort_values(by=['id_G0_nodes'], ascending=True)
    xNODES['node_size'] = (xNODES[node_size] - np.min(xNODES[node_size])) / (np.max(xNODES[node_size]) - np.min(xNODES[node_size]))
    node_sizes = xNODES['node_size'].tolist()
    node_sizes = [i * 1e2 for i in node_sizes]
    
    
####################################################
####    CONTINOUS COLOR GRADIENT (e.g. element maps)
####################################################
fontsize=20
l = ['H', 'C', 'N', 'O', 'HC', 'NC', 'OC', 'Mass (exact)', 'DBE']
#put 'S' and 'SC' away, since they are not needed.
# put 'Intensity' away, since it is not needed.

#l = ['Intensity']

# id_temp = xNODES['id']
# int_temp = np.log10(xNODES['Intensity'])

# merged_temp = pd.concat([id_temp, int_temp], axis=1)

for i in l:
   color_type = i
   
    
   carac = xNODES[['id', color_type]]

   #print(carac)

   carac = carac.set_index('id')
   carac = carac.reindex(G.nodes())
   carac[color_type]=pd.Categorical(carac[color_type])
   carac[color_type].cat.codes
   nodes = G.nodes()
   plt.figure()
   plt.title(color_type, wrap=True, fontsize=25)
##        nx.draw(G, pos = nx.nx_pydot.graphviz_layout(G), with_labels=False, node_color=carac[color_type].cat.codes, cmap=plt.cm.jet, node_size=3, width=.1, font_size=10)
##        pos = nx.spring_layout(G)
   pos = nx.nx_pydot.graphviz_layout(G)
##        pos = nx.fruchterman_reingold_layout(G)
##        pos = nx.nx_pydot.graphviz_layout(G)
   nx.draw(G, pos, with_labels=False, node_color=node_colors, node_size=2, width=.01, alpha=1)
   ec = nx.draw_networkx_edges(G, pos, width=.01, alpha=0.2)
#    nc = nx.draw_networkx_nodes(G, pos, nodelist=nodes, node_color=carac[color_type], with_labels=False, node_size=5, cmap=plt.cm.jet) 
   nc = nx.draw_networkx_nodes(G, pos, nodelist=nodes, node_color=carac[color_type], node_size=3, cmap=plt.cm.jet) 
##    plt.colorbar(nc)
   cb = plt.colorbar(nc, orientation='vertical')
   nc.figure.axes[0].tick_params(axis="both", labelsize=21)           ## change the label size
   nc.figure.axes[1].tick_params(axis="y", labelsize=21)              ## change the tick label size of colorbar
   plt.axis('off')
   plt.savefig('Supporting Figure 7 - ' + color_type + '_minimal_set_NH3' + '.png')
   plt.show()


# In[23]:


print(xNODES0['component'].unique())


# In[16]:


# SUPPORTING FIGURE 8 - Minimal Set

# 5 Components - Layout + Element Maps


#trafos          =       ['O','CO','NH3'] # Minimal Set - H2, 69 components.

#trafos          =       ['H2','CO','NH3'] # Minimal Set - O, 99 components.

#trafos          =       ['H2','O','NH3'] # Minimal Set - CO, 82 components.

trafos          =       ['H2','O','CO'] # Minimal Set - NH3, 73 components.

##trafoWeights    =       [ 1,   5,   1,   2]
trafoWeights    =       [1]*len(trafos)
columnNames     =       ['molecular_formula','abundance_int']             ##  for  molFormula  &  abundance/intensity
sample          =       'Ice composition H2O_CH3OH_NH3'
nComponents     =       5
only_uniques    =       'yes'
color_type      =       sample
node_size       =       'Degree'

######################################
##                              TRAFOS
######################################
cf = re.compile('([A-Z][a-z]?)(\d?\d?\d?)')
all_elements = []
for xxxx in trafos:
    for el, count in re.findall(cf, xxxx):
        all_elements += [el]

atoms = set(all_elements)
counts = [[""] + trafos]
for e in atoms:
    count = [e]
    for xxxx in trafos:    
        d = dict(re.findall(cf, xxxx))
        n = d.get(e, 0)
        if n == '': n = 1
        count += [int(n)]
    counts += [count]

A = np.array(counts)
A_t = (A.transpose())

x2 = pd.DataFrame(A_t)
new_header = x2.iloc[0]
x2 = x2[1:]
x2.columns = new_header
x2 = x2.rename(columns={'': 'Molecular Formula'})
for i in ['H', 'C', 'N', 'O', 'S']:
    if i in x2.columns:
        x2[i] = x2[i].astype(int)
    else:
        x2[i] = 0    

h = x2['H'].to_numpy()
c = x2['C'].to_numpy()
n = x2['N'].to_numpy()    
o = x2['O'].to_numpy()
s = x2['S'].to_numpy()
elements = np.transpose(np.array([h, c, n, o, s]))
massExact = np.dot(elements, masses)
x2['Mass (exact)'] = massExact.round(6)
weights = pd.DataFrame(trafoWeights,columns = ['trafoWeights'])
weights['trafos'] = trafos
x3 = x2.join(weights.set_index(['trafos'], verify_integrity=True ),on=['Molecular Formula'], how='left')

xTRAFOS = x3
mdTRAFOS = xTRAFOS['Mass (exact)'].round(6).to_numpy()

######################################
##                          DETECTIONS
######################################
path = os.getcwd()
files = glob.glob(os.path.join("*.tsv"))
d = {}
for f in files:
    key = f
    name = f.replace(".tsv", "")
    x = globals()[f"orig_{name}"] = pd.read_csv(f, sep='\t', skiprows=0)
    if 'id' in x.columns:
        pass
    else:
        x.insert(0, 'id', range(1, 1 + len(x)))
#    
    try:
        x = x[['id',sample,columnNames[0],columnNames[1]]].rename(columns={columnNames[0]: 'Molecular Formula',columnNames[1]:'Intensity'})
    except:
        x = x[['id',columnNames[0],columnNames[1]]].rename(columns={columnNames[0]: 'Molecular Formula',columnNames[1]:'Intensity'})
#
    molFormulas = x['Molecular Formula']
    a = []
    for i in molFormulas:
        tmp_a = chemparse.parse_formula(i)
        a.append(tmp_a)
#
    x_a = pd.DataFrame(a).fillna(0)
    x2 = pd.concat([x,x_a], axis=1)
#
    if 'H' in x2.columns:
        x2['H'] = x2['H'].astype(int)
    else:
        x2['H'] = 0    
#
    if 'C' in x2.columns:
        x2['C'] = x2['C'].astype(int)
    else:
        x2['C'] = 0    
#
    if 'N' in x2.columns:
        x2['N'] = x2['N'].astype(int)
    else:
        x2['N'] = 0    
#
    if 'O' in x2.columns:
        x2['O'] = x2['O'].astype(int)
    else:
        x2['O'] = 0    
#
    if 'S' in x2.columns:
        x2['S'] = x2['S'].astype(int)
    else:
        x2['S'] = 0    
#
    h = x2['H'].to_numpy()
    c = x2['C'].to_numpy()
    o = x2['O'].to_numpy()
    n = x2['N'].to_numpy()    
    s = x2['S'].to_numpy()
    elements = np.transpose(np.array([h, c, n, o, s]))
    massExact = np.dot(elements, masses)
    x2['Mass (exact)'] = massExact.round(6)
    x2['mu'] = mu = ( (h*valence_h + c*valence_c + n*valence_n + o*valence_o + s*valence_s) / 2 ) - (h + c + o + n + s) + 1
    x2['DBE'] = dbe = c - (h/2) + (n/2) + 1
    x2['HC'] = h/c
    x2['NC'] = n/c
    x2['OC'] = o/c
    x2['SC'] = s/c
    x2['FILTER'] =  1 * ( (mu >= 0) )
    atom_list = ['H','C','N','O','S']
    for i in range(len(atom_list)-1):
        x2['Molecular Formula'] = x2['Molecular Formula'].replace({atom_list[i]+'0'}, {atom_list[i]}, regex=True)
        x2['Molecular Formula'] = x2['Molecular Formula'].replace({atom_list[i]+'1'+atom_list[i+1]}, {atom_list[i]}, regex=True)
        x2['Molecular Formula'] = x2['Molecular Formula'].replace({'S'+'0'}, {''}, regex=True)
        x2['Molecular Formula'] = x2['Molecular Formula'].replace({'S'+'1'}, {'S'}, regex=True)
#
    x2['Mass (frequency)'] = x2['Mass (exact)'].map(dict(x2['Mass (exact)'].value_counts()))
    try:
        if only_uniques == 'yes':
            x3 = x2.drop_duplicates(subset=['Mass (exact)'], keep='first')
        else:
            pass
    except:
        x3 = x2
#
    xNODES0 = xDET = x3[(x3['FILTER']==1)]
    globals()[f"{name}"] = xDET
    mass2 = mass = xDET['Mass (exact)'].round(6).to_numpy()
    data = xDET[['id','Mass (exact)']]
#
    ##########################################     x x    
    ##                              MD MATCHES
    ##########################################
    md_matches = namedtuple('md_matches', 'md_matches hits')
    new_list = []
    for md in mdTRAFOS:
        for element in mass:
            if element+md in mass2:
                tmp_new_list = md_matches(element, element+md)
                new_list.append(tmp_new_list)
    #
    matches = np.array(new_list).reshape(len(new_list),2)
    matches = matches[np.argsort(matches[:,0])]
    sources = data.rename(columns={'id': 'Source', 'Mass (exact)': 'Mass (source)'})
    targets = data.rename(columns={'id': 'Target', 'Mass (exact)': 'Mass (target)'})
    matches2 = pd.DataFrame({'Mass (source)': matches[:, 0], 'Mass (target)': matches[:, 1]})
    source_match = matches2.merge(sources, how='left', on=['Mass (source)'])
    target_match = source_match.merge(targets, how='left', on=['Mass (target)'])
    target_match['Mass difference'] = mass_diff = np.around(target_match['Mass (target)'] - target_match['Mass (source)'],6)
    xxxx = xTRAFOS[['Mass (exact)','Molecular Formula']].rename(columns={'Mass (exact)': 'Mass difference'})
    x = target_match.merge(xxxx, how='left', on=['Mass difference'])
    x['type'] = type = np.full((len(mass_diff),1), 'Undirected')
    x['Label'] = Label = np.full((len(mass_diff),1), 'x')
    xxxx = xTRAFOS[['Mass (exact)','trafoWeights']]
    x2 = x.join(xxxx.set_index(['Mass (exact)'], verify_integrity=True ),on=['Mass difference'], how='left').rename(columns={'trafoWeights': 'Weight'}).sort_values(by=['Source'])
    x2['Mass difference (frequency)'] = x2['Mass difference'].map(dict(x2['Mass difference'].value_counts()))
    #
    xEDGES0 = x2
    #
    md_freq = x2[['Molecular Formula','Mass difference (frequency)']]
    md_freq = md_freq.drop_duplicates(subset=['Molecular Formula'], keep='first')
    md_freq = md_freq.sort_values(by=['Mass difference (frequency)'], ascending=False)
#     print('\n\n\n\n')
#     print(name+', '+str(len(xNODES0))+' nodes,', str(len(xEDGES0))+' edges')
#     print(md_freq)
#     print('\n\n\n\n')
#
    ##########################################     x x    
    ##                             NETWORK ANA
    ##########################################
    G0 = nx.from_pandas_edgelist(xEDGES0, 'Source', 'Target', create_using=nx.Graph())
    G0cc = list(nx.connected_components(G0))
    d = {name:k for k,comp in enumerate(G0cc) for name in comp}               #dict(enumerate(G0cc))
    df_comp = pd.DataFrame.from_dict(d, orient='index', columns=['component']).rename_axis('id').reset_index()
    nodes2 = xNODES0.merge(df_comp, how='left', on='id')
    nans = nodes2[nodes2.isna().any(axis=1)]
    xNODES0 = nodes2[~nodes2['component'].isnull()]
    #
    df_source = xEDGES0.rename(columns={'Source':'id'})
    edges2 = df_source.merge(df_comp, how='left', on='id')
    edges2 = edges2.rename(columns={'component':'component source','id':'Source'})
    df_target = edges2.rename(columns={'Target':'id'})
    edges3 = df_target.merge(df_comp, how='left', on='id')
    edges3 = edges3.rename(columns={'component':'component target','id':'Target','mass_difference':'mass difference','molecular_formula':'molecular formula'})
    #
    df_comp_freq = pd.DataFrame(xNODES0['component'].value_counts()).rename(columns={'component': 'freq_component'})
    xNODES0['df_comp_freq'] = xNODES0['component'].map(dict(xNODES0['component'].value_counts()))
    df_comp_freq_norm = pd.DataFrame(xNODES0['component'].value_counts(normalize=True)).rename(columns={'component': 'freq_component_norm [0-1]'})
    df_comp_freq_comb = pd.concat([df_comp_freq,df_comp_freq_norm], axis=1)
#     print('\n\n\n\n')
#     print(name+' (all components)')
#     print(df_comp_freq_comb)
#     print('\n\n\n\n')
    #
    FILTER_COMPONENTS  =        list(df_comp_freq_comb.index)[:nComponents]
    FILTER_COMPONENTS  = [0,4]
    xEDGES = edges3[(edges3['component source'].isin(FILTER_COMPONENTS))]
    xNODES = xNODES0[(xNODES0['component'].isin(FILTER_COMPONENTS))]
    G = nx.from_pandas_edgelist(xEDGES, 'Source', 'Target', create_using=nx.Graph())
    a = pd.DataFrame(G.nodes(data = True)).rename(columns={0:'id'}).drop([1], axis=1)
    a2 = a.join(xNODES.set_index('id'), on='id')
    a2['id_G0_nodes'] = range(1,1+len(a2))
    xNODES = a2
#
    degrees = pd.DataFrame(G.degree(), columns=["id", "Degree"])
    degrees['id'] = degrees['id'].astype(int)
    xNODES = xNODES.join(degrees.set_index('id'), on='id') 
    xNODES['Degree counts'] = xNODES['Degree'].map(dict(xNODES['Degree'].value_counts()))
#
##    bc = nx.betweenness_centrality(G, normalized=True)                                                                                                                                                                                             ## takes a while....
##    bc2 = pd.DataFrame.from_dict(bc, orient='index', columns=['betCen']).rename_axis('id').reset_index()
##    xNODES = xNODES.merge(bc2, how='left', on='id')
#
    colors = mcp.gen_color(cmap='gist_rainbow',n=len(xNODES[color_type].unique()))                    ##  cmap  AUTUMN tab20b coolwarm gist_rainbow tab20c brg      https://matplotlib.org/stable/tutorials/colors/colormaps.html
    ##colors = ['#15B01A',    'blue', 'orange',     'lightgray',   'red', 'green']
    samples0 = xNODES[color_type].unique().tolist()
    try:
        samples = sorted(samples0, key=lambda x: int("".join([i for i in x if i.isdigit()])))
    except:
        samples = sorted(samples0)
    #
    sample_colors = pd.DataFrame({color_type: samples, 'color': colors})
    xNODES = xNODES.merge(sample_colors, how='left', on=color_type)
    xNODES = xNODES.sort_values(by=['id_G0_nodes'], ascending=True)                              ## re-ordering/re-indexing according to graph G0 (id_G0_nodes)
    node_colors = xNODES['color'].tolist()
#
    xNODES = xNODES.sort_values(by=['id_G0_nodes'], ascending=True)
    xNODES['node_size'] = (xNODES[node_size] - np.min(xNODES[node_size])) / (np.max(xNODES[node_size]) - np.min(xNODES[node_size]))
    node_sizes = xNODES['node_size'].tolist()
    node_sizes = [i * 1e2 for i in node_sizes]
    
    
###########################################
####                                 LAYOUT
###########################################

"""
plt.figure()
plt.title(name, wrap=True, fontsize=12)
    #nx.draw(G, pos = nx.nx_pydot.graphviz_layout(G), with_labels=False, node_color=node_colors, node_size=20, width=.03, font_size=5)
    #nx.draw(G, pos = nx.nx_pydot.graphviz_layout(G), with_labels=False, node_color=node_colors, node_size=node_sizes, width=.05, alpha=1)
    #nx.draw(G, pos = nx.spring_layout(G, scale=1), with_labels=False, node_color=node_colors, node_size=node_sizes, width=.05, alpha=1)
    #nx.draw(G, pos = nx.spring_layout(G, k=.5/math.sqrt(G.order()), scale=10), with_labels=False, node_color=node_colors, node_size=node_sizes, width=.05, alpha=1)
#nx.draw(G, pos = nx.nx_pydot.graphviz_layout(G), with_labels=False, node_color=node_colors, node_size=2, width=.01, alpha=1)
nx.draw(G, pos = nx.nx_pydot.graphviz_layout(G), with_labels=False, node_color=node_colors, node_size=1, width=.05, alpha=1)

for i in range(len(sample_colors)):
    plt.plot([], [], sample_colors.values[i][1], marker='o', markersize=10, label=sample_colors.values[i][0])      


# PRINTING FIGURE 1.

plt.legend()                      ## loc='upper left'
plt.text(.6, -1.1,str(G.number_of_nodes())+' nodes, '+str(G.number_of_edges())+' edges, node size ~ '+str(node_size), fontsize=8, wrap=True)
#plt.text(.6, -1.1,str(G.number_of_nodes())+' nodes, '+str(G.number_of_edges())+' edges', fontsize=8, wrap=True)
plt.savefig('Supporting Figure 8 Layout - NH3'+name+'_Comp'+str(FILTER_COMPONENTS)+'    '+str(color_type)+'    network'+'.png')
plt.show()
"""
    
####################################################
####    CONTINOUS COLOR GRADIENT (e.g. element maps)
####################################################
""""
fontsize=20
l = ['H', 'C', 'N', 'O', 'HC', 'NC', 'OC', 'Mass (exact)', 'DBE']
#put 'S' and 'SC' away, since they are not needed.
# put 'Intensity' away, since it is not needed.

#l = ['Intensity']

# id_temp = xNODES['id']
# int_temp = np.log10(xNODES['Intensity'])

# merged_temp = pd.concat([id_temp, int_temp], axis=1)

for i in l:
   color_type = i
   
    
   carac = xNODES[['id', color_type]]

   #print(carac)

   carac = carac.set_index('id')
   carac = carac.reindex(G.nodes())
   carac[color_type]=pd.Categorical(carac[color_type])
   carac[color_type].cat.codes
   nodes = G.nodes()
   plt.figure()
   plt.title(color_type, wrap=True, fontsize=25)
##        nx.draw(G, pos = nx.nx_pydot.graphviz_layout(G), with_labels=False, node_color=carac[color_type].cat.codes, cmap=plt.cm.jet, node_size=3, width=.1, font_size=10)
##        pos = nx.spring_layout(G)
   pos = nx.nx_pydot.graphviz_layout(G)
##        pos = nx.fruchterman_reingold_layout(G)
##        pos = nx.nx_pydot.graphviz_layout(G)
   nx.draw(G, pos, with_labels=False, node_color=node_colors, node_size=2, width=.01, alpha=1)
   ec = nx.draw_networkx_edges(G, pos, width=.01, alpha=0.2)
#    nc = nx.draw_networkx_nodes(G, pos, nodelist=nodes, node_color=carac[color_type], with_labels=False, node_size=5, cmap=plt.cm.jet) 
   nc = nx.draw_networkx_nodes(G, pos, nodelist=nodes, node_color=carac[color_type], node_size=3, cmap=plt.cm.jet) 
##    plt.colorbar(nc)
   cb = plt.colorbar(nc, orientation='vertical')
   nc.figure.axes[0].tick_params(axis="both", labelsize=21)           ## change the label size
   nc.figure.axes[1].tick_params(axis="y", labelsize=21)              ## change the tick label size of colorbar
   plt.axis('off')
   plt.savefig('Supporting Figure 8 - ' + color_type + '_minimal_set_NH3' + '.png')
   plt.show()
"""


# In[21]:


print(xNODES['component'].unique())

FILTER_COMPONENTS  =        list(df_comp_freq_comb.index)[:nComponents]
print(FILTER_COMPONENTS)
print(df_comp_freq_comb)

print(xNODES)


# In[31]:


##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#trafos          =       ['H2','O','CO','NH3']

trafos          =    ['H2', 'O', 'CO', 'H3N', 'CHN', 'CH3N', 'H2O', 'CO2', 'CH2O', 'CHNO']
##trafoWeights  =       [ 1,   5,   1,   2]
trafoWeights    =       [1]*len(trafos)
columnNames     =       ['molecular_formula','abundance_int']             ##  for  molFormula  &  abundance/intensity
sample          =       'Ice composition H2O_CH3OH_NH3'
nComponents     =       1
only_uniques    =       'yes'
color_type      =       sample
node_size       =       'Degree'
##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

######################################
##                              TRAFOS
######################################
cf = re.compile('([A-Z][a-z]?)(\d?\d?\d?)')
all_elements = []
for xxxx in trafos:
    for el, count in re.findall(cf, xxxx):
        all_elements += [el]

atoms = set(all_elements)
counts = [[""] + trafos]
for e in atoms:
    count = [e]
    for xxxx in trafos:    
        d = dict(re.findall(cf, xxxx))
        n = d.get(e, 0)
        if n == '': n = 1
        count += [int(n)]
    counts += [count]

A = np.array(counts)
A_t = (A.transpose())

x2 = pd.DataFrame(A_t)
new_header = x2.iloc[0]
x2 = x2[1:]
x2.columns = new_header
x2 = x2.rename(columns={'': 'Molecular Formula'})
for i in ['H', 'C', 'N', 'O', 'S']:
    if i in x2.columns:
        x2[i] = x2[i].astype(int)
    else:
        x2[i] = 0    

h = x2['H'].to_numpy()
c = x2['C'].to_numpy()
n = x2['N'].to_numpy()    
o = x2['O'].to_numpy()
s = x2['S'].to_numpy()
elements = np.transpose(np.array([h, c, n, o, s]))
massExact = np.dot(elements, masses)
x2['Mass (exact)'] = massExact.round(6)
weights = pd.DataFrame(trafoWeights,columns = ['trafoWeights'])
weights['trafos'] = trafos
x3 = x2.join(weights.set_index(['trafos'], verify_integrity=True ),on=['Molecular Formula'], how='left')

xTRAFOS = x3
mdTRAFOS = xTRAFOS['Mass (exact)'].round(6).to_numpy()

######################################
##                          DETECTIONS
######################################
path = os.getcwd()
files = glob.glob(os.path.join("*.tsv"))
d = {}
for f in files:
    key = f
    name = f.replace(".tsv", "")
    x = globals()[f"orig_{name}"] = pd.read_csv(f, sep='\t', skiprows=0)
    if 'id' in x.columns:
        pass
    else:
        x.insert(0, 'id', range(1, 1 + len(x)))
#    
    try:
        x = x[['id',sample,columnNames[0],columnNames[1]]].rename(columns={columnNames[0]: 'Molecular Formula',columnNames[1]:'Intensity'})
    except:
        x = x[['id',columnNames[0],columnNames[1]]].rename(columns={columnNames[0]: 'Molecular Formula',columnNames[1]:'Intensity'})
#
    molFormulas = x['Molecular Formula']
    a = []
    for i in molFormulas:
        tmp_a = chemparse.parse_formula(i)
        a.append(tmp_a)
#
    x_a = pd.DataFrame(a).fillna(0)
    x2 = pd.concat([x,x_a], axis=1)
#
    if 'H' in x2.columns:
        x2['H'] = x2['H'].astype(int)
    else:
        x2['H'] = 0    
#
    if 'C' in x2.columns:
        x2['C'] = x2['C'].astype(int)
    else:
        x2['C'] = 0    
#
    if 'N' in x2.columns:
        x2['N'] = x2['N'].astype(int)
    else:
        x2['N'] = 0    
#
    if 'O' in x2.columns:
        x2['O'] = x2['O'].astype(int)
    else:
        x2['O'] = 0    
#
    if 'S' in x2.columns:
        x2['S'] = x2['S'].astype(int)
    else:
        x2['S'] = 0    
#
    h = x2['H'].to_numpy()
    c = x2['C'].to_numpy()
    o = x2['O'].to_numpy()
    n = x2['N'].to_numpy()    
    s = x2['S'].to_numpy()
    elements = np.transpose(np.array([h, c, n, o, s]))
    massExact = np.dot(elements, masses)
    x2['Mass (exact)'] = massExact.round(6)
    x2['mu'] = mu = ( (h*valence_h + c*valence_c + n*valence_n + o*valence_o + s*valence_s) / 2 ) - (h + c + o + n + s) + 1
    x2['DBE'] = dbe = c - (h/2) + (n/2) + 1
    x2['HC'] = h/c
    x2['NC'] = n/c
    x2['OC'] = o/c
    x2['SC'] = s/c
    x2['FILTER'] =  1 * ( (mu >= 0) )
    atom_list = ['H','C','N','O','S']
    for i in range(len(atom_list)-1):
        x2['Molecular Formula'] = x2['Molecular Formula'].replace({atom_list[i]+'0'}, {atom_list[i]}, regex=True)
        x2['Molecular Formula'] = x2['Molecular Formula'].replace({atom_list[i]+'1'+atom_list[i+1]}, {atom_list[i]}, regex=True)
        x2['Molecular Formula'] = x2['Molecular Formula'].replace({'S'+'0'}, {''}, regex=True)
        x2['Molecular Formula'] = x2['Molecular Formula'].replace({'S'+'1'}, {'S'}, regex=True)
#
    x2['Mass (frequency)'] = x2['Mass (exact)'].map(dict(x2['Mass (exact)'].value_counts()))
    try:
        if only_uniques == 'yes':
            x3 = x2.drop_duplicates(subset=['Mass (exact)'], keep='first')
        else:
            pass
    except:
        x3 = x2
#
    xNODES0 = xDET = x3[(x3['FILTER']==1)]
    globals()[f"{name}"] = xDET
    mass2 = mass = xDET['Mass (exact)'].round(6).to_numpy()
    data = xDET[['id','Mass (exact)']]
#
    ##########################################     x x    
    ##                              MD MATCHES
    ##########################################
    md_matches = namedtuple('md_matches', 'md_matches hits')
    new_list = []
    for md in mdTRAFOS:
	    for element in mass:
		    if element+md in mass2:
			    tmp_new_list = md_matches(element, element+md)
			    new_list.append(tmp_new_list)
    #
    matches = np.array(new_list).reshape(len(new_list),2)
    matches = matches[np.argsort(matches[:,0])]
    sources = data.rename(columns={'id': 'Source', 'Mass (exact)': 'Mass (source)'})
    targets = data.rename(columns={'id': 'Target', 'Mass (exact)': 'Mass (target)'})
    matches2 = pd.DataFrame({'Mass (source)': matches[:, 0], 'Mass (target)': matches[:, 1]})
    source_match = matches2.merge(sources, how='left', on=['Mass (source)'])
    target_match = source_match.merge(targets, how='left', on=['Mass (target)'])
    target_match['Mass difference'] = mass_diff = np.around(target_match['Mass (target)'] - target_match['Mass (source)'],6)
    xxxx = xTRAFOS[['Mass (exact)','Molecular Formula']].rename(columns={'Mass (exact)': 'Mass difference'})
    x = target_match.merge(xxxx, how='left', on=['Mass difference'])
    x['type'] = type = np.full((len(mass_diff),1), 'Undirected')
    x['Label'] = Label = np.full((len(mass_diff),1), 'x')
    xxxx = xTRAFOS[['Mass (exact)','trafoWeights']]
    x2 = x.join(xxxx.set_index(['Mass (exact)'], verify_integrity=True ),on=['Mass difference'], how='left').rename(columns={'trafoWeights': 'Weight'}).sort_values(by=['Source'])
    x2['Mass difference (frequency)'] = x2['Mass difference'].map(dict(x2['Mass difference'].value_counts()))
    #
    xEDGES0 = x2
    #
    md_freq = x2[['Molecular Formula','Mass difference (frequency)']]
    md_freq = md_freq.drop_duplicates(subset=['Molecular Formula'], keep='first')
    md_freq = md_freq.sort_values(by=['Mass difference (frequency)'], ascending=False)
    print('\n\n\n\n')
    print(name+', '+str(len(xNODES0))+' nodes,', str(len(xEDGES0))+' edges')
    print(md_freq)
    print('\n\n\n\n')
#
    ##########################################     x x    
    ##                             NETWORK ANA
    ##########################################
    G0 = nx.from_pandas_edgelist(xEDGES0, 'Source', 'Target', create_using=nx.Graph())
    G0cc = list(nx.connected_components(G0))
    d = {name:k for k,comp in enumerate(G0cc) for name in comp}               #dict(enumerate(G0cc))
    df_comp = pd.DataFrame.from_dict(d, orient='index', columns=['component']).rename_axis('id').reset_index()
    nodes2 = xNODES0.merge(df_comp, how='left', on='id')
    nans = nodes2[nodes2.isna().any(axis=1)]
    xNODES0 = nodes2[~nodes2['component'].isnull()]
    #
    df_source = xEDGES0.rename(columns={'Source':'id'})
    edges2 = df_source.merge(df_comp, how='left', on='id')
    edges2 = edges2.rename(columns={'component':'component source','id':'Source'})
    df_target = edges2.rename(columns={'Target':'id'})
    edges3 = df_target.merge(df_comp, how='left', on='id')
    edges3 = edges3.rename(columns={'component':'component target','id':'Target','mass_difference':'mass difference','molecular_formula':'molecular formula'})
    #
    df_comp_freq = pd.DataFrame(xNODES0['component'].value_counts()).rename(columns={'component': 'freq_component'})
    xNODES0['df_comp_freq'] = xNODES0['component'].map(dict(xNODES0['component'].value_counts()))
    df_comp_freq_norm = pd.DataFrame(xNODES0['component'].value_counts(normalize=True)).rename(columns={'component': 'freq_component_norm [0-1]'})
    df_comp_freq_comb = pd.concat([df_comp_freq,df_comp_freq_norm], axis=1)
    print('\n\n\n\n')
    print(name+' (all components)')
    print(df_comp_freq_comb)
    print('\n\n\n\n')
    #
    FILTER_COMPONENTS  =        list(df_comp_freq_comb.index)[:nComponents]
    xEDGES = edges3[(edges3['component source'].isin(FILTER_COMPONENTS))]
    xNODES = xNODES0[(xNODES0['component'].isin(FILTER_COMPONENTS))]
    G = nx.from_pandas_edgelist(xEDGES, 'Source', 'Target', create_using=nx.Graph())
    a = pd.DataFrame(G.nodes(data = True)).rename(columns={0:'id'}).drop([1], axis=1)
    a2 = a.join(xNODES.set_index('id'), on='id')
    a2['id_G0_nodes'] = range(1,1+len(a2))
    xNODES = a2
#
    degrees = pd.DataFrame(G.degree(), columns=["id", "Degree"])
    degrees['id'] = degrees['id'].astype(int)
    xNODES = xNODES.join(degrees.set_index('id'), on='id') 
    xNODES['Degree counts'] = xNODES['Degree'].map(dict(xNODES['Degree'].value_counts()))
#
##    bc = nx.betweenness_centrality(G, normalized=True)                                                                                                                                                                                             ## takes a while....
##    bc2 = pd.DataFrame.from_dict(bc, orient='index', columns=['betCen']).rename_axis('id').reset_index()
##    xNODES = xNODES.merge(bc2, how='left', on='id')
#
    colors = mcp.gen_color(cmap='gist_rainbow',n=len(xNODES[color_type].unique()))                    ##  cmap  AUTUMN tab20b coolwarm gist_rainbow tab20c brg      https://matplotlib.org/stable/tutorials/colors/colormaps.html
    ##colors = ['#15B01A',    'blue', 'orange',     'lightgray',   'red', 'green']
    samples0 = xNODES[color_type].unique().tolist()
    try:
        samples = sorted(samples0, key=lambda x: int("".join([i for i in x if i.isdigit()])))
    except:
        samples = sorted(samples0)
    #
    sample_colors = pd.DataFrame({color_type: samples, 'color': colors})
    xNODES = xNODES.merge(sample_colors, how='left', on=color_type)
    xNODES = xNODES.sort_values(by=['id_G0_nodes'], ascending=True)                              ## re-ordering/re-indexing according to graph G0 (id_G0_nodes)
    node_colors = xNODES['color'].tolist()
#
    xNODES = xNODES.sort_values(by=['id_G0_nodes'], ascending=True)
    xNODES['node_size'] = (xNODES[node_size] - np.min(xNODES[node_size])) / (np.max(xNODES[node_size]) - np.min(xNODES[node_size]))
    node_sizes = xNODES['node_size'].tolist()
    node_sizes = [i * 1e2 for i in node_sizes]


#########################
#     CLUSTER TRANSITIONS                              fig 2: min_set     directions between nh3_poor to medium, etc..
#         WITH DIRECTIONS
#########################
#~~~~~~~~~~~~~~~~~~~~~~~
FROM_TO = 'Ice composition H2O_CH3OH_NH3'
#~~~~~~~~~~~~~~~~~~~~~~~
df_samples = xNODES[['id', FROM_TO]]
df_edges4 = xEDGES.rename(columns={'Source':'id'})
df_edges4 = df_edges4.merge(df_samples, how='left', on='id')
df_edges4 = df_edges4.rename(columns={FROM_TO: FROM_TO+str('_Source'), 'id':'Source'})
df_edges5 = df_edges4.rename(columns={'Target':'id'})
df_edges5 = df_edges5.merge(df_samples, how='left', on='id')
df_edges5 = df_edges5.rename(columns={FROM_TO: FROM_TO+str('_Target'), 'id':'Target'})

#~~~~~~~~~~~~~~~~~~~~~~~
SAMPLE_START = '3_1_0.2'
SAMPLE_END   = '3_1_1'

#SAMPLE_START = '3_1_1'
#SAMPLE_END   = '10_1_1'

#SAMPLE_START = '10_1_1'
#SAMPLE_END   = '3_1_5'
#~~~~~~~~~~~~~~~~~~~~~~~
df_edges6 = df_edges5[(df_edges5[FROM_TO+str('_Source')] == SAMPLE_START) & (df_edges5[FROM_TO+str('_Target')] == SAMPLE_END)]
df_edges6_backward= df_edges5[(df_edges5[FROM_TO+str('_Source')] == SAMPLE_END) & (df_edges5[FROM_TO+str('_Target')] == SAMPLE_START)]
products6  = df_edges6['Mass (target)'] - df_edges6['Mass (source)']
products6_backward  = df_edges6_backward['Mass (target)'] - df_edges6_backward['Mass (source)']

for x in trafos:
    TRAFO = x
    globals()['df_edges6_%s' % x] = df_edges6[(df_edges6['Molecular Formula'].str.strip() == TRAFO)]
    globals()['products6_%s' % x]  = globals()['df_edges6_%s' % x]['Mass (target)'] - globals()['df_edges6_%s' % x]['Mass (source)']  
    globals()['df_edges6_backward_%s' % x] = df_edges6_backward[(df_edges6_backward['Molecular Formula'].str.strip() == TRAFO)]
    globals()['products6_backward_%s' % x] = globals()['df_edges6_backward_%s' % x]['Mass (target)'] - globals()['df_edges6_backward_%s' % x]['Mass (source)']

forward = []
for x in trafos:
    a = globals()['products6_%s' % x].shape[0]
    b = int(a)
    forward.append(b)

backward = []
for x in trafos:
    a = globals()['products6_backward_%s' % x].shape[0]
    b = int(a)
    backward.append(b)

df = pd.DataFrame({'Forward': forward,  'Backward': backward}, index=trafos)

fontsize = 12
df.plot(kind='bar', rot=0,  width=.8, color={"Forward": "black", 'Backward': 'w'},  edgecolor='black', stacked=True, fontsize=fontsize)
plt.title(SAMPLE_START+r'$\rightarrow$'+SAMPLE_END+'    ('+str(sample)+', Component ' +str(FILTER_COMPONENTS)+')', fontsize=fontsize+2)
plt.xlabel('Transformations', fontsize=fontsize+2)
plt.ylabel('Frequency', fontsize=fontsize+2)
plt.legend(fontsize=fontsize)
##plt.savefig(name+'_Component'+str(FILTER_COMPONENTS)+'_'+SAMPLE_START+SAMPLE_END+'    trafo_bar_chart'+'.png')
plt.show()

