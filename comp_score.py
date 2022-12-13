###########################################

# CALCULATING COMPONENT SCORES (COMP_SCORE)
# for each component
# to automatize "key component selection"

# assumes you have computed "xNODES0" already.

###########################################

df = xNODES0

#print(df)

num_comps = len(df['component'].unique())

# list_elements = ['H', 'C', 'N', 'O', 'HC', 'NC', 'OC', 'Mass (exact)', 'Intensity', 'DBE']

comp_dict = dict()

for i in range(num_comps):
    
    list_H = []
    list_C = []
    list_N = []
    list_O = []
    list_HC = []
    list_NC = []
    list_OC = []
    list_mz = []
    list_int = []
    list_DBE = []
    
    size_comp = 0
    
    for index, row in df.iterrows():
        #my_head = df.at[index, 'DOT_Head']
        if(df.at[index, 'component'] == i):
            #print("heyy")
            list_H.append(df.at[index, 'H'])
            list_C.append(df.at[index, 'C'])
            list_N.append(df.at[index, 'N'])
            list_O.append(df.at[index, 'O'])
            list_HC.append(df.at[index, 'HC'])
            list_NC.append(df.at[index, 'NC'])
            list_OC.append(df.at[index, 'OC'])
            list_mz.append(df.at[index, 'Mass (exact)'])
            list_int.append(df.at[index, 'Intensity'])
            list_DBE.append(df.at[index, 'DBE'])
            size_comp = df.at[index, 'df_comp_freq']
    
    min_H = min(list_H)
    max_H = max(list_H)
    
    min_C = min(list_C)
    max_C = max(list_C)
    
    min_N = min(list_N)
    max_N = max(list_N)
    
    min_O = min(list_O)
    max_O = max(list_O)
    
    min_HC = min(list_HC)
    max_HC = max(list_HC)
    
    min_NC = min(list_NC)
    max_NC = max(list_NC)
    
    min_OC = min(list_OC)
    max_OC = max(list_OC)
    
    min_mz = min(list_mz)
    max_mz = max(list_mz)
    
    min_int = min(list_int)
    max_int = max(list_int)
    
    min_DBE = min(list_DBE)
    max_DBE = max(list_DBE)

    
    comp_score = 0
    comp_score += ((max_H + min_H) / (1 + max_H - min_H)) + ((max_C + min_C) / (1 + max_C - min_C))
    comp_score += ((max_N + min_N) / (1 + max_N - min_N)) + ((max_O + min_O) / (1 + max_O - min_O)) + ((max_HC + min_HC) / (1 + max_HC - min_HC))
    comp_score += ((max_NC + min_NC) / (1 + max_NC - min_NC)) + ((max_OC + min_OC) / (1 + max_OC - min_OC)) + ((max_mz + min_mz) / (1 + max_mz - min_mz))
    comp_score += ((max_int + min_int) / (1 + max_int - min_int)) + ((max_DBE + min_DBE) / (1 + max_DBE - min_DBE))

    comp_score = comp_score * size_comp
     
    comp_dict[i] = comp_score
    
#print(comp_dict)

# sorting this dictionary of component scores in a descending fashion
sorted_comp_dict = dict(sorted(comp_dict.items(), key=lambda item: item[1], reverse = True))


print(sorted_comp_dict)