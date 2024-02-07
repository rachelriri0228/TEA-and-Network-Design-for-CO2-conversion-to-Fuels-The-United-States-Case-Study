# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 13:26:23 2024

@author: User
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 14:53:49 2024

@author: User
"""

import numpy as np
from gurobipy import * # Import Gurobi solver
import pandas as pd
import xlrd

model = Model('Linear Program')

# Set the integrality tolerance
model.Params.OptimalityTol = 0.01  # Adjust the value as need
# Set the optimization gap (e.g., 1%)
optimization_gap = 0.01
model.Params.MIPGap = optimization_gap
# Set the optimization gap (e.g., 1%)
optimization_gap = 0.01
model.Params.MIPGap = optimization_gap

#Index===================================
Region = ['Midwest','DEN','MSP','MCI','Northeast','CLT_RDU','BNA_ATL','Southeast','MSY','Southwest','West','SFO','SLC','SEA_PDX']
region = ['mid west','DEN','MSP','MCI','north east','CLT_RDU','BNA_ATL','south east','MSY','south west','West','SFO','SLC','SEA_PDX']
I = [211,26,60,39,223,46,52,48,68,121,76,59,12,20]
J = [250,36,87,54,247,63,68,58,84,142,98,79,31,34]
A = [8,1,1,1,8,2,2,5,1,6,5,4,1,2]
P = [2,2,2,2,2,2,2,2,2,2,2,2,2,2]
# H2 = [500,600,700,800,900,1000]
# Op = [0.08,0.1,0.12,0.14,0.16,0.18,0.2]
# for R,airport,r,source,refinery,pathway in zip(Region,A,region,I,J,P):
#     print(R,airport,r,source,refinery,pathway)

#Parameter====================================================================
#annual jet fuel demand in airprt a (Ton)
#path = '/Users/User/OneDrive - University of Tennessee/CO2/Data/CO2 supply chain network/CO2-CO-FTL-' #win
#path = '/Users/rachelriri/Library/CloudStorage/OneDrive-UniversityofTennessee/CO2/Data/CO2 supply chain network/CO2-CO-FTL-' #mac
#path = '/Users/ruizhou/Library/CloudStorage/OneDrive-UniversityofTennessee/CO2/Data/CO2 supply chain network/CO2-CO-FTL-' #win2
#path = '/Users/ruizhou/Library/CloudStorage/OneDrive-UniversityofTennessee/CO2/Data/CO2 supply chain network/CO2-CO-FTL-' #mac2
path = '/Users/Rui Zhou/OneDrive - University of Tennessee/CO2/Data/CO2 supply chain network/CO2-CO-FTL-' #win 303


D_a = []
alpha_s = []
h_s = []
n_s = []
d_s = []
g_s = []
l_s = []
sp = []
C_sj = []
c_i_c = []
E_i = []
c_sj_o = []
c_sj_f = []
d_ij = []
d_ja = []
c_ja_v = []
c_ja_f = []

for R,airport,r,source,refinery,pathway in zip(Region,A,region,I,J,P):
    # H2 price
    h_i = 1000
    #productio times
    production_time = 1.86
    #operating percentage
    op_percentage = 0.05
   #The unit operating costs of potential conversion facilities j through conversion pathway s ($/ton)
    c_sj_o.append(np.asmatrix(pd.read_excel(path + R +'/Unit operating cost of more refineries compare_same.xlsx').iloc[0:2,1:].values)*production_time*op_percentage) #1.86 is the production times under 85.8% recycle rate
    D_a.append(pd.read_excel(path + R +'/Jet fuel demand of '+str(airport)+' major airports in '+r+'.xlsx').values.ravel()*100)
    alpha_s.append(np.asmatrix(pd.read_excel(path + R +'/Conversion rate_compare.xlsx').iloc[:,0].values*production_time))#*1.82 is under 85.8% recycle rate
    h_s.append(np.asmatrix(pd.read_excel(path + R +'/Hydrogen consuming_compare.xlsx').iloc[:,0].values/2))    
    #side production ratio through conversion pathway s (ton).
    n_s.append(np.asmatrix(pd.read_excel(path + R +'/Side production ratio.xlsx',sheet_name = 'naphtha' ).iloc[:,0:1].values*production_time))#*1.82 is under 85.8% recycle rate
    d_s.append(np.asmatrix(pd.read_excel(path + R +'/Side production ratio.xlsx',sheet_name = 'diesel' ).iloc[:,0:1].values*production_time))#*1.82 is under 85.8% recycle rate
    g_s.append(np.asmatrix(pd.read_excel(path + R +'/Side production ratio.xlsx',sheet_name = 'gasoline' ).iloc[:,0:1].values))
    l_s.append(np.asmatrix(pd.read_excel(path + R +'/Side production ratio.xlsx',sheet_name = 'liquefied petrpleum gas' ).iloc[:,0:1].values))   
    #Side products price $/t
    sp.append(pd.read_excel(path + R +'/Side products price.xlsx').values.ravel())
    #Maximum annual conversion capacity in potential conversion facilities  j through conversion pathway s (Ton).
    C_sj.append((np.asmatrix(pd.read_excel(path + R +'/refinery capacity '+str(refinery)+'_compare.xlsx').iloc[0:2,1:].values))/159*7.9*100)#1 barrel =159kg,1 barrel = 7.9 ton   
    #annual CO2 emission amount in staionary source i (Ton)
    E_i.append((np.asmatrix(pd.read_excel(path + R +'/CO2 emission from power plant in '+r+'.xlsx').iloc[:,0:].values))*1.1)#*1.1 to convert metric ton to ton
    #The unit CO2 capture costs at stationary CO2 source i ($/ton)
    c_i_c.append(np.asmatrix(pd.read_excel(path + R +'/CO2 capture cost in '+str(source)+' sources.xlsx').iloc[:,1:]))
    #The fixed annual capital costs of potential conversion facilities  j through conversion pathway s ($).
    c_sj_f.append(np.asmatrix(pd.read_excel(path + R +'/Annual capital cost of '+str(refinery)+' refineries_compare.xlsx').iloc[0:2,0:].values))
    #The unit pipeline cost from stationary CO2 source i to potential conversion facilities  j by pipeline ($/ton).
    c_ij_l = 0.09    
    #d_ij = pd.read_excel(path_windows + Region +'/Distance between source and more refinery in south east.xlsx')
    d_ij.append(np.asmatrix(pd.read_excel(path + R +'/Distance between source and more refinery in '+r+'.xlsx').iloc[:,1:]).T)    
    #The unit transportation cost from potential conversion facilities  j to airport a by truck ($/gallon).
    #distance
    d_ja.append(np.asmatrix(pd.read_excel(path + R +'/Distance between more refinery and airport in '+r+'.xlsx').iloc[:,1:]))
    #transportation cost from refinery to airport
    #variable cost    
    c_ja_v.append(0.9*pd.read_excel(path + R +'/Distance between more refinery and airport in '+r+'.xlsx').iloc[:,1:]) #mile
    #fixed cost
    c_ja_f = 24

    #The average jet fuel sale price per ton ($/t). 2022 price is 1144.7$/mt 1mt=1.1t
    p = 732.5 # jet fuel price per ton
    n = 624.4# naphtha
    d = 1110.3# diesel
    g = 917.4# gasoline
    l = 221# liquefied petrpleum gas
    M = 100000000

  
    #Decision variables===========================================================
    #The captured CO2 flow transported to potential conversion facilities  j from stationary CO2 source i (Ton).
for regions in range(0,14):
    airport = A[regions]
    source = I[regions]
    refinery = J[regions]
    pathway = P[regions]
    r = region[regions]
    
    x_ij = {} 
    for i in range(0,source):
        for j in range(0,refinery):
            # Ensure unique variable names
            var_name = f'x_ij_{i}_{j}'
            x_ij[i,j] = model.addVar(lb=0.0,ub=GRB.INFINITY,vtype=GRB.CONTINUOUS,name=var_name)
    
    #The converted jet fuel was transported to airport k from potential conversion facilities  j (gallon).
    y_ja = {} 
    for j in range(0,refinery):
        for a in range(0,airport):
            var_name = f'y_ja_{j}_{a}'
            y_ja[j,a] = model.addVar(lb=0.0,ub=GRB.INFINITY,vtype=GRB.CONTINUOUS,name=var_name)
    
    n_ja = {} 
    for j in range(0,refinery):
        for a in range(0,airport):
            var_name = f'n_ja_{j}_{a}'
            n_ja[j,a] = model.addVar(lb=0.0,ub=GRB.INFINITY,vtype=GRB.CONTINUOUS,name=var_name)  
    
    de_ja = {} 
    for j in range(0,refinery):
        for a in range(0,airport):
            var_name = f'de_ja_{j}_{a}'
            de_ja[j,a] = model.addVar(lb=0.0,ub=GRB.INFINITY,vtype=GRB.CONTINUOUS,name=var_name)  
    
    g_ja = {} 
    for j in range(0,refinery):
        for a in range(0,airport):
            var_name = f'g_ja_{j}_{a}'
            g_ja[j,a] = model.addVar(lb=0.0,ub=GRB.INFINITY,vtype=GRB.CONTINUOUS,name=var_name)        
    
    l_ja = {} 
    for j in range(0,refinery):
        for a in range(0,airport):
            var_name = f'l_ja_{j}_{a}'
            l_ja[j,a] = model.addVar(lb=0.0,ub=GRB.INFINITY,vtype=GRB.CONTINUOUS,name=var_name)  
    
    #Binary variable, =1 if converstion pathway s is adopted at facility " = 0 "otherwise" )
    z_sj = {} 
    for s in range(0,pathway):
        for j in range(0,refinery):
            var_name = f'z_sj_{s}_{j}'
            z_sj[s,j] = model.addVar(vtype=GRB.BINARY,name=var_name)
            
    #capture cost
    capture_cost = model.addVar(lb=0.0,ub=GRB.INFINITY,vtype=GRB.CONTINUOUS,name="capture_cost")
    
    #h2 cost
    h2_cost = model.addVar(lb=0.0,ub=GRB.INFINITY,vtype=GRB.CONTINUOUS,name="h2_cost")
    
    #pipeline cost
    pipeline_cost = model.addVar(lb=0.0,ub=GRB.INFINITY,vtype=GRB.CONTINUOUS,name="pipeline_cost")
    
    #operation cost
    operation_cost = model.addVar(lb=0.0,ub=GRB.INFINITY,vtype=GRB.CONTINUOUS,name="operation_cost")
    
    #capital cost
    capital_cost = model.addVar(lb=0.0,ub=GRB.INFINITY,vtype=GRB.CONTINUOUS,name="capital_cost")
    
    #truck cost
    truck_cost = model.addVar(lb=0.0,ub=GRB.INFINITY,vtype=GRB.CONTINUOUS,name="truck_cost")
    
    #naphtha_profit
    naphtha_profit = model.addVar(lb=0.0,ub=GRB.INFINITY,vtype=GRB.CONTINUOUS,name="naphtha_profit")
    
    #operation cost
    diesel_profit = model.addVar(lb=0.0,ub=GRB.INFINITY,vtype=GRB.CONTINUOUS,name="diesel_profit")
    
    #capital cost
    gasoline_profit = model.addVar(lb=0.0,ub=GRB.INFINITY,vtype=GRB.CONTINUOUS,name="gasoline_profit")
    
    #truck cost
    lpg_profit = model.addVar(lb=0.0,ub=GRB.INFINITY,vtype=GRB.CONTINUOUS,name="lpg_profit")
    
    #naphtha_profit
    jetfuel_profit = model.addVar(lb=0.0,ub=GRB.INFINITY,vtype=GRB.CONTINUOUS,name="jetfuel_profit")
    

#Objective function==========================================maintains conversion balance at each facility j=================
    model.setObjective(quicksum(c_i_c[regions][i] * x_ij[i,j] for i in range (0,source) for j in range(0,refinery))
                     + quicksum(h_i * x_ij[i,j] * h_s[regions][s] for i in range(0,source) for j in range(0,refinery) for s in range(0,pathway))
                     + quicksum(c_ij_l * d_ij[regions][i,j] * x_ij[i,j] for i in range(0,source) for j in range(0,refinery))
                     + quicksum(c_sj_o[regions][s,j] * x_ij[i,j] for s in range (0,pathway) for i in range(0,source) for j in range(0,refinery))
                     + quicksum(c_sj_f[regions][s,j] * z_sj[s,j] for s in range(0,pathway) for j in range(0,refinery))
                     + quicksum(0.9*d_ja[regions][j,a] * y_ja[j,a] for j in range(0,refinery) for a in range (0,airport))
                     + quicksum(c_ja_f * y_ja[j,a] for j in range(0,refinery) for a in range (0,airport))
                     - quicksum(n * n_ja[j,a] for j in range(0,refinery) for a in range(0,airport))
                     - quicksum(d * de_ja[j,a] for j in range(0,refinery) for a in range(0,airport))
                     - quicksum(g * g_ja[j,a] for j in range(0,refinery) for a in range(0,airport))
                     - quicksum(l * l_ja[j,a] for j in range(0,refinery) for a in range(0,airport))
                     - quicksum(p * y_ja[j,a] for j in range(0,refinery) for a in range(0,airport)),GRB.MINIMIZE)           
                
    #Constrians===================================================================
    #Maintains conversion balance at each facility j
    # for j in range(0,J):
    #     model.addConstr(quicksum(M*(1-z_sj[s,j])+alpha_s[s] * x_ij[i,j] for i in range(0,I) for s in range(0,P))
    #                     == quicksum(y_ja[j,a] for a in range(0,A)))
     
    
    #Maintains conversion balance at each facility j
    for j in range(0,refinery):
        model.addConstr(quicksum(alpha_s[regions][s] * x_ij[i,j] for i in range(0,source) for s in range(0,pathway))
                        ==quicksum(y_ja[j,a] for a in range(0,airport)))
    
    for j in range(0,refinery):
        model.addConstr(quicksum(n_s[regions][s] * x_ij[i,j] for i in range(0,source) for s in range(0,pathway))
                        ==quicksum(n_ja[j,a] for a in range(0,airport)))    
    
    for j in range(0,refinery):
        model.addConstr(quicksum(d_s[regions][s]  * x_ij[i,j] for i in range(0,source) for s in range(0,pathway))
                        ==quicksum(de_ja[j,a] for a in range(0,airport))) 
        
    for j in range(0,refinery):
        model.addConstr(quicksum(g_s[regions][s] * x_ij[i,j] for i in range(0,source) for s in range(0,pathway))
                        ==quicksum(g_ja[j,a] for a in range(0,airport))) 
    
    for j in range(0,refinery):
        model.addConstr(quicksum(l_s[regions][s] * x_ij[i,j] for i in range(0,source) for s in range(0,pathway))
                        ==quicksum(l_ja[j,a] for a in range(0,airport)))   
    
    #enforces the potential facilities conversion capacity
    for j in range(0,refinery):
        model.addConstr(quicksum(x_ij[i,j] for i in range(0,source)) <= quicksum(C_sj[regions][s,j] * z_sj[s,j] for s in range(0,pathway)))
    
    #the amount of captured CO2 does not exceed the available of CO2 emissions
    for i in range(0,source):
        model.addConstr(quicksum(x_ij[i,j] for j in range(0,refinery)) <= E_i[regions][i])
    
    #the converted jet fuel shipped to an airport is bounded by the airports’ demand
    for a in range(0,airport):
        model.addConstr(quicksum(y_ja[j,a] for j in range(0,refinery)) <= D_a[regions][a])
    
    # Add constraint to limit the total number of selected refineries to 1
    model.addConstr(quicksum(z_sj[s,j] for j in range(0,refinery)) == 1, "select_one_refinery")
    
    
    for a in range(0,airport):
        model.addConstr(quicksum(n_ja[j,a] for j in range(0,refinery)) >= 0)
        
    for a in range(0,airport):
        model.addConstr(quicksum(de_ja[j,a] for j in range(0,refinery)) >= 0)
    
    for a in range(0,airport):
        model.addConstr(quicksum(g_ja[j,a] for j in range(0,refinery)) >= 0)
    
    for a in range(0,airport):
        model.addConstr(quicksum(l_ja[j,a] for j in range(0,refinery)) >= 0)
        
    #each potential conversion facility only use one type of conversion pathway, which defines a quadruplet {annual capacity, conversion rate, fixed annual cost, unit conversion cost per ton}.
    for j in range(0,refinery):
        model.addConstr(quicksum(z_sj[s,j] for s in range(0,pathway)) <= 1)
        
    
    # for a in range(0,A):
    #     model.addConstr(quicksum(d_ja[j,a] for j in range(0,J)) <=300)
    
    #capture cost
    model.addConstr(quicksum(c_i_c[regions][i] * x_ij[i,j] for i in range (0,source) for j in range(0,refinery))
                        == capture_cost)
    
    #h2 cost
    model.addConstr(quicksum(h_i * x_ij[i,j]* h_s[regions][s] for i in range(0,source) for j in range(0,refinery) for s in range(0,pathway))
                        == h2_cost)
    
    
    #pipeline cost
    model.addConstr(quicksum(c_ij_l * d_ij[regions][i,j] * x_ij[i,j] for i in range(0,source) for j in range(0,refinery))
                        == pipeline_cost)
    
    #operation cost
    model.addConstr(quicksum(c_sj_o[regions][s,j] * x_ij[i,j] for s in range (0,pathway) for i in range(0,source) for j in range(0,refinery))
                        == operation_cost)
    
    
    #capital_cost
    model.addConstr(quicksum(c_sj_f[regions][s,j] * z_sj[s,j] for s in range(0,pathway) for j in range(0,refinery))
                        == capital_cost)
    
    #truck cost
    model.addConstr(quicksum(0.9*d_ja[regions][j,a]* y_ja[j,a] for j in range(0,refinery) for a in range (0,airport))
                  + quicksum(c_ja_f * y_ja[j,a] for j in range(0,refinery) for a in range (0,airport))
                        == truck_cost)
    
    #naphtha_profit
    model.addConstr(quicksum(n * n_ja[j,a] for j in range(0,refinery) for a in range(0,airport))
                        == naphtha_profit)
    
    #diesel_profit
    model.addConstr(quicksum(d * de_ja[j,a] for j in range(0,refinery) for a in range(0,airport))
                        == diesel_profit)
    #gasoline_profit
    model.addConstr(quicksum(g * g_ja[j,a] for j in range(0,refinery) for a in range(0,airport))
                        == gasoline_profit)
    #lpg_profit
    model.addConstr(quicksum(l * l_ja[j,a] for j in range(0,refinery) for a in range(0,airport))
                        == lpg_profit)
    #jetfuel_profit
    model.addConstr(quicksum(p * y_ja[j,a] for j in range(0,refinery) for a in range(0,airport))
                        == jetfuel_profit)
    
    
    # Set NonConvex parameter to handle non-convex quadratic constraints
    model.setParam('NonConvex', 2)
    
    
    model.write('model.lp') 
    model.optimize() 
    # Specify the directory path and file name
    #directory_path = "/Users/User/OneDrive - University of Tennessee/CO2/Code/Code_A/Result/1_23_2024_state/100% recycle rate" #win
    #directory_path = "/Users/rachelriri/Library/CloudStorage/OneDrive-UniversityofTennessee/CO2/Code/Code_A/Result/1_23_2024_state/1100 H2 price" #mac
    #directory_path = '/Users/ruizhou/Library/CloudStorage/OneDrive-UniversityofTennessee/CO2/Code/Code_A/Result/1_30_map' #mac315
    directory_path = '/Users/Rui Zhou/OneDrive - University of Tennessee/CO2/Code/Code_A/Result/1_30_map' #mac315
  
    # Combine the directory path and file name to create the full file path
    file_path = os.path.join(directory_path, 'FTS CO2 conversion to jet fuel CO2_same in '+r+' 1000 H2 price 5% operating cost.sol')
    
    # Save the model to the specified file path
    model.write(file_path)



    # Print solution==========================================================================================================
    print("======objective value =======")
    obj = model.getObjective()
    print(obj.getValue())
    
    print("====== cost =======")
    var_name = model.getVarByName("capture_cost")
    print("capture_cost", "%g"%(var_name.X))
    
    var_name = model.getVarByName("h2_cost")
    print("h2_cost", "%g"%(var_name.X))
    
    var_name = model.getVarByName("pipeline_cost")
    print("pipeline_cost", "%g"%(var_name.X))
    
    var_name = model.getVarByName("operation_cost")
    print("operation_cost", "%g"%(var_name.X))
    
    var_name = model.getVarByName("capital_cost")
    print("capital_cost", "%g"%(var_name.X))
    
    var_name = model.getVarByName("truck_cost")
    print("truck_cost", "%g"%(var_name.X))
    
    var_name = model.getVarByName("naphtha_profit")
    print("naphtha_profit", "%g"%(var_name.X))
    
    var_name = model.getVarByName("diesel_profit")
    print("diesel_profit", "%g"%(var_name.X))
    
    var_name = model.getVarByName("gasoline_profit")
    print("gasoline_profit", "%g"%(var_name.X))
    
    var_name = model.getVarByName("lpg_profit")
    print("lpg_profit", "%g"%(var_name.X))
    
    var_name = model.getVarByName("jetfuel_profit")
    print("jetfuel_profit", "%g"%(var_name.X))           
            
        
        
            
            
            
            
            
            
            
