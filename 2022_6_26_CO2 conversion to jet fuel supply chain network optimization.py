#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 26 16:12:33 2022

@author: rachelriri
"""
import numpy as np
from gurobipy import * # Import Gurobi solver
import pandas as pd
import xlrd

# Creating Model===================================

model = Model('Linear Program')

# Set the integrality tolerance
model.Params.OptimalityTol = 0.01  # Adjust the value as need

#Index========================================================================
Region = 'Midwest'
region = 'mid west'
I = 211 #U.S. stationary CO2 sources 
J = 250 #U.S. potential conversion facilities 126 or 554
A = 8 #U.S. airports
P = 2#(2689.7,0.75,34885531,5.23)#CO2 converson pathway set

# Region = 'DEN'
# region = 'DEN'
# I = 26 #U.S. stationary CO2 sources 
# J = 36 #U.S. potential conversion facilities 126 or 554
# A = 1 #U.S. airports
# P = 2#(2689.7,0.75,34885531,5.23)#CO2 converson pathway set

# Region = 'MSP'
# region = 'MSP'
# I = 60 #U.S. stationary CO2 sources 
# J = 87 #U.S. potential conversion facilities 126 or 554
# A = 1 #U.S. airports
# P = 2#(2689.7,0.75,34885531,5.23)#CO2 converson pathway set

# Region = 'MCI'
# region = 'MCI'
# I = 39 #U.S. stationary CO2 sources 
# J = 54 #U.S. potential conversion facilities 126 or 554
# A = 1 #U.S. airports
# P = 2#(2689.7,0.75,34885531,5.23)#CO2 converson pathway set

# Region = 'Northeast'
# region = 'north east'
# I = 223 #U.S. stationary CO2 sources 
# J = 247 #U.S. potential conversion facilities 31 or 309
# A = 8 #U.S. airports
# P = 2#(2689.7,0.75,34885531,5.23)#CO2 converson pathway set

# Region = 'CLT_RDU'
# region = 'CLT_RDU'
# I = 46 #U.S. stationary CO2 sources 
# J = 63 #U.S. potential conversion facilities 126 or 554
# A = 2 #U.S. airports
# P = 2#(2689.7,0.75,34885531,5.23)#CO2 converson pathway set

# Region = 'BNA_ATL'
# region = 'BNA_ATL'
# I = 50 #U.S. stationary CO2 sources 
# J = 68 #U.S. potential conversion facilities 126 or 554
# A = 2 #U.S. airports
# P = 2#(2689.7,0.75,34885531,5.23)#CO2 converson pathway set

# Region = 'Southeast'
# region = 'south east_1'
# I = 48 #U.S. stationary CO2 sources 
# J = 58 #U.S. potential conversion facilities 189 or  744
# A = 2 #U.S. airports
# P = 2#(2689.7,0.75,34885531,5.23)#CO2 converson pathway set

# Region = 'Southeast'
# region = 'south east_2'
# I = 48 #U.S. stationary CO2 sources 
# J = 58 #U.S. potential conversion facilities 189 or  744
# A = 3 #U.S. airports
# P = 2#(2689.7,0.75,34885531,5.23)#CO2 converson pathway set

# Region = 'MSY'
# region = 'MSY'
# I = 68 #U.S. stationary CO2 sources 
# J = 84 #U.S. potential conversion facilities 80 or 183
# A = 1 #U.S. airports
# P = 2#(2689.7,0.75,34885531,5.23)#CO2 converson pathway set

# Region = 'Southwest'
# region = 'south west'
# I = 121 #U.S. stationary CO2 sources 
# J = 142 #U.S. potential conversion facilities 80 or 183
# A = 6 #U.S. airports
# P = 2#(2689.7,0.75,34885531,5.23)#CO2 converson pathway set

# Region = 'West'
# region = 'west'
# I = 76 #U.S. stationary CO2 sources 
# J = 98 #U.S. potential conversion facilities 199 or 476
# A = 5 #U.S. airports
# P = 2#(2689.7,0.75,34885531,5.23)#CO2 converson pathway set

# Region = 'SFO'
# region = 'SFO'
# I = 60 #U.S. stationary CO2 sources 
# J = 80 #U.S. potential conversion facilities 199 or 476
# A = 4 #U.S. airports
# P = 1#(2689.7,0.75,34885531,5.23)#CO2 converson pathway set

# Region = 'SLC'
# region = 'SLC'
# I = 12 #U.S. stationary CO2 sources 
# J = 31 #U.S. potential conversion facilities 199 or 476
# A = 1 #U.S. airports
# P = 2#(2689.7,0.75,34885531,5.23)#CO2 converson pathway set

# Region = 'SEA_PDX'
# region = 'SEA_PDX'
# I = 20 #U.S. stationary CO2 sources 
# J = 35 #U.S. potential conversion facilities 199 or 476
# A = 2 #U.S. airports
# P = 2#(2689.7,0.75,34885531,5.23)#CO2 converson pathway set

#Parameter====================================================================
#annual jet fuel demand in airprt a (Ton)
#path = '/Users/User/OneDrive - University of Tennessee/CO2/Data/CO2 supply chain network/CO2-CO-FTL-' #win
path = '/Users/rachelriri/Library/CloudStorage/OneDrive-UniversityofTennessee/CO2/Data/CO2 supply chain network/CO2-CO-FTL-' #mac

# jet fuel demand in airports
D_a = pd.read_excel(path+ Region +'/Jet fuel demand of '+str(A)+' major airports in '+region+'.xlsx').values.ravel()

#CO2 Conversion rate through conversion pathway s (kg/ton).
alpha_s = pd.read_excel(path + Region +'/Conversion rate_compare_2.xlsx').values.ravel()

#Maximum annual conversion capacity in potential conversion facilities  j through conversion pathway s (Ton).
C_sj = pd.read_excel(path + Region +'/refinery capacity '+str(J)+'_compare.xlsx')
C_sj = (np.asmatrix(C_sj.iloc[:,1:]))/159*7.9#1 barrel =159kg,1 barrel = 7.9 ton

#The unit CO2 capture costs at stationary CO2 source i ($/ton)
#c_i_c = pd.read_excel(path_mac + Region +'/CO2 capture cost in 277 sources.xlsx').values.ravel()
c_i_c = pd.read_excel(path + Region +'/CO2 capture cost in '+str(I)+' sources.xlsx')#.values.ravel()
c_i_c = np.asmatrix(c_i_c.iloc[:,1:])

#The unit operating costs of potential conversion facilities  j through conversion pathway s ($/ton)
c_sj_o = pd.read_excel(path + Region +'/Unit operating cost of more refineries_compare_state.xlsx')
c_sj_o = np.asmatrix(c_sj_o.iloc[:,1:].values)

#The fixed annual capital costs of potential conversion facilities  j through conversion pathway s ($).
c_sj_f = pd.read_excel(path + Region +'/Annual capital cost of '+str(J)+' refineries_compare.xlsx')
c_sj_f = np.asmatrix(c_sj_f.iloc[:2,1:].values)

#annual CO2 emission amount in staionary source i (Ton)
E_i = pd.read_excel(path + Region +'/CO2 emission from power plant in '+region+'.xlsx').values.ravel()*1.1#*1.1 to convert metric ton to ton

#The unit transportation cost from stationary CO2 source i to potential conversion facilities  j by pipeline ($/ton).
c_ij_l = pd.read_excel(path + Region +'/Unit trnasportation cost between source and more refinery in '+region+'.xlsx')
c_ij_l = np.asmatrix(c_ij_l.iloc[:,1:])

#The unit transportation cost from potential conversion facilities  j to airport a by truck ($/gallon).
#distance
d_ja = pd.read_excel(path + Region +'/Distance between more refinery and airport in '+region+'.xlsx')
d_ja = np.asmatrix(d_ja.iloc[:,1:])

#d_ja = np.asmatrix(d_ja.where(d_ja <= 300, 10000000000000))


#transportation cost from refinery to airport
#variable cost    
c_ja_v = 0.028*d_ja*0.62 #convert km to mile
#fixed cost
c_ja_f = 0.36 

#The average jet fuel sale price per ton ($/t). 2022 price is 1144.7$/mt 1mt=1.1t

p = 1144.7*1.1
M = 1000000

#Decision variables===========================================================
#The captured CO2 flow transported to potential conversion facilities  j from stationary CO2 source i (Ton).
x_ij = {} 
for i in range(0,I):
    for j in range(0,J):
        x_ij[i,j] = model.addVar(lb=0.0,ub=GRB.INFINITY,vtype=GRB.CONTINUOUS,name="x_ij%s,%s"%(i,j))

#The converted jet fuel transported to airport k from potential conversion facilities  j (gallon).
y_ja = {} 
for j in range(0,J):
    for a in range(0,A):
        y_ja[j,a] = model.addVar(lb=0.0,ub=GRB.INFINITY,vtype=GRB.CONTINUOUS,name="y_ja%s,%s"%(j,a))

#Binary variable, =1 if converstion pathway s is adopted at facility " = 0 "otherwise" )
z_sj = {} 
for s in range(0,P):
    for j in range(0,J):
        z_sj[s,j] = model.addVar(vtype=GRB.BINARY,name="z_sj%s,%s"%(s,j))

#capture cost
capture_cost = model.addVar(lb=0.0,ub=GRB.INFINITY,vtype=GRB.CONTINUOUS,name="capture_cost")

#pipeline cost
pipeline_cost = model.addVar(lb=0.0,ub=GRB.INFINITY,vtype=GRB.CONTINUOUS,name="pipeline_cost")

#operation cost
operation_cost = model.addVar(lb=0.0,ub=GRB.INFINITY,vtype=GRB.CONTINUOUS,name="operation_cost")

#capital cost
capital_cost = model.addVar(lb=0.0,ub=GRB.INFINITY,vtype=GRB.CONTINUOUS,name="capital_cost")

#truck cost
truck_cost = model.addVar(lb=0.0,ub=GRB.INFINITY,vtype=GRB.CONTINUOUS,name="truck_cost")



#Objective function==========================================maintains conversion balance at each facility j=================
model.setObjective(quicksum(c_i_c[i] * x_ij[i,j] for i in range (0,I) for j in range(0,J))
                 + quicksum(c_ij_l[i,j] * x_ij[i,j] for i in range(0,I) for j in range(0,J))
                 + quicksum(c_sj_o[s,j] * x_ij[i,j] for s in range (0,P) for i in range(0,I) for j in range(0,J))
                 + quicksum(c_sj_f[s,j] * z_sj[s,j] for s in range(0,P) for j in range(0,J))
                 + quicksum(0.028*d_ja[j,a]*0.62 * y_ja[j,a] for j in range(0,J) for a in range (0,A))
                 + quicksum(c_ja_f * y_ja[j,a] for j in range(0,J) for a in range (0,A))
                 - quicksum(p * y_ja[j,a] for j in range(0,J) for a in range(0,A)),GRB.MINIMIZE)

#Constrians===================================================================
#Maintains conversion balance at each facility j
#for j in range(0,J):
#    model.addConstr(quicksum(M*(1-z_sj[s,j])+alpha_s[s] * x_ij[i,j] for i in range(0,I) for s in range(0,P))
#                    == quicksum(y_ja[j,a] for a in range(0,A)))

  

#Maintains conversion balance at each facility j
for j in range(0,J):
    model.addConstr(quicksum(alpha_s[s] * x_ij[i,j] for i in range(0,I) for s in range(0,P))
                    ==quicksum(y_ja[j,a] for a in range(0,A)))

#enforces the potential facilities conversion capacity
for j in range(0,J):
    model.addConstr(quicksum(x_ij[i,j] for i in range(0,I)) <= quicksum(C_sj[s,j] * z_sj[s,j] for s in range(0,P)))

#the amount of captured CO2 does not exceed the available of CO2 emissions
for i in range(0,I):
    model.addConstr(quicksum(x_ij[i,j] for j in range(0,J)) <= E_i[i])

#the converted jet fuel shipped to an airport is bounded by the airports’ demand
for a in range(0,A):
    model.addConstr(quicksum(y_ja[j,a] for j in range(0,J)) >= D_a[a])
    
#each potential conversion facility only use one type of conversion pathway, which defines a quadruplet {annual capacity, conversion rate, fixed annual cost, unit conversion cost per ton}.
for j in range(0,J):
    model.addConstr(quicksum(z_sj[s,j] for s in range(0,P)) <= 1)


#model.addConstrs(z_sj[s,j] <= 1 for s in range(0,P)for j in range(0,J))  
#model.addConstrs( d_ja[j,a] * z_sj[1,j] <=300 for a in range (0,A) for j in range (0,J) )
# for a in range(0,A):
#     model.addConstr(quicksum(d_ja[j,a] for j in range(0,J)) <=300)

#capture cost
model.addConstr(quicksum(c_i_c[i] * x_ij[i,j] for i in range (0,I) for j in range(0,J))
                    == capture_cost)

#pipeline cost
model.addConstr(quicksum(c_ij_l[i,j] * x_ij[i,j] for i in range(0,I) for j in range(0,J))
                    == pipeline_cost)

#operation cost
model.addConstr(quicksum(c_sj_o[s,j] * x_ij[i,j] for s in range (0,P) for i in range(0,I) for j in range(0,J))
                    == operation_cost)


#capital_cost
model.addConstr(quicksum(c_sj_f[s,j] * z_sj[s,j] for s in range(0,P) for j in range(0,J))
                    == capital_cost)

#truck cost
model.addConstr(quicksum(0.028*d_ja[j,a]*0.62 * y_ja[j,a] for j in range(0,J) for a in range (0,A))
              + quicksum(c_ja_f * y_ja[j,a] for j in range(0,J) for a in range (0,A))
                    == truck_cost)




model.write('model.lp') 
model.optimize() 

# Specify the directory path and file name
directory_path = "/Users/rachelriri/Library/CloudStorage/OneDrive-UniversityofTennessee/CO2/Code/Code_A/Result"

# Combine the directory path and file name to create the full file path
file_path = os.path.join(directory_path, 'CO2 conversion to jet fuel CO2_compare_state in '+region+'.sol')

# Save the model to the specified file path
model.write(file_path)



# Print solution==========================================================================================================
print("======objective value =======")
obj = model.getObjective()
print(obj.getValue())

print("====== cost =======")
var_name = model.getVarByName("capture_cost")
print("capture_cost", "%g"%(var_name.X))

var_name = model.getVarByName("pipeline_cost")
print("pipeline_cost", "%g"%(var_name.X))

var_name = model.getVarByName("operation_cost")
print("operation_cost", "%g"%(var_name.X))

var_name = model.getVarByName("capital_cost")
print("capital_cost", "%g"%(var_name.X))

var_name = model.getVarByName("truck_cost")
print("truck_cost", "%g"%(var_name.X))



