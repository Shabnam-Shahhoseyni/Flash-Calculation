# -*- coding: utf-8 -*-
"""
Created on Wed Aug  2 00:48:32 2023
Flash Calculations
Ideal Solution & Ideal Gas phase
Data generation for Machine Learning purposes
The 12th International Chemical Engineering Congress & Exhibition (IChEC 2023)
@author: shabnam shahhoseyni
14 Aug 2023
Rev 03
"""
#------------------------------------------------------------------------------
import pandas as pd
import numpy as np
from scipy.optimize import root_scalar
import random

#------------------------------------------------------------------------------
''' Calculations part'''
''' Each component is considered as an object'''
class Component:
     def __init__(self, params):
         self.name, self.A, self.B, self.C = params
     def Psat (self, T):
         return 10** (self.A- self.B/(T+self.C))                               # Psat from Antoine equation

''' Each liquid mixture is considered as an object'''
class Mixture:
    def __init__(self, z, comps, T, P):
        self.z = list(z.values())                                              # mixture compositions
        self.comps = comps                                                     # mixture component objects
        self.T = T
        self.P = P

    ''' Bubble point Pressure calculation'''
    def Pt_bubble(self):
        #print('len (self.comps) in pt_bubble', len (self.comps))
        P_b = np.array([self.z[i] * self.comps[i].Psat(self.T) 
                        for i in np.arange(len (self.comps))])                 # P(i) = x(i) * Psat(i)
        self.y_bubble = [x / np.sum(P_b) for x in P_b]                         # vapor compositions at bubble point
        return np.sum(P_b)                                                     # bubble point pressure

    ''' Dew point Pressure calculation'''
    def Pt_dew(self):
        #print('len (self.comps) in pt_dew', len (self.comps))
        tmp = np.array([self.z[i] / self.comps[i].Psat(self.T) 
                        for i in np.arange(len (self.comps))])                 # tmp = y(i) / Psat(i)
        self.Pd = [1 / x for x in tmp]                                         # P(i)
        #print('len (self.Pd) in pt_dew', len (self.Pd))
        self.x_dew = np.array([(1 / np.sum(tmp)) / self.Pd[i]                  # 1/ ∑(y(i)/Psat(i))
                               for i in np.arange(len (self.Pd))])             # liquid compositions at dew point
        return 1 / np.sum(tmp)

    ''' Bubble point Temperature calculation'''
    def T_bubble(self):
        def f(Tb):
            W = 0
            #print('len (M) in T_bubble', len (M))
            for i in np.arange (len (M)):
                W += list(feed_comps.values())[i] * M[i].Psat(Tb)
            return self.P - W

        return root_scalar(f, bracket=[273, 500]).root                         # T_bubble

    ''' Dew point Temperature calculation'''
    def T_dew(self):
        def f(Td):
            W = 0   
            #print('len (M) in T_dew', len (M))                                # y(i)/Psat(i)
            for i in np.arange (len (M)):
                W += list(feed_comps.values())[i] / M[i].Psat(Td)              # ∑y(i)/Psat(i)
            return 1 / self.P - W                                              # 1/P-∑y(i)/Psat(i)

        return root_scalar(f, bracket=[273, 500]).root                         # T_dew
    
#------------------------------------------------------------------------------
''' Flash calculations as a function'''

def flash_calc(VF, z, K):                                                      # VF is vapor split
    RR = 0
    #print('len (z) in flash_calc', len (z))
    for i in np.arange(len(z)):
        RR += (z[i] * (K[i] - 1)) / (1 + VF * (K[i] - 1))                      # Rachford-Rice flash equation
    return RR

#------------------------------------------------------------------------------

''' Start of the main'''

# Reading the Antoine database
data = pd.read_excel('Antoine_database-data generation.xls')
comps_data = data[['Hydrocarbon_Name', 'A', 'B', 'C']]

#------------------------------------------------------------------------------

''' Data generation for different situations'''
# Define the range of parameters
num_situations = 1000                                                          # Number of diverse situations to generate
T_range = (273, 400)                                                           # Temperature range (Kelvin)
P_range = (1, 5)                                                              # Pressure range (bar)

situation_data = []
for i in range(num_situations):
    # Randomly choose the number of hydrocarbons in the mixture
    num_hydrocarbons = random.randint(2, len(comps_data.index))

    # Randomly select the hydrocarbons for the mixture without replacement
    selected_hydrocarbons = comps_data.sample(num_hydrocarbons)

    # Randomly assign compositions to each hydrocarbon in the mixture
    proportions = np.random.dirichlet(np.ones(num_hydrocarbons))

    # Create a dictionary to store the mixture composition (hydrocarbon: proportion)
    feeds = {hc: proportion for hc, proportion in zip(
        selected_hydrocarbons['Hydrocarbon_Name'].tolist(), proportions)}
    
    # Randomly choose the temperature and pressure for the situation
    T = random.uniform(T_range[0], T_range[1])
    P = random.uniform(P_range[0], P_range[1])

    situation_data.append({
        'Situation': i + 1,
        'Temperature (K)': T,
        'Pressure (bar)': P,
        'Number of Hydrocarbons': num_hydrocarbons,
        **feeds  # Mixture composition
    })

# Create a DataFrame from the situation_data list
result_df = pd.DataFrame(situation_data)
#------------------------------------------------------------------------------

# Calculate flash calculations for each situation
for index, row in result_df.iterrows():
    
    T = row['Temperature (K)']
    P = row['Pressure (bar)']
    M=[]
    feed_comps={}
     
    # Iterate over columns and check for NaN values
    for col_name, value in row.iloc[4:].items():
         
        if pd.notna(value):  # Check if the value is not NaN
            search_string = col_name
            
            # Filter rows where the search string appears in any column
            matching_rows = comps_data[comps_data.apply(
                lambda row: search_string in row.values, axis=1)]
            
            #feed_comps dic generation (names and compositions)
            feed_comps[col_name] = value
                                  
            # Convert the first matching row to a list
            if not matching_rows.empty:
                first_matching_row = matching_rows.iloc[0].tolist()
                component=Component(first_matching_row)
                M.append(component)
            else:
                print("No matching rows found.")
            
   
    # Create an object from 'Mixture class' for the generated mixture
    mixture_obj = Mixture(feed_comps, M, T, P)
    
    try:
        # Calculate bubble point and dew point values
        result_df.at[index, 'Bubble Pres. (bar)'] = mixture_obj.Pt_bubble()
        result_df.at[index, 'Dew Pres. (bar)'] = mixture_obj.Pt_dew()
        result_df.at[index, 'Bubble Temp (K)'] = mixture_obj.T_bubble()
        result_df.at[index, 'Dew Point Temp. (K)'] = mixture_obj.T_dew()
        
        # Check if flash occurs
        if (P < mixture_obj.Pt_bubble() and P > mixture_obj.Pt_dew()) and (
                T > mixture_obj.T_bubble() and T < mixture_obj.T_dew()):
            K = [x.Psat(T) / P for x in M]                                     # equilibrium K-values
            z = list(feed_comps.values())
            vf = root_scalar(flash_calc, (z, K), bracket=[0, 1])               # vapor split (V/F)
            x = np.array([z[i] / (1 + vf.root * (K[i] - 1)) 
                          for i in range(len(K))])                             # liquid compositions @ flash
            y = np.array([K[i] * x[i] for i in range(len(K))])                 # vapor compositions @ flash
    
            result_df.at[index, 'Vapor Split (V/F)'] = vf.root
            result_df.at[index, 'Liquid Compositions'] = ','.join([f'{comp:.3f}' for comp in x])
            result_df.at[index, 'Vapor Compositions'] = ','.join([f'{comp:.3f}' for comp in y])
    
        elif P > mixture_obj.Pt_bubble() and T < mixture_obj.T_bubble():
            result_df.at[index, 'Vapor Split (V/F)'] = 0
            result_df.at[index, 'Liquid Compositions'] = "Nothing will flash ('V/F=0')"
            result_df.at[index, 'Vapor Compositions'] = ""
        else:
            result_df.at[index, 'Vapor Split (V/F)'] = 1
            result_df.at[index, 'Liquid Compositions'] = "Everything flashes ('V/F=1')"
            result_df.at[index, 'Vapor Compositions'] = ""
    
        

        # Save the DataFrame to an Excel file
        result_df.to_excel('flash_calculation_results.xlsx', index=False)
    
        print("Flash calculation results saved to flash_calculation_results.xlsx")

    except:
        print('bad compositions')

#------------------------------------------------------------------------------

