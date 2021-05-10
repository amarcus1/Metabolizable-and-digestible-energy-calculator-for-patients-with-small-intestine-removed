# -*- coding: utf-8 -*-
"""
Andrew Marcus Gut Model (Python version 1.0)
Authors: Andrew Marcus & Taylor Davis
Date Created: 05-03-21
"""


'''
This complementary python script implements the complete calculations in our 
work. It considers one scenario at a time, but a user can alter parameters in the 
"Parameters.csv" document to consider other scenarios. 
'''

# Load necessary packages
import pandas as pd
import numpy as np


# =============================================================================
# FUNCTIONS USED WITHIN MODEL
# =============================================================================

def load_variables():
    '''
    Load the csv documents containing needed variables into dataframes

    Returns
    -------
    para : pandas DataFrame
        Contains all parameters used within the model.
    diet : pandas DataFrame
        Contains all dietary input variables.
    comp : pandas DataFrame
        Contains the stoichiometry, gCOD/d, enthalpy of combustion, 
        and other data for tracked components.

    '''
    # Load the parameters file, input parameter values into a series with symbols as index
    parameters = pd.read_csv("Parameters.csv")
    para = parameters.set_index("Symbol")
    para.drop(["Description", "Units", "Note"], axis=1, inplace=True)
    
    
    # Load the dietary input file
    diet_intake = pd.read_csv("Dietary Input.csv")
    diet = diet_intake.set_index("Symbol")
    diet.drop(["Description", "Units", "Note"], axis=1, inplace=True)
    
    
    # Load the dietary input file
    components = pd.read_csv("Components.csv")
    comp = components.set_index("Compound")
    
    return para, diet, comp


def UGI_calcs(para, diet, comp):
    '''
    Uppergastrointestinal tract calculations

    Parameters
    ----------
    para : pandas DataFrame
        Contains all parameters used within the model.
    diet : pandas DataFrame
        Contains all dietary input variables.
    comp : pandas DataFrame
        Contains the stoichiometry, gCOD/d, enthalpy of combustion, 
        and other data for tracked components.

    Returns
    -------
    UGI : pandas DataFrame
        Contains the input, absorption, and output of tracked components within
        the upper gastrointestinal tract in gCOD/d.

    '''
    
    # Define the componenets to track throughout model
    track_comp = ["NSP", "RS", "AvSS", "C_sec", "P", "P_sec", "F", "F_sec"]
    
    # Calculate the dietary input in gCOD/d for all tracked components
    diet_input = [diet.loc["NSP"]*diet.loc["CI"]*comp.loc["Carbohydrate","gCOD/g"]/comp.loc["Carbohydrate","Enthalpy of Combustion"],
                  diet.loc["RS"]*diet.loc["CI"]*comp.loc["Carbohydrate","gCOD/g"]/comp.loc["Carbohydrate","Enthalpy of Combustion"],
                  diet.loc["AvSS"]*diet.loc["CI"]*comp.loc["Carbohydrate","gCOD/g"]/comp.loc["Carbohydrate","Enthalpy of Combustion"],
                  diet.loc["PI"]*comp.loc["Protein","gCOD/g"]/comp.loc["Protein","Enthalpy of Combustion"],
                  diet.loc["FI"]*comp.loc["Fat","gCOD/g"]/comp.loc["Fat","Enthalpy of Combustion"]]
    
    # Convert to list of values rather than series object
    diet_input = [diet_input[0].values[0], 
                  diet_input[1].values[0], 
                  diet_input[2].values[0], 
                  0.0, 
                  diet_input[3].values[0], 
                  0.0, 
                  diet_input[4].values[0],
                  0.0]
    
    # Calculate absorption within the UGI tract in gCOD/d for all tracked components
    absorption = [diet_input[2]*(para.loc["aAvSS"] - para.loc["bAvSS"]*max(0, para.loc["SIR"].values - para.loc["ARL_AvSS"].values)),
                  -para.loc["GIS_carb"]*comp.loc["Carbohydrate","gCOD/g"],
                  diet_input[4]*(para.loc["aprotein"] - para.loc["bprotein"]*max(0, para.loc["SIR"].values - para.loc["ARL_protein"].values)),
                  -para.loc["GIS_protein"]*comp.loc["Protein","gCOD/g"],
                  diet_input[6]*(para.loc["afat"] - para.loc["bfat"]*max(0, para.loc["SIR"].values - para.loc["ARL_fat"].values)),
                  -para.loc["GIS_fat"]*comp.loc["Fat","gCOD/g"]]
    
    # Convert to list of values rather than series object
    absorption = [0.0,
                  0.0,
                  absorption[0].values[0],
                  absorption[1].values[0],
                  absorption[2].values[0],
                  absorption[3].values[0],
                  absorption[4].values[0],
                  absorption[5].values[0]]

    # Create dataframe with gCOD/d data for UGI tract
    UGI = pd.DataFrame(np.array([diet_input, absorption]).T, columns = ["Diet_Input", "UGI_Absorption"], index = track_comp)
    
    # Calculate amount leaving the UGI, to ileocecal passage, in gCOD/d for all tracked components
    UGI["Ileocecal_Passage"] = UGI["Diet_Input"] - UGI["UGI_Absorption"]
    
    # Add a unit indicator column
    UGI["Units"] = "gCOD/d"
    
    # Return the UGI dataframe
    return UGI


def LGI_calcs(UGI, para, diet, comp):
    '''
    Lower Gastorintestinal tract calculations

    Parameters
    ----------
    UGI :  pandas DataFrame
        Contains the input, absorption, and output of tracked components within
        the upper gastrointestinal tract in gCOD/d.
    para : pandas DataFrame
        Contains all parameters used within the model.
    diet : pandas DataFrame
        Contains all dietary input variables.
    comp : pandas DataFrame
        Contains the stoichiometry, gCOD/d, enthalpy of combustion, 
        and other data for tracked components.

    Returns
    -------
    LGI :  pandas DataFrame
        Contains the input, absorption, and output of tracked components within
        the lower gastrointestinal tract in gCOD/d.

    '''
    # Hydrolysis of Carbohydrates in gCOD/d
    carb_hyd = [UGI.loc["NSP","Ileocecal_Passage"]/(1 + para.loc["q"]*para.loc["khyd_NSP"]),
                UGI.loc["RS","Ileocecal_Passage"]/(1 + para.loc["q"]*para.loc["khyd_RS"]),
                UGI.loc["AvSS","Ileocecal_Passage"]/(1 + para.loc["q"]*para.loc["khyd_RS"]),
                UGI.loc["C_sec","Ileocecal_Passage"]/(1 + para.loc["q"]*para.loc["khyd_RS"])]
    
    tot_carb_hyd = UGI.Ileocecal_Passage.iloc[:4].sum() \
        - (carb_hyd[0].values[0] + carb_hyd[1].values[0] + carb_hyd[2].values[0] + carb_hyd[3].values[0])
    
    # Hydrolysis of Proteins in gCOD/d
    prot_hyd = [UGI.loc["P","Ileocecal_Passage"]/(1 + para.loc["q"]*para.loc["khyd_P"]),
                UGI.loc["P_sec","Ileocecal_Passage"]/(1 + para.loc["q"]*para.loc["khyd_P"])]
    
    tot_prot_hyd = UGI.Ileocecal_Passage.iloc[4:6].sum() \
        - (prot_hyd[0].values[0] + prot_hyd[1].values[0])
    
    # Define products to be tracked within the LGI model 
    track_comp = ["NSP", "RS", "AvSS", "C_sec", "P", "P_sec", "F", "F_sec","X_C", "X_P", "Ace", "Prop","nBut","iBut","Aro"]
    
    # Microbial biomass and products formed in gCOD/d
    X_C = tot_carb_hyd*para.loc["fs0_C"].values[0]
    
    X_P = tot_prot_hyd*para.loc["fs0_P"].values[0]
    
    Ace = (tot_carb_hyd*(1 - para.loc["fs0_C"])*para.loc["fa (Ac)_C"] \
        + tot_prot_hyd*(1 - para.loc["fs0_P"])*para.loc["fa (Ac)_P"]).values[0]
    Prop = (tot_carb_hyd*(1 - para.loc["fs0_C"])*para.loc["fa (Pr)_C"] \
        + tot_prot_hyd*(1 - para.loc["fs0_P"])*para.loc["fa (Pr)_P"]).values[0]
    nBut = (tot_carb_hyd*(1 - para.loc["fs0_C"])*para.loc["fa (Bu)_C"]).values[0]
    
    iBut = (tot_prot_hyd*(1 - para.loc["fs0_P"])*para.loc["fa (Bu)_P"]).values[0]
    
    Aro = (tot_prot_hyd*(1 - para.loc["fs0_P"])*para.loc["fa (Aro)_P"]).values[0]
    
    # Absorbed fats and microbial products in gCOD/d
    fat_abs = [UGI.loc["F","Ileocecal_Passage"] - UGI.loc["F","Ileocecal_Passage"]/(1 + para.loc["q"]*para.loc["kabs_F"]),
               UGI.loc["F_sec","Ileocecal_Passage"] - UGI.loc["F_sec","Ileocecal_Passage"]/(1 + para.loc["q"]*para.loc["kabs_F"])]
    
    prod_abs = [0.95*Ace,
                0.95*Prop,
                0.95*nBut,
                0.95*iBut,
                0.0] # Assume 95% of SCFAs absorbed and 0% aromatics
    
    # Convert to list of values rather than series object
    absorption = [0.0,
                  0.0,
                  0.0,
                  0.0,
                  0.0,
                  0.0,
                  fat_abs[0].values[0],
                  fat_abs[1].values[0],
                  0.0,
                  0.0,
                  prod_abs[0],
                  prod_abs[1],
                  prod_abs[2],
                  prod_abs[3],
                  prod_abs[4]]
    
    # Convert to list of values rather than series object
    fecal = [carb_hyd[0].values[0],
             carb_hyd[1].values[0],
             carb_hyd[2].values[0],
             carb_hyd[3].values[0],
             prot_hyd[0].values[0],
             prot_hyd[1].values[0],
             fat_abs[0].values[0],
             fat_abs[1].values[0],
             X_C,
             X_P,
             Ace - prod_abs[0],
             Prop - prod_abs[1],
             nBut - prod_abs[2],
             iBut - prod_abs[3],
             Aro]
    
    # Create a dataframe with the LGI input (ileocecal passage), absorbed, and output (fecal) in gCOD/d
    LGI = pd.DataFrame(np.array([absorption, fecal]).T, columns = ["LGI_Absorption", "Fecal Output"], index = track_comp)
    
    # Add in UGI Ileocecal passage column
    LGI = pd.concat([UGI.Ileocecal_Passage, LGI], axis=1)
    
    # Add a unit indicator column
    LGI["Units"] = "gCOD/d"
    
    # Return the LGI dataframe
    return LGI.fillna(0)


def ME_calc(UGI,LGI):
    # Calculate ME from absorption columns
    ME = UGI.UGI_Absorption.sum() + LGI.LGI_Absorption.sum()
    return ME







# =============================================================================
# MAIN CALL TO TEST
# =============================================================================

# Load variable dataframes
para, diet, comp = load_variables()

# UGI calculations
UGI = UGI_calcs(para, diet, comp)

# LGI calculations
LGI = LGI_calcs(UGI, para, diet, comp)

# ME in gCOD/d
ME = ME_calc(UGI, LGI)








