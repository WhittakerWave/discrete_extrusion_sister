

import numpy as np
import sympy as sym
from sympy import symbols, Eq, pprint, init_printing
import scipy.integrate
import matplotlib.pyplot as plt
import tellurium as te
import pandas as pd
from IPython.display import display
sym.init_printing()

residence_time = 10
## Rates for cohesive network 
rates_coh = sym.symbols("K_RacP_RacPW, K_RacPW_RacP, K_RacP_RacPS, K_RacPS_RacP, K_RacP_Rac, K_Rac_RacP, K_Rac_RacN, K_RacN_Rac, K_RacPW_Rac_free")
K_RacP_RacPW, K_RacPW_RacP, K_RacP_RacPS, K_RacPS_RacP, K_RacP_Rac, K_Rac_RacP, K_Rac_RacN, K_RacN_Rac, K_RacPW_Rac_free = rates_coh

paras_coh = sym.symbols("tau_S, F_S, N_S, tau_W, F_W, N_W, tau_P, F_P, N_P, tau_N, F_N, N_N, tau_R, F_R_sister, N_R")
tau_S, F_S, N_S, tau_W, F_W, N_W, tau_P, F_P, N_P, tau_N, F_N, N_N, tau_R, F_R_sister, N_R = paras_coh 

## Equations for cohesive network
rhs_coh = [K_RacPW_Rac_free * N_W * F_W - F_R_sister*N_R/tau_R,                              # RAD21 unbinding  
           F_N/(tau_N * (1-F_N)) - K_Rac_RacN * (F_R_sister * N_R - F_P * N_P - F_N * N_N),  # Nipbl binding
           K_RacN_Rac - 1/tau_N,                                                             # Nipbl unbinding 
           F_W/(tau_W * (1-F_W)) - K_RacP_RacPW * (F_P * N_P - F_S * N_S - F_W * N_W),       # Wapl binding 
           K_RacPW_RacP + K_RacPW_Rac_free - 1/tau_W,                                        # Wapl unbinding  
           F_S/(tau_S * (1-F_S)) - K_RacP_RacPS * (F_P * N_P - F_S * N_S - F_W * N_W),       # Sororin binding
           K_RacPS_RacP - 1/tau_S,                                                           # Sororin unbinding
           F_P/(tau_P * (1-F_P)) - K_Rac_RacP * (F_R_sister * N_R - F_P * N_P - F_N * N_N),                  # PDS5 binding
           K_RacPW_Rac_free * F_W *N_W + K_RacP_Rac*(F_P * N_P - F_S * N_S - F_W * N_W) - F_P * N_P/tau_P,   # PDS5 unbinding  
]

## solutions cohesive network
sol_rates_coh = sym.solve(rhs_coh, rates_coh)

for i, eq in enumerate(rhs_coh, 1):
    pprint(Eq(symbols(f"eq_{i}"), eq))

## parameters for cohesive network
# Define the input parameters
x = 0.1306   # for W
y = 0.3067  # for P  
z = 0.1536  # for N

paras_values_coh = [
    100.,                  # tau_S
    0.52,                  # F_S
    79770,                 # N_S
    45.,                   # tau_W
    x/(0.65 + x),          # modified F_W for sister 
    69542*(0.65 + x),      # modified N_W for sister
    72,                    # tau_P
    y/(0.58 + y),          # modified F_P for sister
    180615*(0.58 + y),     # modified N_P for sister
    72.,                   # tau_N
    z/(0.6 + z),           # modified F_N for sister
    119308*(0.6 + z),      # modified N_N for sister
    3600 * residence_time,             # tau_R_sister, 6h 
    1/2,                   # modified F_R_sister
    284470*2/3,            # modified N_R
]

paras_dict_coh = dict(zip(paras_coh, paras_values_coh))

for s in sol_rates_coh.items():
    rate = s[1].evalf(subs=paras_dict_coh)
    paras_dict_coh[s[0]] = rate

## print the rates for the cohesive network
paras_dict_coh = dict(zip(paras_coh, paras_values_coh))
for s in sol_rates_coh.items():
    rate = s[1].evalf(subs=paras_dict_coh)
    paras_dict_coh[s[0]] = rate
    display(s)
    print(rate)

## Rates for extrusive network
rates_ext = sym.symbols("Kext_R_free_RN, Kext_RN_R, Kext_R_RN, Kext_R_RP, Kext_RP_R, Kext_RP_RPW, Kext_RPW_RP, Kext_RPW_R_free")
Kext_R_free_RN, Kext_RN_R, Kext_R_RN, Kext_R_RP, Kext_RP_R, Kext_RP_RPW, Kext_RPW_RP, Kext_RPW_R_free = rates_ext

paras_ext = sym.symbols("tau_N_ext, F_N_ext, N_N_ext, tau_W_ext, F_W_ext, N_W_ext, tau_P_ext, F_P_ext, N_P_ext, tau_R_ext, F_R_ext, N_R_ext")
tau_N_ext, F_N_ext, N_N_ext, tau_W_ext, F_W_ext, N_W_ext, tau_P_ext, F_P_ext, N_P_ext, tau_R_ext, F_R_ext, N_R_ext = paras_ext

## Equations for extrusive network
rhs_ext = [ Kext_RN_R - 1 / tau_N_ext,  # NIPBL unbinding kinetics
        Kext_R_free_RN * N_R_ext *(1-F_R_ext) + Kext_R_RN * (N_R_ext*F_R_ext - N_N_ext*F_N_ext - N_P_ext*F_P_ext) -  F_N_ext / tau_N_ext/ (1-F_N_ext), # NIPBL binding equilibrium
        Kext_RP_R * (N_P_ext * F_P_ext - N_W_ext * F_W_ext) + Kext_RPW_R_free * N_W_ext * F_W_ext  - N_P_ext * F_P_ext/tau_P_ext, # PDS5 unbinding kinetics
        Kext_R_RP * (N_R_ext * F_R_ext - N_N_ext * F_N_ext - N_P_ext * F_P_ext)  - F_P_ext / tau_P_ext /(1-F_P_ext), # PDS5 binding equilibrium,
        Kext_RPW_RP + Kext_RPW_R_free - 1 / tau_W_ext, # WAPL unbinding kinetics
        Kext_RP_RPW * (N_P_ext * F_P_ext - N_W_ext* F_W_ext)  - F_W_ext / tau_W_ext / (1-F_W_ext), # WAPL binding equilibrium
        Kext_RPW_R_free * N_W_ext * F_W_ext - F_R_ext * N_R_ext / tau_R_ext,  # RAD21 unbinding kinetics
        Kext_R_free_RN * N_N_ext * (1-F_N_ext) - F_R_ext / tau_R_ext / (1-F_R_ext) # RAD21 binding equilibrium, 
]

sol_rates_ext = sym.solve(rhs_ext, rates_ext)
# Print equations
for i, eq in enumerate(rhs_ext, 1):
    pprint(Eq(symbols(f"eq_{i}"), eq))

paras_values_ext = [
    72.,                          # tau_N
    (0.4 - z)/(1 - z),            # F_N
    119308*(1 - z),               # N_N
    45.,                          # tau_W
    (0.35 - x)/(1 - x),           # modified F_W for extrusive
    69542*(1 - x),                # modified N_W for extrusive
    72,                           # tau_P
    (0.42 - y)/(1 - y),           # modified F_P for extrusive
    180615*(1 - y),               # modified N_P for extrusive
    822.,                         # tau_R_extrusive
    1/2,                          # modified F_R_sister
    284470*2/3,                   # modified N_R
]

paras_dict_ext = dict(zip(paras_ext, paras_values_ext))

for s in sol_rates_ext.items():
    rate = s[1].evalf(subs=paras_dict_ext)
    paras_dict_ext[s[0]] = rate

### print the rates for the extrusive network
paras_dict_ext = dict(zip(paras_ext, paras_values_ext))
for s in sol_rates_ext.items():
    rate = s[1].evalf(subs=paras_dict_ext)
    paras_dict_ext[s[0]] = rate
    display(s)
    print(rate)

## models for combined cohesive and extrusive networks
model_ext_coh ='''
    # Define species and parameters
    
    # Cohesive network 
    Rac + N -> RacN; K_Rac_RacN*Rac*N - K_RacN_Rac*RacN
    Rac + P -> RacP; K_Rac_RacP*Rac*P - K_RacP_Rac*RacP
    RacP + S -> RacPS; K_RacP_RacPS*RacP*S - K_RacPS_RacP*RacPS
    RacP + W -> RacPW; K_RacP_RacPW*RacP*W - K_RacPW_RacP*RacPW
    RacPW -> Rac_free + P + W; K_RacPW_Rac_free*RacPW 
    
    # Extrusive network 
    R_free + N -> RN; Kext_R_free_RN*R_free*N 
    RN  -> R + N; Kext_RN_R*RN - Kext_R_RN*R*N
    R + P -> RP; Kext_R_RP*R*P - Kext_RP_R*RP
    RP + W -> RPW; Kext_RP_RPW*RP*W - Kext_RPW_RP*RPW 
    RPW -> R_free + P + W; Kext_RPW_R_free*RPW
    
    # deacetylation
    # Rac_free -> R_free; K_Rac_free_R_free*Rac_free
    
    # Rates for cohesive network,  9 rates
    K_Rac_RacN = {K_Rac_RacN};
    K_RacN_Rac = {K_RacN_Rac};
    K_Rac_RacP = {K_Rac_RacP}; 
    K_RacP_Rac = {K_RacP_Rac}; 
    K_RacP_RacPS = {K_RacP_RacPS}; 
    K_RacPS_RacP = {K_RacPS_RacP}; 
    K_RacP_RacPW = {K_RacP_RacPW}; 
    K_RacPW_RacP = {K_RacPW_RacP};  
    K_RacPW_Rac_free = {K_RacPW_Rac_free}; 

    # Rates for extrusive network, 8 rates
    Kext_R_free_RN = {Kext_R_free_RN};
    Kext_RN_R = {Kext_RN_R};
    Kext_R_RN = {Kext_R_RN};
    Kext_R_RP = {Kext_R_RP};
    Kext_RP_R = {Kext_RP_R};
    Kext_RP_RPW = {Kext_RP_RPW};
    Kext_RPW_RP = {Kext_RPW_RP};
    Kext_RPW_R_free = {Kext_RPW_R_free};

    # K_Rac_free_R_free = {K_Rac_free_R_free}; 
    
    # Initial conditions
    Rac_free = {Rac_free_init}; 
    Rac = {Rac_init}; 
    RacP = {RacP_init}; 
    RacN = {RacN_init}; 
    RacPW = {RacPW_init}; 
    RacPS = {RacPS_init}; 

    R_free = {R_free_init};
    RN = {RN_init};
    R = {R_init};
    RP = {RP_init};
    RPW = {RPW_init};
    
    N = {N_init};
    S = {S_init}; 
    W = {W_init}; 
    P = {P_init};
'''

def build_model_ext_coh(model, parameter_dict):
    string_params = {str(k): v for k, v in parameter_dict.items()}
    return model.format(**string_params)


# Define symbolic initial conditions
Rac_free_init, Rac_init, RacN_init, RacP_init, RacPW_init, RacPS_init, R_free_init, RN_init, R_init, RP_init, RPW_init, N_init, S_init, W_init, P_init = sym.symbols(
    "Rac_free_init, Rac_init, RacN_init, RacP_init, RacPW_init, RacPS_init, R_free_init, RN_init, R_init, RP_init, RPW_init, N_init, S_init, W_init, P_init")

init_conditions_ext_coh = {
    Rac_free_init: 0, 
    Rac_init: paras_dict_coh[N_R]*1/2, 
    RacN_init: 0, 
    RacP_init: 0,
    RacPW_init: 0,
    RacPS_init: 0,

    R_free_init: paras_dict_coh[N_R],
    RN_init: 0,
    R_init: 0,
    RP_init: 0, 
    RPW_init: 0, 
    
    N_init: 119308,
    S_init: paras_dict_coh[N_S],
    W_init: 69542,
    P_init: 180615,
}

paras_dict = paras_dict_coh | paras_dict_ext | init_conditions_ext_coh | {"K_Rac_free_R_free": 1/2495}
paras_dict_ext_coh = {str(key): value for key, value in paras_dict.items()}
Model_ext_coh = build_model_ext_coh(model_ext_coh, paras_dict_ext_coh)
# print(model_ext_coh)
# Load the modes
r_ext_coh = te.loada(Model_ext_coh)
# Simulate the model
Model_ext_coh_WT = r_ext_coh.simulate(0, 3600*18, 3600*18)

columns = ['time', 'Rac', 'N', 'RacN', 'P', 'RacP', 'S', 'RacPS', 'W', 'RacPW', 
           'Rac_free', 'R_free', 'RN', 'R', 'RP', 'RPW']
df_WT = pd.DataFrame(Model_ext_coh_WT, columns=columns)

num_sisterCs = 7765 
lattice_size = 320000 

for h in range(19):  # 0h to 18h 
    index = 3600 * h - 1  if h > 0 else 0
    # calculate the ratio of bouned sisterC / total initial sisterC
    bound_sisterC_ratio = (df_WT['Rac'][index] + df_WT['RacN'][index] + df_WT['RacP'][index] + df_WT['RacPW'][index] + df_WT['RacPS'][index]) / (paras_dict_coh[N_R]*0.5)
    bound_sisterC_ratio_f = "{:.4f}".format(bound_sisterC_ratio)
    sisterC_value = int(num_sisterCs * bound_sisterC_ratio)
    if index > 0: 
        bound_extC_ratio = (df_WT['R'][index] + df_WT['RN'][index] + df_WT['RP'][index] + df_WT['RPW'][index]) / (paras_dict_coh[N_R]*0.5)
        bound_extC_ratio_f = "{:.4f}".format(bound_extC_ratio)
        extC_bound_frac = (df_WT['R'][index] + df_WT['RN'][index] + df_WT['RP'][index] + df_WT['RPW'][index])/((df_WT['R'][index] + df_WT['RN'][index] + df_WT['RP'][index] + df_WT['RPW'][index]) + df_WT['R_free'][index])
        extC_value = int(num_sisterCs * bound_extC_ratio)
        velocity = 1/5*(df_WT['R'][index] + df_WT['RN'][index] + df_WT['RP'][index] + df_WT['RPW'][index])/df_WT['RN'][index]
    else:
        bound_extC_ratio = 1
        bound_extC_ratio_f = "{:.4f}".format(bound_extC_ratio)
        extC_bound_frac = 1/2
        extC_value = int(num_sisterCs * bound_extC_ratio)
        velocity = 'NAN'
    LEF_sep = lattice_size *extC_bound_frac/(extC_value/2)
    
    if h==4:
        LEF_sep_4h = int(LEF_sep)
        velocity_4h = velocity
    if h==9:
        LEF_sep_9h = int(LEF_sep)
        velocity_9h = velocity
    # print(f"{h}h -- {sisterC_value} -- {bound_sisterC_ratio_f}")
    if index == 0:
        print("h -- sisterC_value -- bound_sisterC_ratio_f -- extC_value -- bound_extC_ratio_f -- extC_bound_frac -- LEF -- velocity -- RacPS")
    print(f"{h}h -- {sisterC_value} -- {bound_sisterC_ratio_f} -- {extC_value} -- {bound_extC_ratio_f} -- {extC_bound_frac} -- {LEF_sep} -- {velocity} -- {df_WT['RacPS'][index]}")


time_9h = 3600*9 -1
print("WT 9h")
rate_R_free_to_RN = paras_dict_ext_coh['Kext_R_free_RN']*df_WT['N'][time_9h]
rate_RPW_R_free = paras_dict_ext_coh['Kext_RPW_R_free']
rate_RN_R = paras_dict_ext_coh['Kext_RN_R']
rate_R_RN = paras_dict_ext_coh['Kext_R_RN']*df_WT['N'][time_9h]
rate_R_RP = paras_dict_ext_coh['Kext_R_RP']*df_WT['P'][time_9h]
rate_RP_R = paras_dict_ext_coh['Kext_RP_R']
rate_RP_RPW = paras_dict_ext_coh['Kext_RP_RPW']*df_WT['W'][time_9h]
rate_RPW_RP = paras_dict_ext_coh['Kext_RPW_RP']

print(f"R_free to RN: {rate_R_free_to_RN }")
print(f"RPW to R_free: {rate_RPW_R_free}")
print(f"RN to R: {rate_RN_R}")
print(f"R to RN: {rate_R_RN}")
print(f"R to RP: {rate_R_RP}")
print(f"RP to R: {rate_RP_R }")
print(f"RP to RPW: {rate_RP_RPW}")
print(f"RPW to RP: {rate_RPW_RP}")

def sister_RAD21_bound_time(K_RacPW_Rac_free, B_W_sister, B_R_sister):
    # K_RPW_R_free * N_W * F_W - F_R_sister*N_R/tau_R
    ### F_R is the fraction of bounded RAD21, T_W is the total number of bounded Wapl 
    return B_R_sister/(K_RacPW_Rac_free * B_W_sister)

time_9h = 3600*9 - 1

sister_RAD21_time_9h = sister_RAD21_bound_time(
        K_RacPW_Rac_free = paras_dict_ext_coh['K_RacPW_Rac_free'], \
        B_W_sister = df_WT['RacPW'][time_9h], \
        B_R_sister = df_WT['RacPS'][time_9h] \
            + df_WT['RacPW'][time_9h] \
            + df_WT['RacP'][time_9h] \
            + df_WT['Rac'][time_9h]  \
            + df_WT['RacN'][time_9h] )

print(sister_RAD21_time_9h)

import json
with open("extrusion_dict_RN_RB_RP_RW_HBD_WT.json", "r") as f:
    params = json.load(f)
params


params["LEF_on_rate"]["A"] = float(rate_R_free_to_RN)
params["LEF_off_rate"]["A"] = float(rate_RPW_R_free)
params["LEF_stalled_off_rate"]["A"] = float(rate_RPW_R_free)
params["LEF_transition_rates"]["21"]["A"] = float(rate_R_RN)
params["LEF_transition_rates"]["23"]["A"] = float(rate_R_RP)
params["LEF_transition_rates"]["12"]["A"] = float(rate_RN_R)
params["LEF_transition_rates"]["32"]["A"] = float(rate_RP_R)
params["LEF_transition_rates"]["34"]["A"] = float(rate_RP_RPW)
params["LEF_transition_rates"]["43"]["A"] = float(rate_RPW_RP)
params["LEF_separation"] = LEF_sep_9h
params["velocity_multiplier"] = float(velocity_9h)

params['monomers_per_replica'] = 32000
params['num_of_sisters'] = 776

params['sister_damping'] = 50
params['sister_lifetime'] = residence_time*3600


with open(f"extrusion_dict_RN_RB_RP_RW_HBD_WT_alpha{params['sister_damping']}_tau{residence_time}h.json", "w") as f:
    json.dump(params, f, indent=4)


