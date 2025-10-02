


import json
import numpy as np
import sympy as sym 
import tellurium as te
import pandas as pd

# Define the ranges you want to loop over
residence_times = [10]  # in hours
sister_dampings = [50]  # damping values

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

def sister_RAD21_bound_time(K_RacPW_Rac_free, B_W_sister, B_R_sister):
    # K_RPW_R_free * N_W * F_W - F_R_sister*N_R/tau_R
    ### F_R is the fraction of bounded RAD21, T_W is the total number of bounded Wapl 
    return B_R_sister/(K_RacPW_Rac_free * B_W_sister)


def run_simulation(residence_time, sister_damping):
    """
    Run the simulation for given residence_time and sister_damping
    Returns the params dictionary ready to save
    """
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
        3600 * residence_time,           # tau_R_sister, 6h 
        1/2,                   # modified F_R_sister
        284470*2/3,            # modified N_R
    ]
    paras_values_coh_copy = paras_values_coh.copy()
    # Rebuild the cohesive parameter dictionary
    paras_dict_coh_local = dict(zip(paras_coh, paras_values_coh_copy))
    for s in sol_rates_coh.items():
        rate = s[1].evalf(subs=paras_dict_coh_local)
        paras_dict_coh_local[s[0]] = rate
    
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
    
    # Rebuild the extrusive parameter dictionary (need to recalculate with new values)
    paras_dict_ext_local = dict(zip(paras_ext, paras_values_ext))
    for s in sol_rates_ext.items():
        rate = s[1].evalf(subs=paras_dict_ext_local)
        paras_dict_ext_local[s[0]] = rate

    # Define symbolic initial conditions
    Rac_free_init, Rac_init, RacN_init, RacP_init, RacPW_init, RacPS_init, R_free_init, RN_init, R_init, RP_init, RPW_init, N_init, S_init, W_init, P_init = sym.symbols(
        "Rac_free_init, Rac_init, RacN_init, RacP_init, RacPW_init, RacPS_init, R_free_init, RN_init, R_init, RP_init, RPW_init, N_init, S_init, W_init, P_init")
    
    init_conditions_ext_coh = {
       Rac_free_init: 0, 
       Rac_init: paras_dict_coh_local[N_R]*1/2, 
       RacN_init: 0, 
       RacP_init: 0,
       RacPW_init: 0,
       RacPS_init: 0,

       R_free_init: paras_dict_coh_local[N_R],
       RN_init: 0,
       R_init: 0,
       RP_init: 0, 
       RPW_init: 0, 
    
       N_init: 119308,
       S_init: paras_dict_coh_local[N_S],
       W_init: 69542,
       P_init: 180615,
    }
    # Update combined dictionary
    paras_dict_local = paras_dict_coh_local | paras_dict_ext_local | init_conditions_ext_coh | {"K_Rac_free_R_free": 1/2495}
    paras_dict_ext_coh_local = {str(key): value for key, value in paras_dict_local.items()}
    
    # Rebuild and simulate model
    Model_ext_coh = build_model_ext_coh(model_ext_coh, paras_dict_ext_coh_local)
    r_ext_coh = te.loada(Model_ext_coh)
    Model_ext_coh_WT = r_ext_coh.simulate(0, 3600*18, 3600*18)

    columns = ['time', 'Rac', 'N', 'RacN', 'P', 'RacP', 'S', 'RacPS', 'W', 'RacPW', 
           'Rac_free', 'R_free', 'RN', 'R', 'RP', 'RPW']
    df_WT = pd.DataFrame(Model_ext_coh_WT, columns=columns)
    time_2h = 3600 * 2 - 1
    depletion_level = 0.95
    remaining_level = 1 - depletion_level 
    init_conditions_ext_coh_dNipbl_2h = {
        Rac_free_init: df_WT['Rac_free'][time_2h], 
        Rac_init: df_WT['Rac'][time_2h] + df_WT['RacN'][time_2h] * remaining_level, 
        RacN_init: df_WT['RacN'][time_2h] * depletion_level , 
        RacP_init: df_WT['RacP'][time_2h],
        RacPW_init: df_WT['RacPW'][time_2h],
        RacPS_init: df_WT['RacPS'][time_2h],

        R_free_init: df_WT['R_free'][time_2h],
        RN_init: df_WT['RN'][time_2h] * depletion_level ,
        R_init: df_WT['R'][time_2h] + df_WT['RN'][time_2h] * remaining_level ,
        RP_init: df_WT['RP'][time_2h], 
        RPW_init: df_WT['RPW'][time_2h], 
    
        N_init: df_WT['N'][time_2h] * depletion_level ,
        S_init: df_WT['S'][time_2h],
        W_init: df_WT['W'][time_2h],
        P_init: df_WT['P'][time_2h],
    }

    paras_dict_since_2h_dN = paras_dict_coh_local | paras_dict_ext_local | init_conditions_ext_coh_dNipbl_2h | {"K_Rac_free_R_free": 1/2495}
    paras_dict_ext_coh_since_2h_dN = {str(key): value for key, value in paras_dict_since_2h_dN.items()}
    Model_ext_coh_since_2h_dN = build_model_ext_coh(model_ext_coh, paras_dict_ext_coh_since_2h_dN)
    # print(model_ext_coh)
    # Load the modes
    r_ext_coh_since_2h_dN = te.loada(Model_ext_coh_since_2h_dN)
    # Simulate the model
    Model_ext_coh_since_2h_dN = r_ext_coh_since_2h_dN.simulate(0, 3600*16, 3600*16)

    columns = ['time', 'Rac', 'N', 'RacN', 'P', 'RacP', 'S', 'RacPS', 'W', 'RacPW', 
           'Rac_free', 'R_free', 'RN', 'R', 'RP', 'RPW']
    
    df_since_2h = pd.DataFrame(Model_ext_coh_since_2h_dN, columns=columns)
    
    h = 8
    index = 3600 * h - 1
    bound_extC_ratio = (df_since_2h['R'][index] + df_since_2h['RN'][index] + 
                        df_since_2h['RP'][index] + df_since_2h['RPW'][index]) / (paras_dict_coh_local[N_R]*0.5)
    extC_bound_frac = ((df_since_2h['R'][index] + df_since_2h['RN'][index] + 
                        df_since_2h['RP'][index] + df_since_2h['RPW'][index]) /
                       ((df_since_2h['R'][index] + df_since_2h['RN'][index] + 
                         df_since_2h['RP'][index] + df_since_2h['RPW'][index]) + df_since_2h['R_free'][index]))
    extC_value = int(num_sisterCs * bound_extC_ratio)
    velocity_8h = 1/5 * (df_since_2h['R'][index] + df_since_2h['RN'][index] + 
                         df_since_2h['RP'][index] + df_since_2h['RPW'][index]) / df_since_2h['RN'][index]
    LEF_sep_8h = int(lattice_size * extC_bound_frac / (extC_value / 2))

    time_10h = 3600*10 - 1

    sister_RAD21_time_10h = sister_RAD21_bound_time(
        K_RacPW_Rac_free = paras_dict_ext_coh_since_2h_dN['K_RacPW_Rac_free'], \
        B_W_sister = df_since_2h['RacPW'][time_10h], \
        B_R_sister = df_since_2h['RacPS'][time_10h] \
            + df_since_2h['RacPW'][time_10h] \
            + df_since_2h['RacP'][time_10h] \
            + df_since_2h['Rac'][time_10h]  \
            + df_since_2h['RacN'][time_10h])

    
    time_8h = 3600 * 8 - 1
    rate_R_free_to_RN = paras_dict_ext_coh_since_2h_dN['Kext_R_free_RN']*df_since_2h['N'][time_8h]
    rate_RPW_R_free = paras_dict_ext_coh_since_2h_dN['Kext_RPW_R_free']
    rate_RN_R = paras_dict_ext_coh_since_2h_dN['Kext_RN_R']
    rate_R_RN = paras_dict_ext_coh_since_2h_dN['Kext_R_RN']*df_since_2h['N'][time_8h]
    rate_R_RP = paras_dict_ext_coh_since_2h_dN['Kext_R_RP']*df_since_2h['P'][time_8h]
    rate_RP_R = paras_dict_ext_coh_since_2h_dN['Kext_RP_R']
    rate_RP_RPW = paras_dict_ext_coh_since_2h_dN['Kext_RP_RPW']*df_since_2h['W'][time_8h]
    rate_RPW_RP = paras_dict_ext_coh_since_2h_dN['Kext_RPW_RP']
    
    # Load base parameters and update
    with open(f"extrusion_dict_RN_RB_RP_RW_HBD_dN.json", "r") as f:
        params = json.load(f)
    
    params["LEF_on_rate"]["A"] = float(rate_R_free_to_RN)
    params["LEF_off_rate"]["A"] = float(rate_RPW_R_free)
    params["LEF_stalled_off_rate"]["A"] = float(rate_RPW_R_free)
    params["LEF_transition_rates"]["21"]["A"] = float(rate_R_RN)
    params["LEF_transition_rates"]["23"]["A"] = float(rate_R_RP)
    params["LEF_transition_rates"]["12"]["A"] = float(rate_RN_R)
    params["LEF_transition_rates"]["32"]["A"] = float(rate_RP_R)
    params["LEF_transition_rates"]["34"]["A"] = float(rate_RP_RPW)
    params["LEF_transition_rates"]["43"]["A"] = float(rate_RPW_RP)

    params["LEF_separation"] = LEF_sep_8h
    params["velocity_multiplier"] = float(velocity_8h)
    params["monomers_per_replica"] = 32000
    params["num_of_sisters"] = 776
    params["sister_damping"] = sister_damping
    params["sister_lifetime"] = int(sister_RAD21_time_10h)
    
    return params

# Main loop
num_sisterCs = 7765 
lattice_size = 320000

for residence_time in residence_times:
    for sister_damping in sister_dampings:
        print(f"\nRunning: residence_time={residence_time}h, sister_damping={sister_damping}")
        
        # Run simulation
        params = run_simulation(residence_time, sister_damping)
        # Save to file
        filename = f"extrusion_dict_RN_RB_RP_RW_HBD_dN10h_alpha{sister_damping}_tau{residence_time}h.json"
        with open(filename, "w") as f:
            json.dump(params, f, indent=4)
        
        print(f"Saved: {filename}")

print("\nAll simulations complete!")






    



