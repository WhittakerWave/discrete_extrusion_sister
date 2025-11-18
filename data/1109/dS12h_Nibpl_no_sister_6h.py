

### 
### This script generates the parameters for extrusion simulations with dS 9h

import json
import numpy as np
import sympy as sym 
import tellurium as te
import pandas as pd
from pathlib import Path
from itertools import product

# Load configuration files
def load_config(filename):
    """Load configuration from JSON file"""
    with open(filename, 'r') as f:
        return json.load(f)
    
# Define the ranges 
# RESIDENCE_TIMES = [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]          
# Define the damping values
# SISTER_DAMPINGS = [40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60]  
# SISTER_DAMPINGS = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 25, 5000]
# 
RESIDENCE_TIMES = [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 25, 5000]          
# Define the damping values, 17 
SISTER_DAMPINGS = [10, 20, 25, 30, 35, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 5000]  

pairs = [(a, b) for a, b in product(RESIDENCE_TIMES, SISTER_DAMPINGS)]  

# Physical constants
NUM_EXTRUDERS = 7765 
LATTICE_SIZE = 320000 

## Rates for cohesive network 
rates_coh = sym.symbols("K_RacP_RacPW, K_RacPW_RacP, K_RacP_RacPS, K_RacPS_RacP, K_RacP_Rac, K_Rac_RacP,  K_RacPW_Rac_free")
K_RacP_RacPW, K_RacPW_RacP, K_RacP_RacPS, K_RacPS_RacP, K_RacP_Rac, K_Rac_RacP, K_RacPW_Rac_free = rates_coh

paras_coh = sym.symbols("tau_S, F_S, N_S, tau_W, F_W, N_W, tau_P, F_P, N_P, tau_R, F_R_sister, N_R")
tau_S, F_S, N_S, tau_W, F_W, N_W, tau_P, F_P, N_P, tau_R, F_R_sister, N_R = paras_coh 

## Equations for cohesive network
rhs_coh = [K_RacPW_Rac_free * N_W * F_W - F_R_sister*N_R/tau_R,                              # RAD21 unbinding  
           F_W/(tau_W * (1-F_W)) - K_RacP_RacPW * (F_P * N_P - F_S * N_S - F_W * N_W),       # Wapl binding 
           K_RacPW_RacP + K_RacPW_Rac_free - 1/tau_W,                                        # Wapl unbinding  
           F_S/(tau_S * (1-F_S)) - K_RacP_RacPS * (F_P * N_P - F_S * N_S - F_W * N_W),       # Sororin binding
           K_RacPS_RacP - 1/tau_S,                                                           # Sororin unbinding
           F_P/(tau_P * (1-F_P)) - K_Rac_RacP * (F_R_sister * N_R - F_P * N_P),                  # PDS5 binding
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

def build_model(model, parameter_dict):
    string_params = {str(k): v for k, v in parameter_dict.items()}
    return model.format(**string_params)

def calculate_sister_RAD21_bound_time(K_RacPW_Rac_free, B_W_sister, B_R_sister):
    # K_RPW_R_free * N_W * F_W - F_R_sister*N_R/tau_R
    ### F_R is the fraction of bounded RAD21, T_W is the total number of bounded Wapl 
    return B_R_sister/(K_RacPW_Rac_free * B_W_sister)

def calculate_cohesive_parameters(config, residence_time):
    """
    Calculate cohesive network parameters from configuration
    
    Args:
        config: Dictionary containing base parameters and modifiers
        residence_time: Residence time in hours
    
    Returns:
        List of cohesive parameter values
    """
    base = config['base_parameters']
    mod = config['sister_networks']
    
    return [
        base['tau_S'],
        base['F_S'],
        base['N_S'],
        base['tau_W'],
        mod['F_W_sister'] / (1 - base['F_W'] + mod['F_W_sister']),
        base['N_W'] * (1 - base['F_W'] + mod['F_W_sister']),
        base['tau_P'],
        mod['F_P_sister'] / (1 - base['F_P'] + mod['F_P_sister']),
        base['N_P'] * (1 - base['F_P'] + mod['F_P_sister']),
        3600 * residence_time,  # tau_R_sister
        base['F_R_sister'],
        base['N_R'] * base['N_R_fraction'],
    ]

def calculate_extrusive_parameters(config):
    """
    Calculate extrusive network parameters from configuration
    
    Args:
        config: Dictionary containing base parameters and modifiers
    
    Returns:
        List of extrusive parameter values
    """
    base = config['base_parameters']
    mod = config['sister_networks']
    
    return [
        base['tau_N'],
        base['F_N'],
        base['N_N'],
        base['tau_W'],
        (base['F_W'] - mod['F_W_sister'])/(1 - mod['F_W_sister']),
        base['N_W']*(1 - mod['F_W_sister']),
        base['tau_P'],
        (base['F_P'] - mod['F_P_sister'])/(1 - mod['F_P_sister']),
        base['N_P']*(1 - mod['F_P_sister']),
        base["tau_R_extrusive"], 
        base['F_R_sister'],
        base['N_R'] * base['N_R_fraction'],
    ]

# run simulations for given parameters  

def run_simulation(config, residence_time, sister_damping):
    """
    Run the simulation for given residence_time and sister_damping
    Returns the params dictionary ready to save
    """
    NUM_SISTERCS = int(6988 * np.exp(6/residence_time))
    # Calculate parameter values
    paras_values_coh = calculate_cohesive_parameters(config, residence_time)
    paras_dict_coh_local = dict(zip(paras_coh, paras_values_coh))

    # Solve for cohesive rates
    for rate_symbol, rate_expr in sol_rates_coh.items():
        rate = rate_expr.evalf(subs=paras_dict_coh_local)
        paras_dict_coh_local[rate_symbol] = rate
    
    # Calculate extrusive parameters
    paras_values_ext = calculate_extrusive_parameters(config)
    paras_dict_ext_local = dict(zip(paras_ext, paras_values_ext))

    # Solve for extrusive rates
    for rate_symbol, rate_expr in sol_rates_ext.items():
        rate = rate_expr.evalf(subs=paras_dict_ext_local)
        paras_dict_ext_local[rate_symbol] = rate

    # Define symbolic initial conditions
    Rac_free_init, Rac_init, RacP_init, RacPW_init, RacPS_init, R_free_init, RN_init, R_init, RP_init, RPW_init, N_init, S_init, W_init, P_init = sym.symbols(
        "Rac_free_init, Rac_init, RacP_init, RacPW_init, RacPS_init, R_free_init, RN_init, R_init, RP_init, RPW_init, N_init, S_init, W_init, P_init")
    
    depletion_level_sororin = config['simulation_parameters']['depletion_level'] 
    remaining_level_sororin = 1 - depletion_level_sororin 

    # whether to use the new number of sisterCs: 
    init_conditions_ext_coh = {
       Rac_free_init: 0, 
       Rac_init: NUM_SISTERCS,  
       RacP_init: 0,
       RacPW_init: 0,
       RacPS_init: 0,

       R_free_init: paras_dict_coh_local[N_R],
       RN_init: 0,
       R_init: 0,
       RP_init: 0, 
       RPW_init: 0, 
    
       N_init: config['base_parameters']['N_N'],
       S_init: paras_dict_coh_local[N_S] * remaining_level_sororin,
       W_init: config['base_parameters']['N_W'],
       P_init: config['base_parameters']['N_P'],
    }
    # Update combined dictionary
    paras_dict_local = paras_dict_coh_local | paras_dict_ext_local | init_conditions_ext_coh | {"K_Rac_free_R_free": 1/2495}
    paras_dict_ext_coh_local = {str(key): value for key, value in paras_dict_local.items()}
    
    # Rebuild and simulate model
    Model_ext_coh = build_model(model_ext_coh, paras_dict_ext_coh_local)
    r_ext_coh = te.loada(Model_ext_coh)
    Model_ext_coh_dS = r_ext_coh.simulate(0, 3600*24, 3600*24)
    
    columns = ['time', 'Rac', 'P', 'RacP', 'S', 'RacPS', 'W', 'RacPW', 
           'Rac_free', 'R_free', 'N', 'RN', 'R', 'RP', 'RPW']
    
    df_dS = pd.DataFrame(Model_ext_coh_dS, columns=columns)
    
    index_start_G2 = 3600*6 - 1 
    NUM_SISTERCS_start_G2 = int((df_dS['Rac'][index_start_G2]  + df_dS['RacPS'][index_start_G2] + 
                       df_dS['RacP'][index_start_G2] + df_dS['RacPW'][index_start_G2])//10)
    # Calculate LEF_sep and velocity at 9h
    analysis_hours = config['simulation_parameters']['analysis_timepoint_hours'] 
    index = 3600 * analysis_hours - 1
    total_bound_ext = (df_dS['R'][index] + df_dS['RN'][index] + 
                       df_dS['RP'][index] + df_dS['RPW'][index])
    bound_extC_ratio = total_bound_ext / (paras_dict_coh_local[N_R]*0.5)
    extC_bound_frac =  total_bound_ext / (total_bound_ext + df_dS['R_free'][index])
    extC_value = int(NUM_EXTRUDERS * bound_extC_ratio)
    # velocity_9h = 1/5 * total_bound_ext / df_dS['RN'][index]
    LEF_sep_9h = int(LATTICE_SIZE * extC_bound_frac / (extC_value / 2))
    

    total_sister_rad21 = (df_dS['RacPS'][index] + df_dS['RacPW'][index] + 
                          df_dS['RacP'][index] + df_dS['Rac'][index])

    sister_RAD21_time_9h = calculate_sister_RAD21_bound_time(
        K_RacPW_Rac_free = paras_dict_ext_coh_local['K_RacPW_Rac_free'], \
        B_W_sister = df_dS['RacPW'][index], \
        B_R_sister = total_sister_rad21)
    
    # Calculate transition rates
    rates = {
        'R_free_to_RN': paras_dict_ext_coh_local['Kext_R_free_RN'] * df_dS['N'][index],
        'RPW_R_free': paras_dict_ext_coh_local['Kext_RPW_R_free'],
        'RN_R': paras_dict_ext_coh_local['Kext_RN_R'],
        'R_RN': paras_dict_ext_coh_local['Kext_R_RN'] * df_dS['N'][index],
        'R_RP': paras_dict_ext_coh_local['Kext_R_RP'] * df_dS['P'][index],
        'RP_R': paras_dict_ext_coh_local['Kext_RP_R'],
        'RP_RPW': paras_dict_ext_coh_local['Kext_RP_RPW'] * df_dS['W'][index],
        'RPW_RP': paras_dict_ext_coh_local['Kext_RPW_RP'],
    }

    # Load base parameters and update
    with open(f"extrusion_dict_RN_RB_RP_RW_HBD_WT.json", "r") as f:
        output_params = json.load(f)
    
    output_params["LEF_on_rate"]["A"] = float(rates['R_free_to_RN'])
    output_params["LEF_off_rate"]["A"] = float(rates['RPW_R_free'])
    output_params["LEF_stalled_off_rate"]["A"] = float(rates['RPW_R_free'])
    output_params["LEF_transition_rates"]["21"]["A"] = float(rates['R_RN'])
    output_params["LEF_transition_rates"]["23"]["A"] = float(rates['R_RP'])
    output_params["LEF_transition_rates"]["12"]["A"] = float(rates['RN_R'])
    output_params["LEF_transition_rates"]["32"]["A"] = float(rates['RP_R'])
    output_params["LEF_transition_rates"]["34"]["A"] = float(rates['RP_RPW'])
    output_params["LEF_transition_rates"]["43"]["A"] = float(rates['RPW_RP'])

    output_params["LEF_separation"] = LEF_sep_9h
    # output_params["velocity_multiplier"] = float(velocity_9h)
    output_params["velocity_multiplier"] = 0.8
    output_params["monomers_per_replica"] = 32000
    # output_params["num_of_sisters"] = config['simulation_parameters']['num_of_sisters']
    output_params["num_of_sisters"] = int(NUM_SISTERCS_start_G2)
    output_params["sister_damping"] = sister_damping
    output_params["sister_lifetime"] = int(sister_RAD21_time_9h)
    
    return output_params


def main():
    """Main execution function"""
    # Load configuration
    config = load_config('network_parameters_dS12h.json')
    
    # Create output directory
    output_dir = Path(config['output_directory'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run parameter sweep
    # for residence_time in RESIDENCE_TIMES:
    #    for sister_damping in SISTER_DAMPINGS:
    for residence_time, sister_damping in pairs:
            print(f"\nRunning: residence_time={residence_time}h, sister_damping={sister_damping}")
            
            # Run simulation
            params = run_simulation(config, residence_time, sister_damping)
            
            # Save to file
            filename = (f"{config['output_prefix']}_"
                       f"alpha{sister_damping}_tau{residence_time}h.json")
            filepath = output_dir / filename
            
            with open(filepath, "w") as f:
                json.dump(params, f, indent=4)
            
            print(f"Saved: {filepath}")

    print("\nAll simulations complete!")


if __name__ == "__main__":
    main()


