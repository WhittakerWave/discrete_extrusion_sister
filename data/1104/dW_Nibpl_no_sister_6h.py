


import json
import numpy as np
import sympy as sym 
import tellurium as te
import pandas as pd
from pathlib import Path


# Load configuration files
def load_config(filename):
    """Load configuration from JSON file"""
    with open(filename, 'r') as f:
        return json.load(f)

# Define the residence time [in hours]
RESIDENCE_TIMES = [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 25, 30, 5000]          
# Define the damping values
SISTER_DAMPINGS = [5000000]
BYPASS_PROBS = [0, 0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]  

# Physical constants
NUM_SISTERCS = 7765 
NUM_EXTRUDERS = 7765 
LATTICE_SIZE = 320000

## Rates for cohesive network 
rates_coh = sym.symbols("K_RacP_RacPW, K_RacPW_RacP, K_RacP_RacPS, K_RacPS_RacP, K_RacP_Rac, K_Rac_RacP,  K_RacPW_Rac_free")
K_RacP_RacPW, K_RacPW_RacP, K_RacP_RacPS, K_RacPS_RacP, K_RacP_Rac, K_Rac_RacP,  K_RacPW_Rac_free = rates_coh

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
MODEL_EXT_COH_TEMPLATE ='''
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
    
    # Rates for cohesive network,  7 rates
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

def calculate_extrusive_RAD21_bound_time(K_RPW_R_free, B_W_extruder, B_R_extruder):
    # K_RPW_R_free * N_W * F_W - F_R_sister*N_R/tau_R
    ### F_R is the fraction of bounded RAD21, T_W is the total number of bounded Wapl 
    return B_R_extruder/(K_RPW_R_free * B_W_extruder)

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

def run_simulation(config, residence_time, bypass_prob):
    """
    Run the simulation for given residence_time and sister_damping
    Returns the params dictionary ready to save
    """
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
    
    init_conditions_ext_coh = {
       Rac_free_init: 0, 
       Rac_init: paras_dict_coh_local[N_R]*1/2, 
       RacP_init: 0,
       RacPW_init: 0,
       RacPS_init: 0,

       R_free_init: paras_dict_coh_local[N_R],
       RN_init: 0,
       R_init: 0,
       RP_init: 0, 
       RPW_init: 0, 
    
       N_init: config['base_parameters']['N_N'],
       S_init: paras_dict_coh_local[N_S],
       W_init: config['base_parameters']['N_W'],
       P_init: config['base_parameters']['N_P'],
    }

    # Update combined dictionary
    paras_dict_local = paras_dict_coh_local | paras_dict_ext_local | init_conditions_ext_coh | {"K_Rac_free_R_free": 1/2495}
    paras_dict_ext_coh_local = {str(key): value for key, value in paras_dict_local.items()}
    
    # Rebuild and simulate model
    Model_ext_coh = build_model(MODEL_EXT_COH_TEMPLATE, paras_dict_ext_coh_local)
    r_ext_coh = te.loada(Model_ext_coh)
    Model_ext_coh_WT = r_ext_coh.simulate(0, 3600*18, 3600*18)

    columns = ['time', 'Rac', 'P', 'RacP', 'S', 'RacPS', 'W', 'RacPW', 
               'Rac_free', 'R_free', 'N', 'RN', 'R', 'RP', 'RPW']
    df_WT = pd.DataFrame(Model_ext_coh_WT, columns=columns)
    
    ## Apply Wapl depletion at 8h in G2 (14h since start of S phase)
    time_8h = 3600 * 8 - 1
    depletion_level = config['simulation_parameters']['depletion_level']
    remaining_level = 1 - depletion_level 
    
    init_conditions_ext_coh_dWapl_8h = {
        Rac_free_init: df_WT['Rac_free'][time_8h], 
        Rac_init: df_WT['Rac'][time_8h], 
        RacP_init: df_WT['RacP'][time_8h] + df_WT['RacPW'][time_8h] * depletion_level,
        RacPW_init: df_WT['RacPW'][time_8h] * remaining_level,
        RacPS_init: df_WT['RacPS'][time_8h],

        R_free_init: df_WT['R_free'][time_8h],
        RN_init: df_WT['RN'][time_8h] ,
        R_init: df_WT['R'][time_8h]  ,
        RP_init: df_WT['RP'][time_8h] + df_WT['RPW'][time_8h] * depletion_level, 
        RPW_init: df_WT['RPW'][time_8h] * remaining_level, 
    
        N_init: df_WT['N'][time_8h],
        S_init: df_WT['S'][time_8h],
        W_init: df_WT['W'][time_8h] * remaining_level,
        P_init: df_WT['P'][time_8h],
       }

    paras_dict_since_8h_dW = paras_dict_coh_local | paras_dict_ext_local | init_conditions_ext_coh_dWapl_8h | {"K_Rac_free_R_free": 1/2495}
    paras_dict_ext_coh_since_8h_dW = {str(key): value for key, value in paras_dict_since_8h_dW.items()}
    
    Model_ext_coh_since_8h_dW = build_model(MODEL_EXT_COH_TEMPLATE, paras_dict_ext_coh_since_8h_dW)
    # print(model_ext_coh)
    # Load the modes
    r_ext_coh_since_8h_dW = te.loada(Model_ext_coh_since_8h_dW)
    # Simulate the model
    Model_ext_coh_since_8h_dW = r_ext_coh_since_8h_dW.simulate(0, 3600*10, 3600*10)
    
    df_since_8h = pd.DataFrame(Model_ext_coh_since_8h_dW, columns=columns)
    
    analysis_hours = config['simulation_parameters']['analysis_timepoint_hours'] 
    index = 3600 * analysis_hours  - 1

    # Calculate metrics
    total_bound_ext = (df_since_8h['R'][index] + df_since_8h['RN'][index] + 
                       df_since_8h['RP'][index] + df_since_8h['RPW'][index])
    
    bound_extC_ratio = total_bound_ext / (paras_dict_coh_local[N_R]*0.5)
    extC_bound_frac = total_bound_ext / (total_bound_ext + df_since_8h['R_free'][index])
    extC_value = int(NUM_SISTERCS * bound_extC_ratio)
    # velocity_7h = 1/5 * total_bound_ext / df_since_8h['RN'][index]
    LEF_sep_13h = int(LATTICE_SIZE * extC_bound_frac / (extC_value / 2))

    total_sister_rad21 = (df_since_8h['RacPS'][index] + df_since_8h['RacPW'][index] + 
                          df_since_8h['RacP'][index] + df_since_8h['Rac'][index])

    sister_RAD21_time_10h = calculate_sister_RAD21_bound_time(
        K_RacPW_Rac_free = paras_dict_ext_coh_since_8h_dW['K_RacPW_Rac_free'], \
        B_W_sister = df_since_8h['RacPW'][index], \
        B_R_sister = total_sister_rad21)
    
    total_extrusive_rad21 = (df_since_8h['RPW'][index] + 
                          df_since_8h['RP'][index] + df_since_8h['R'][index] + 
                          df_since_8h['RN'][index])
    
    extruder_RAD21_time_10h = calculate_extrusive_RAD21_bound_time(
        K_RPW_R_free = paras_dict_ext_coh_since_8h_dW['Kext_RPW_R_free'], \
        B_W_extruder = df_since_8h['RPW'][index], \
        B_R_extruder = total_extrusive_rad21)
    
    # Calculate transition rates
    rates = {
        'R_free_to_RN': paras_dict_ext_coh_since_8h_dW['Kext_R_free_RN']*df_since_8h['N'][index],
        'RPW_R_free': paras_dict_ext_coh_since_8h_dW['Kext_RPW_R_free'],
        'RN_R': paras_dict_ext_coh_since_8h_dW['Kext_RN_R'],
        'R_RN': paras_dict_ext_coh_since_8h_dW['Kext_R_RN']*df_since_8h['N'][index],
        'R_RP': paras_dict_ext_coh_since_8h_dW['Kext_R_RP']*df_since_8h['P'][index],
        'RP_R': paras_dict_ext_coh_since_8h_dW['Kext_RP_R'],
        'RP_RPW': paras_dict_ext_coh_since_8h_dW['Kext_RP_RPW']*df_since_8h['W'][index],
        'RPW_RP': paras_dict_ext_coh_since_8h_dW['Kext_RPW_RP'],
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
    
    output_params["LEF_separation"] = LEF_sep_13h
    # output_params["velocity_multiplier"] = float(velocity_8h)
    output_params["velocity_multiplier"] = 0.8
    output_params["monomers_per_replica"] = 32000
    output_params["num_of_sisters"] = config['simulation_parameters']['num_of_sisters']
    output_params["sister_damping"] = SISTER_DAMPINGS[0]
    output_params["bypass_probs"] = bypass_prob
    output_params["sister_lifetime"] = int(sister_RAD21_time_10h)
    
    return output_params 


def main():
    """Main execution function"""
    # Load configuration
    config = load_config('network_parameters_dW.json')
    
    # Create output directory
    output_dir = Path(config['output_directory'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run parameter sweep
    for residence_time in RESIDENCE_TIMES:
        for i, bypass_probs in enumerate(BYPASS_PROBS):
            print(f"\nRunning: residence_time={residence_time}h, bypass_probs ={bypass_probs}")
            
            # Run simulation
            params = run_simulation(config, residence_time, bypass_probs)

            # Save to file
            filename = (f"{config['output_prefix']}_"
                       f"alpha{i}_tau{residence_time}h.json")
            filepath = output_dir / filename
            
            with open(filepath, "w") as f:
                json.dump(params, f, indent=4)
            
            print(f"Saved: {filepath}") 

    print("\nAll simulations complete!")


if __name__ == "__main__":
    main()





