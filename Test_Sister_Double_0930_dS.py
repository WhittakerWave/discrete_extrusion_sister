

import pickle
import sys
import os
import json
# import hoomd
import codecs
# import cooltools
import time
import numpy as np
import matplotlib.pyplot as plt

# import discrete_time_extrusion

import discrete_time_extrusion.arrays as arrays 
# from discrete_time_extrusion.Translocator import Translocator
from discrete_time_extrusion.Translocator_Sister import Translocator_Sister
from discrete_time_extrusion.boundaries.StaticBoundary import StaticBoundary
# from discrete_time_extrusion.boundaries.DynamicBoundary import DynamicBoundary
# from discrete_time_extrusion.extruders.BaseExtruder_Sister import BaseExtruder_Sister
# from discrete_time_extrusion.extruders.BaseExtruder_Sister_Slow import BaseExtruder_Sister_Slow
# from discrete_time_extrusion.extruders.MultistateExtruder import MultistateExtruder
from discrete_time_extrusion.extruders.MultistateExtruder_Sister import MultistateExtruder_Sister

with open("data/new_dict/alpha100_tau20h/extrusion_dict_RN_RB_RP_RW_HBD_dS_alpha100_tau20h.json", 'r') as dict_file:
        paramdict_dS = json.load(dict_file)
    
monomers_per_replica = paramdict_dS['monomers_per_replica'] 
sites_per_monomer = paramdict_dS['sites_per_monomer']

# sites_per_replica = s_per_monomer
# Work with a single type of monomers (A, assigned to type index 0)
type_list = ['A']
monomer_types = type_list.index('A') * np.ones(monomers_per_replica, dtype=int)
site_types = np.repeat(monomer_types, sites_per_monomer)
    
# LEF/CTCF properties in type A monomers may be obtained from the paramdict as follows
LEF_off_rate = paramdict_dS['LEF_off_rate']
CTCF_facestall = paramdict_dS['CTCF_facestall']
print(LEF_off_rate['A'], CTCF_facestall['A'])

# anchor_positions = np.genfromtxt(f'{path}/anchor_{N_sister_RAD21}_{iteration}.txt')
anchor_positions = []
ctcf_left_positions = []
ctcf_right_positions = []

start = time.time()
translocator1 = Translocator_Sister(MultistateExtruder_Sister,
                            StaticBoundary,
                            type_list, 
                            site_types,
                            ctcf_left_positions,
                            ctcf_right_positions, 
                            **paramdict_dS)

translocator2 = Translocator_Sister(MultistateExtruder_Sister,
                            StaticBoundary,
                            type_list, 
                            site_types,
                            ctcf_left_positions,
                            ctcf_right_positions, 
                            **paramdict_dS)

def shared_sister_update(translocator1, translocator2, step_number, shared_decay_decisions):
        """
        Update sister states for both translocators using shared decisions
        """
        if step_number >= len(shared_decay_decisions):
            return
        # Get the decay mask for this step
        fall_off_mask = shared_decay_decisions[step_number]
        # Apply to translocator1
        if translocator1.extrusion_engine.sister_positions is not None:
            active_mask_1 = translocator1.extrusion_engine.sister_positions != -1
            final_mask_1 = fall_off_mask & active_mask_1
            translocator1.extrusion_engine.sister_positions[final_mask_1] = -1
        # Apply to translocator2 
        if translocator2.extrusion_engine.sister_positions is not None:
            active_mask_2 = translocator2.extrusion_engine.sister_positions != -1
            final_mask_2 = fall_off_mask & active_mask_2
            translocator2.extrusion_engine.sister_positions[final_mask_2] = -1

def run_synchronized_trajectories_continue(translocator1, translocator2, sister_lifetime, 
                        period=None, steps=None, 
                        prune_unbound_LEFs=True, track_sisters=False, 
                        sample_interval=1, seed=42, clear_trajectories=False, **kwargs):
    """
    Modified version that optionally preserves existing trajectory data
    """
    # Pre-generate shared decay decisions
    num_sisters = translocator1.extrusion_engine.num_sisters
    decay_prob = 1 - np.exp(-1 / sister_lifetime)
        
    # Calculate total steps needed
    steps = int(steps) if steps else translocator1.params['steps']
    period = int(period) if period else translocator1.params['sites_per_monomer']
    dummy_steps = translocator1.params['dummy_steps'] * period
    total_steps = steps * period + dummy_steps
    
    np.random.seed(seed)
    shared_decay_decisions = np.random.random((total_steps, num_sisters)) < decay_prob
    
    print(f"Generated shared decay schedule for {total_steps} total steps")
    
    # Initialize trajectory lists only if clearing or they don't exist
    if clear_trajectories or not hasattr(translocator1, 'state_trajectory'):
        translocator1.state_trajectory = []
        translocator1.lef_trajectory = []
        translocator1.coupling_trajectory = []
        translocator1.ctcf_trajectory = []
        translocator1.sister_trajectory = []
    
    if clear_trajectories or not hasattr(translocator2, 'state_trajectory'):
        translocator2.state_trajectory = []
        translocator2.lef_trajectory = []
        translocator2.coupling_trajectory = []
        translocator2.ctcf_trajectory = []
        translocator2.sister_trajectory = []
    
    step_counter = 0
    # Dummy steps
    translocator1.run(dummy_steps*period, **kwargs)
    translocator2.run(dummy_steps*period, **kwargs)
    print(f"Running {dummy_steps} dummy steps...")

    for step_idx in range(steps):
        # Run 'period' number of steps
        translocator1.run(1)
        translocator2.run(1)
        if step_idx % 1000 == 0:
            print(f"Step {step_idx} completed")
        # Update sisters after each individual step
        shared_sister_update(translocator1, translocator2, step_counter, shared_decay_decisions)
        step_counter += 1
        # Data collection at sampling intervals
        if step_idx % sample_interval == 0:
            # Collect data for translocator1
            LEF_states_1 = translocator1.extrusion_engine.get_states()
            CTCF_positions_1 = translocator1.barrier_engine.get_bound_positions()
            
            if prune_unbound_LEFs:
                LEF_positions_1 = translocator1.extrusion_engine.get_bound_positions()
            else:
                LEF_positions_1 = translocator1.extrusion_engine.get_positions()
            
            translocator1.state_trajectory.append(LEF_states_1)
            translocator1.lef_trajectory.append(LEF_positions_1)
            translocator1.ctcf_trajectory.append(CTCF_positions_1)
            
            # Collect data for translocator2
            LEF_states_2 = translocator2.extrusion_engine.get_states()
            CTCF_positions_2 = translocator2.barrier_engine.get_bound_positions()
            
            if prune_unbound_LEFs:
                LEF_positions_2 = translocator2.extrusion_engine.get_bound_positions()
            else:
                LEF_positions_2 = translocator2.extrusion_engine.get_positions()
            
            translocator2.state_trajectory.append(LEF_states_2)
            translocator2.lef_trajectory.append(LEF_positions_2)
            translocator2.ctcf_trajectory.append(CTCF_positions_2)
            
            # Track sisters
            if track_sisters:
                if hasattr(translocator1.extrusion_engine, 'sister_positions'):
                    sister_positions_1 = translocator1.extrusion_engine.sister_positions.copy()
                    coupling_status_1 = translocator1.extrusion_engine.get_coupling_status()
                    translocator1.sister_trajectory.append(sister_positions_1)
                    translocator1.coupling_trajectory.append(coupling_status_1)
                
                if hasattr(translocator2.extrusion_engine, 'sister_positions'):
                    sister_positions_2 = translocator2.extrusion_engine.sister_positions.copy()
                    coupling_status_2 = translocator2.extrusion_engine.get_coupling_status()
                    translocator2.sister_trajectory.append(sister_positions_2)
                    translocator2.coupling_trajectory.append(coupling_status_2)

run_synchronized_trajectories_continue(
    translocator1, translocator2,
    paramdict_dS['sister_lifetime'],
    steps = 64800, 
    prune_unbound_LEFs = True,
    track_sisters = True,
    sample_interval = 100,
    seed = 42,
    clear_trajectories=True  # Start fresh
)

end = time.time()
print(f"Run time: {end - start:.2f} seconds")
print(f"Before manual init: num_sisters = {translocator1.extrusion_engine}")


import pickle
with open('dS_trajectory1.pkl', 'wb') as f:
    pickle.dump({
        "sister": translocator1.sister_trajectory,
        "lef": translocator1.lef_trajectory,
        "ctcf": translocator1.ctcf_trajectory,
    }, f)

with open('dS_trajectory2.pkl', 'wb') as f:
    pickle.dump({
        "sister": translocator2.sister_trajectory,
        "lef": translocator2.lef_trajectory,
        "ctcf": translocator2.ctcf_trajectory,
    }, f)
print("Sister trajectory saved!")

