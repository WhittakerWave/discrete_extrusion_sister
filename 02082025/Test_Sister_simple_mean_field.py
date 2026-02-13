

import sys
import os
import json
# import hoomd
import codecs
# import cooltools
import time
import pickle
import numpy as np
import matplotlib.pyplot as plt

# import polychrom_hoomd
# import polychrom_hoomd.log as log
# import polychrom_hoomd.build as build
# import polychrom_hoomd.utils as utils
# import polychrom_hoomd.forces as forces
# import polychrom_hoomd.render as render
# import polychrom_hoomd.extrude as extrude
# from polykit.analysis import polymer_analyses, contact_maps
# from polykit.generators.initial_conformations import grow_cubic
# import gsd.hoomd

from discrete_time_extrusion_02082026.Translocator_Sister_v02092026 import Translocator_Sister
from discrete_time_extrusion_02082026.boundaries.StaticBoundary import StaticBoundary
from discrete_time_extrusion_02082026.boundaries.DynamicBoundary import DynamicBoundary
from discrete_time_extrusion_02082026.extruders.BaseExtruder_Sister_v02092026 import BaseExtruder_Sister


with open("extrusion_dict_simple_mean_field.json", 'r') as dict_file:
        paramdict = json.load(dict_file)
    
monomers_per_replica = paramdict['monomers_per_replica'] 
sites_per_monomer = paramdict['sites_per_monomer']

# sites_per_replica = monomers_per_replica*sites_per_monomer
# Work with a single type of monomers (A, assigned to type index 0)
type_list = ['A']
monomer_types = type_list.index('A') * np.ones(monomers_per_replica, dtype=int)
site_types = np.repeat(monomer_types, sites_per_monomer)

# LEF/CTCF properties in type A monomers may be obtained from the paramdict as follows
LEF_off_rate = paramdict['LEF_off_rate']
CTCF_facestall = paramdict['CTCF_facestall']
print(LEF_off_rate['A'], CTCF_facestall['A'])

# anchor_positions = np.genfromtxt(f'{path}/anchor_{N_sister_RAD21}_{iteration}.txt')
anchor_positions = []
# Create some CTCF boundary sites
# ctcf_left_positions = anchor_positions
# ctcf_right_positions = anchor_positions

### keep the smae CTCF left and right 
# number_of_ctcf = 2540
# ctcf_left_positions = np.random.choice(monomers_per_replica, size=number_of_ctcf, replace=False)
# ctcf_right_positions =  ctcf_left_positions.copy()

### sample half and half
number_of_ctcf = 364
# ctcf_positions = np.random.choice(monomers_per_replica, size=number_of_ctcf, replace=False)
# shuffle and split into two halves
# np.random.shuffle(ctcf_positions)
# half = number_of_ctcf // 2
# ctcf_left_positions = ctcf_positions[:half]
# np.save('ctcf_left_positions.npy', ctcf_left_positions)
# ctcf_right_positions = ctcf_positions[half:]
# np.save('ctcf_right_positions.npy', ctcf_right_positions)

# ctcf_left_positions = np.load('ctcf_left_positions_simple.npy')
# ctcf_right_positions = np.load('ctcf_right_positions_simple.npy')

# np.save('ctcf_left_positions.npy', ctcf_left_positions)
# ctcf_left_positions = np.load('ctcf_left_positions.npy')
# ctcf_right_positions =  ctcf_left_positions.copy()
ctcf_left_positions = []
ctcf_right_positions = [0]
np.save('ctcf_left_positions.npy', ctcf_left_positions)
np.save('ctcf_right_positions.npy', ctcf_right_positions)

start = time.time()
translocator1 = Translocator_Sister(BaseExtruder_Sister,
                    DynamicBoundary,
                    type_list, 
                    site_types,
                    ctcf_left_positions,
                    ctcf_right_positions, 
                    initial_sister_positions = None,
                    **paramdict)

# translocator1.run(10000)
translocator1.run_trajectory_one_sister(steps = 65000, prune_unbound_LEFs=True, track_sisters=True, sample_interval=100)

end = time.time()
print(f"Run time: {end - start:.2f} seconds")
print(f"Before manual init: num_sisters = {translocator1.extrusion_engine}")

with open('test_case_mean_field.pkl', 'wb') as f:
    pickle.dump({
        "sister": translocator1.sister_trajectory,
        "lef": translocator1.lef_trajectory,
        "ctcf": translocator1.ctcf_trajectory,
    }, f)
print("Sister trajectory saved!")

translocator1.extrusion_engine._initialize_sisters()
