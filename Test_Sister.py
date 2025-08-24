

import sys
import os
import json
# import hoomd
import codecs
# import cooltools
import time
import numpy as np
import matplotlib.pyplot as plt

import discrete_time_extrusion

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

from discrete_time_extrusion.Translocator import Translocator
from discrete_time_extrusion.Translocator_Sister import Translocator_Sister
from discrete_time_extrusion.boundaries.StaticBoundary import StaticBoundary
# from discrete_time_extrusion.boundaries.DynamicBoundary import DynamicBoundary
# from discrete_time_extrusion.extruders.MultistateExtruder import MultistateExtruder
from discrete_time_extrusion.extruders.BaseExtruder_Sister import BaseExtruder_Sister
from discrete_time_extrusion.extruders.BaseExtruder import BaseExtruder

with open("data/extrusion_dict_test.json", 'r') as dict_file:
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
ctcf_left_positions = anchor_positions
ctcf_right_positions = anchor_positions

translocator1 = Translocator_Sister(BaseExtruder_Sister,
                            StaticBoundary,
                            type_list, 
                            site_types,
                            ctcf_left_positions,
                            ctcf_right_positions, 
                            **paramdict)

translocator1.run(10000)
print(f"Before manual init: num_sisters = {translocator1.extrusion_engine}")
translocator1.extrusion_engine._initialize_sisters()

