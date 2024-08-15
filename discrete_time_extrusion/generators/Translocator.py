from . import arrays


class Translocator():

    def __init__(self,
                 extrusion_engine,
                 barrier_engine,
                 type_list,
                 site_types,
                 ctcf_left_positions,
                 ctcf_right_positions,
                 **kwargs):

        sites_per_replica = kwargs['monomers_per_replica'] * kwargs['sites_per_monomer']
        number_of_LEFs = (kwargs['number_of_replica'] * kwargs['monomers_per_replica']) // kwargs['LEF_separation']
        
        assert len(site_types) == sites_per_replica, ("Site type array (%d) doesn't match replica lattice size (%d)"
                                                      % (len(site_types), sites_per_replica))

        self.time_unit = 1. / (kwargs['sites_per_monomer'] * kwargs['velocity_multiplier'])

        lef_arrays = arrays.make_LEF_arrays(type_list, site_types, **kwargs)
        lef_transition_dict = arrays.make_LEF_transition_dict(type_list, site_types, **kwargs)

        ctcf_arrays = arrays.make_CTCF_arrays(type_list, site_types, ctcf_left_positions, ctcf_right_positions, **kwargs)
        ctcf_dynamic_arrays = arrays.make_CTCF_dynamic_arrays(type_list, site_types, **kwargs)
        
        self.barrier_engine = barrier_engine(*ctcf_arrays, *ctcf_dynamic_arrays)
        self.extrusion_engine = extrusion_engine(number_of_LEFs, self.barrier_engine, *lef_arrays, **lef_transition_dict)
                
        kwargs['steps'] = int(kwargs['steps'] / self.time_unit)
        kwargs['dummy_steps'] = int(kwargs['dummy_steps'] / self.time_unit)

        self.lef_trajectory = []
        self.ctcf_trajectory = []
        
        self.state_trajectory = []
        
        self.params = kwargs
    
    
    def run(self, period=None):

        period = int(period) if period else self.params['sites_per_monomer']
        self.extrusion_engine.steps(self.params['dummy_steps'] * period)
    
        for _ in range(self.params['steps']):
            self.extrusion_engine.steps(period)
        
            bound_LEF_positions = self.extrusion_engine.get_bound()
            bound_CTCF_positions = self.barrier_engine.get_bound()
            
            self.lef_trajectory.append(bound_LEF_positions)
            self.ctcf_trajectory.append(bound_CTCF_positions)
            
            self.state_trajectory.append(self.extrusion_engine.states.copy())
