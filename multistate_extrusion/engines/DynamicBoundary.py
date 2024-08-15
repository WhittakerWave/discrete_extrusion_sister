import numpy as np

from . import StaticBoundary


class DynamicBoundary(StaticBoundary.StaticBoundary):
     
    def __init__(self,
                 stall_left,
                 stall_right,
                 birth_prob,
                 death_prob,
                 *args, **kwargs):
        
        super().__init__(stall_left, stall_right)
        
        self.birth_prob = birth_prob
        self.death_prob = death_prob
            
        occupancy = birth_prob / (birth_prob + death_prob)

        self.states_left = stall_left > 0
        self.states_right = stall_right > 0
                
        rng_left = np.random.random(self.lattice_size) < occupancy
        rng_right = np.random.random(self.lattice_size) < occupancy
        
        self.states_left = np.where(self.states_left,
                                    self.states_left*rng_left,
                                    -1)
        self.states_right = np.where(self.states_right,
                                     self.states_right*rng_right,
                                     -1)

        self.stall_left = (self.states_left == 1)
        self.stall_right = (self.states_right == 1)
                         

    def birth(self, unbound_state_id):
    
        rng_left = np.random.random(self.lattice_size) < self.birth_prob
        rng_right = np.random.random(self.lattice_size) < self.birth_prob

        ids_left = np.flatnonzero(rng_left * (self.states_left == unbound_state_id))
        ids_right = np.flatnonzero(rng_right * (self.states_right == unbound_state_id))
        
        self.stall_left[ids_left] = 1
        self.stall_right[ids_right] = 1
        
        return ids_left, ids_right
                
        
    def death(self, bound_state_id):

        rng_left = np.random.random(self.lattice_size) < self.death_prob
        rng_right = np.random.random(self.lattice_size) < self.death_prob

        ids_left = np.flatnonzero(rng_left * (self.states_left == bound_state_id))
        ids_right = np.flatnonzero(rng_right * (self.states_right == bound_state_id))
        
        self.stall_left[ids_left] = 0
        self.stall_right[ids_right] = 0
        
        return ids_left, ids_right

    
    def step(self, extrusion_engine, unbound_state_id=0, bound_state_id=1):
    
        ids_birth = self.birth(unbound_state_id)
        ids_death = self.death(bound_state_id)

        self.states_left[ids_birth[0]] = bound_state_id
        self.states_left[ids_death[0]] = unbound_state_id

        self.states_right[ids_birth[1]] = bound_state_id
        self.states_right[ids_death[1]] = unbound_state_id

        lef_ids_left = np.flatnonzero(np.in1d(extrusion_engine.positions[:, 0], ids_death[0]))
        lef_ids_right = np.flatnonzero(np.in1d(extrusion_engine.positions[:, 1], ids_death[1]))

        extrusion_engine.stalled[lef_ids_left, 0] = 0
        extrusion_engine.stalled[lef_ids_right, 1] = 0
