import numpy as np

from . import StaticBoundary


class DynamicBoundary(StaticBoundary.StaticBoundary):
     
    def __init__(self,
                 stall_prob_left,
                 stall_prob_right,
                 ctcf_birth_prob,
                 ctcf_death_prob,
                 *args, **kwargs):
        
        super().__init__(stall_prob_left, stall_prob_right)
        
        self.ctcf_birth_prob = ctcf_birth_prob
        self.ctcf_death_prob = ctcf_death_prob
            
        occupancy = ctcf_birth_prob / (ctcf_birth_prob + ctcf_death_prob)

        self.states_left = stall_prob_right > 0
        self.states_right = stall_prob_right > 0
                
        rng_left = np.random.random(self.num_site) < occupancy
        rng_right = np.random.random(self.num_site) < occupancy
        
        self.states_left = np.where(self.states_left,
                                    self.states_left*rng_left,
                                    -1)
        self.states_right = np.where(self.states_right,
                                     self.states_right*rng_right,
                                     -1)

        self.prob_left = (self.states_left == 1)
        self.prob_right = (self.states_right == 1)
                         

    def ctcf_birth(self):
    
        rng_left = np.random.random(self.num_site) < self.ctcf_birth_prob
        rng_right = np.random.random(self.num_site) < self.ctcf_birth_prob

        ids_left = np.flatnonzero(rng_left * (self.states_left == 0))
        ids_right = np.flatnonzero(rng_right * (self.states_right == 0))
        
        self.prob_left[ids_left] = 1
        self.prob_right[ids_right] = 1
        
        return ids_left, ids_right
                
        
    def ctcf_death(self):

        rng_left = np.random.random(self.num_site) < self.ctcf_death_prob
        rng_right = np.random.random(self.num_site) < self.ctcf_death_prob

        ids_left = np.flatnonzero(rng_left * (self.states_left == 1))
        ids_right = np.flatnonzero(rng_right * (self.states_right == 1))
        
        self.prob_left[ids_left] = 0
        self.prob_right[ids_right] = 0
        
        return ids_left, ids_right

    
    def step(self, extrusion_engine):
    
        ids_death = self.ctcf_death()
        ids_birth = self.ctcf_birth()

        self.states_left[ids_death[0]] = 0
        self.states_left[ids_birth[0]] = 1

        self.states_right[ids_death[1]] = 0
        self.states_right[ids_birth[1]] = 1

        lef_ids_left = np.flatnonzero(np.in1d(extrusion_engine.positions[:, 0], ids_death[0]))
        lef_ids_right = np.flatnonzero(np.in1d(extrusion_engine.positions[:, 1], ids_death[1]))

        extrusion_engine.stalled[lef_ids_left, 0] = 0
        extrusion_engine.stalled[lef_ids_right, 1] = 0
