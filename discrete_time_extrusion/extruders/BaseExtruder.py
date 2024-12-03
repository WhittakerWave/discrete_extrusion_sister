from . import NullExtruder, EngineFactory
    

class BaseExtruder(NullExtruder.NullExtruder):
    
    def __init__(self,
                 number,
                 barrier_engine,
                 birth_prob,
                 death_prob,
                 stalled_death_prob,
                 pause_prob,
                 *args, **kwargs):
    
        super().__init__(number,
                         barrier_engine)
		
        self.birth_prob = birth_prob
        self.death_prob = death_prob

        self.pause_prob = pause_prob
        self.stalled_death_prob = stalled_death_prob
        
        self.stepping_engine = EngineFactory.SteppingEngine
        
                            
    def birth(self, unbound_state_id):
    
        free_sites = self.sites[~self.occupied]
        binding_sites = self.xp.random.choice(free_sites, size=self.number, replace=False)

        rng = self.xp.less(self.xp.random.random(self.number), self.birth_prob[binding_sites])
        ids = self.xp.flatnonzero(rng * self.xp.equal(self.states, unbound_state_id))
                
        if len(ids) > 0:
            binding_sites = binding_sites[ids]
            self.positions[ids] = binding_sites[:, None]
        
            rng_stagger = self.xp.less(self.xp.random.random(len(ids)), 0.5) * ~self.occupied[binding_sites+1]

            self.positions[ids, 1] = self.xp.where(rng_stagger,
                                                   self.positions[ids, 1] + 1,
                                                   self.positions[ids, 1])
                                                           
        return ids
                                                                                
        
    def death(self, bound_state_id):
    
        death_prob = self.xp.where(self.stalled,
                                   self.stalled_death_prob[self.positions],
                                   self.death_prob[self.positions])
        death_prob = self.xp.max(death_prob, axis=1)
        
        rng = self.xp.less(self.xp.random.random(self.number), death_prob)
        ids = self.xp.flatnonzero(rng * self.xp.equal(self.states, bound_state_id))
        
        return ids
        
	
    def unload(self, ids_death):
    
        self.stalled[ids_death] = False
        self.positions[ids_death] = -1
        
        
    def update_states(self, unbound_state_id, bound_state_id):
    
        ids_birth = self.birth(unbound_state_id)
        ids_death = self.death(bound_state_id)
        
        self.states[ids_birth] = bound_state_id
        self.states[ids_death] = unbound_state_id

        self.unload(ids_death)
        

    def step(self, mode, unbound_state_id=0, bound_state_id=1, active_state_id=1, **kwargs):
    
        self.update_states(unbound_state_id, bound_state_id)
        self.update_occupancies()

        super().step(**kwargs)

        self.stepping_engine(self, mode, active_state_id, **kwargs)
