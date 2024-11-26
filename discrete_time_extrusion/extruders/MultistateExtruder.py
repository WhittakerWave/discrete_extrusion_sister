from . import SimpleExtruder

try:
    import cupy as xp
    use_cuda = xp.cuda.is_available()
    
    if not use_cuda:
        raise ImportError

except:
    import numpy as xp
    

class MultistateExtruder(SimpleExtruder.SimpleExtruder):
    
    def __init__(self,
                 number,
                 barrier_engine,
                 birth_prob,
                 death_prob,
                 stalled_death_prob,
                 pause_prob,
                 *args,
                 **transition_dict):
    
        super().__init__(number,
                         barrier_engine,
                         birth_prob,
                         death_prob,
                         stalled_death_prob,
                         pause_prob)
        
        self.state_dict = transition_dict["LEF_states"]
        self.transition_dict = transition_dict["LEF_transitions"]
        

    def state_transitions(self, unbound_state_id):
        
        ids_array = xp.zeros((len(self.state_dict), self.number), dtype=xp.int32)
        products_array = xp.zeros((len(self.state_dict), self.number), dtype=xp.int32)
            
        for id, state_id in enumerate(self.state_dict.values()):
            ctr = 0
            buffer = 1 if state_id == min(self.state_dict.values()) else 2
            
            state_array = xp.zeros(buffer, dtype=xp.int32)
            transition_array = xp.zeros((buffer, self.number), dtype=xp.float32)
            
            for ids, transition_prob in self.transition_dict.items():
                if state_id == int(ids[0]):
                    state_array[ctr] = int(ids[1])
                    transition_array[ctr] = transition_prob[self.positions].max(axis=1)
                    
                    ctr += 1
                    
            if state_id == max(self.state_dict.values()):
                death_prob = xp.where(self.stalled,
                                      self.stalled_death_prob[self.positions],
                                      self.death_prob[self.positions])
                                      
                state_array[-1] = unbound_state_id
                transition_array[-1] = death_prob.max(axis=1)

            rng = xp.random.random(self.number)
            cumul_prob = xp.cumsum(transition_array, axis=0)

            rng1 = xp.less(rng, cumul_prob[0])
            rng2 = xp.logical_and(xp.logical_not(rng1), xp.less(rng, cumul_prob[-1]))
            
            product_states = xp.where(rng1, state_array[0], state_array[-1])
            transitions = xp.logical_and(xp.logical_or(rng1, rng2), xp.equal(self.states, state_id))
        
            ids = xp.flatnonzero(transitions)
            products = product_states[ids]
            
            ids_array[id] = xp.pad(ids, (0, self.number-len(ids)), constant_values=(0, -1))
            products_array[id] = xp.pad(products, (0, self.number-len(ids)), constant_values=(0, -1))
            
        return ids_array, products_array
            
        
    def update_states(self, unbound_state_id, bound_state_id):
        
        ids_array, products_array = self.state_transitions(unbound_state_id)
        
        ids_birth = self.birth(unbound_state_id)
        self.states[ids_birth] = bound_state_id
        
        for ids, products in zip(ids_array, products_array):
            self.states[ids] = xp.where(xp.greater_equal(ids, 0), products, self.states[ids])
            
        ids_death = ids[xp.equal(products, unbound_state_id)]
        self.unload(ids_death)


    def step(self, mode, **kwargs):
    
        super().step(mode, active_state_id=self.state_dict['RN'], **kwargs)
