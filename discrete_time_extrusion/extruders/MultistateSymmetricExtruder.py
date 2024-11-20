from . import SymmetricExtruder

try:
    import cupy as xp
    use_cuda = xp.cuda.is_available()
    
    if not use_cuda:
        raise ImportError

except:
    import numpy as xp
    

class MultistateSymmetricExtruder(SymmetricExtruder.SymmetricExtruder):
    
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
        
        ids_list = []
        products_list = []
            
        for state_id in self.state_dict.values():
            state_list = []
            transition_list = []
            
            for ids, transition_prob in self.transition_dict.items():
                if state_id == int(ids[0]):
                    state_list.append(int(ids[1]))
                    transition_list.append(transition_prob[self.positions].max(axis=1))
                    
            if state_id == max(self.state_dict.values()):
                death_prob = xp.where(self.stalled,
                                      self.stalled_death_prob[self.positions],
                                      self.death_prob[self.positions])
                                      
                state_list.append(unbound_state_id)
                transition_list.append(death_prob.max(axis=1))

            rng = xp.random.random(self.number)
            cumul_prob = xp.cumsum(transition_list, axis=0)
            
            rng1 = (rng < cumul_prob[0])
            rng2 = ~rng1 * (rng < cumul_prob[-1])
            
            product_states = xp.where(rng1, state_list[0], state_list[-1])
        
            ids = xp.flatnonzero((rng1 + rng2) * (self.states == state_id))
            products = product_states[ids]
            
            ids_list.append(ids)
            products_list.append(products)
            
        return ids_list, products_list
            
        
    def update_states(self, unbound_state_id, bound_state_id):
        
        ids_list, products_list = self.state_transitions(unbound_state_id)
        ids_birth = self.birth(unbound_state_id)
        
        self.states[ids_birth] = bound_state_id
        
        for ids, products in zip(ids_list, products_list):
            self.states[ids] = products
            
        ids_death = ids[products == unbound_state_id]
        
        self.unload(ids_death)


    def step(self):
    
        super().step(active_state_id=self.state_dict['RN'])
