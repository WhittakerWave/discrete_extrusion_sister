import numpy as np

        
class StaticBoundary():
     
    def __init__(self,
                 stall_prob_left,
                 stall_prob_right,
                 *args, **kwargs):
        
        self.num_site = len(stall_prob_left)
        self.num_CTCF = np.count_nonzero(stall_prob_left) + np.count_nonzero(stall_prob_right)

        self.prob_left = stall_prob_left
        self.prob_right = stall_prob_right
        
                                
    def step(self, extrusion_engine):
    
        pass
