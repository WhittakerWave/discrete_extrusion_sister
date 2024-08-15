import numpy as np

        
class StaticBoundary():
     
    def __init__(self,
                 stall_left,
                 stall_right,
                 *args, **kwargs):
        
        self.lattice_size = len(stall_left)
        self.number = np.count_nonzero(stall_left) + np.count_nonzero(stall_right)

        self.stall_left = stall_left
        self.stall_right = stall_right
        
                                
    def step(self, *args):
    
        pass
