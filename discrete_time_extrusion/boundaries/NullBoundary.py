import numpy as np


class NullBoundary():
     
    def __init__(self,
                 stall_left,
                 stall_right,
                 *args, **kwargs):
                 
        self.number = 0
        self.lattice_size = len(stall_left)

        self.stall_left = np.zeros_like(stall_left)
        self.stall_right = np.zeros_like(stall_right)

                                
    def step(self, *args, **kwargs):
    
        pass
        
        
    def get_bound_positions(self):

        bound_left_positions = np.flatnonzero(self.stall_left)
        bound_right_positions = np.flatnonzero(self.stall_right)
    
        return bound_left_positions.tolist() + bound_right_positions.tolist()
