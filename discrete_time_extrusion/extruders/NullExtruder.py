import numpy as np


class NullExtruder():
    
    def __init__(self,
                 number,
                 barrier_engine,
                 *args, **kwargs):
    
        self.number = number
        self.barrier_engine = barrier_engine
        
        self.lattice_size = barrier_engine.lattice_size
        self.sites = np.arange(self.lattice_size, dtype=int)

        self.states = np.zeros(self.number, dtype=int)
        self.positions = np.zeros((self.number, 2), dtype=int) - 1
        
        self.occupied = np.zeros(self.lattice_size, dtype=bool)
        self.stalled = np.zeros((self.number, 2), dtype=bool)

        self.occupied[0] = self.occupied[-1] = True


    def step(self, *args, **kwargs):
    
        self.barrier_engine.step(self)
        
    
    def steps(self, N, *args, **kwargs):
    
        for _ in range(N):
            self.step(*args, **kwargs)
            
            
    def get_bound_positions(self):

        bound_ids = (self.positions >= 0).all(axis=1)
        bound_positions = self.positions[bound_ids]

        return bound_positions.tolist()
