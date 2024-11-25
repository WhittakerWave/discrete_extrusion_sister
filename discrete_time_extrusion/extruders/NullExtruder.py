try:
    import cupy as xp
    use_cuda = xp.cuda.is_available()
    
    if not use_cuda:
        raise ImportError

except:
    import numpy as xp


class NullExtruder():
    
    def __init__(self,
                 number,
                 barrier_engine,
                 *args, **kwargs):
    
        self.number = number
        self.barrier_engine = barrier_engine
        
        self.lattice_size = barrier_engine.lattice_size
        self.sites = xp.arange(self.lattice_size, dtype=xp.int32)

        self.states = xp.zeros(self.number, dtype=xp.int32)
        self.positions = xp.zeros((self.number, 2), dtype=xp.int32) - 1
        
        self.stalled = xp.zeros((self.number, 2), dtype=bool)
        self.occupied = xp.zeros(self.lattice_size, dtype=bool)

        self.update_occupancies()
        

    def step(self, *args, **kwargs):
    
        self.barrier_engine.step(self)
        
    
    def steps(self, N, *args, **kwargs):
    
        for _ in range(N):
            self.step(*args, **kwargs)
            

    def update_occupancies(self):
        
        ids = self.positions[xp.greater_equal(self.positions, 0)]
        
        self.occupied.fill(False)
        self.occupied[0] = self.occupied[-1] = True

        self.occupied[ids] = True
        
        
    def get_bound_positions(self):

        ids = xp.greater_equal(self.positions, 0).all(axis=1)
        bound_positions = self.positions[ids]

        return bound_positions.tolist()
