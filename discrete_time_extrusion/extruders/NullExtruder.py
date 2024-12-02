class NullExtruder():
    
    def __init__(self,
                 number,
                 barrier_engine,
                 *args,
                 **kwargs):
    
        self.number = number
        
        self.xp = barrier_engine.xp
        self.barrier_engine = barrier_engine
        
        self.lattice_size = barrier_engine.lattice_size
        self.sites = self.xp.arange(self.lattice_size, dtype=self.xp.int32)

        self.states = self.xp.zeros(self.number, dtype=self.xp.int32)
        self.positions = self.xp.zeros((self.number, 2), dtype=self.xp.int32) - 1
        
        self.stalled = self.xp.zeros((self.number, 2), dtype=bool)
        self.occupied = self.xp.zeros(self.lattice_size, dtype=bool)

        self.update_occupancies()
        

    def step(self, *args, **kwargs):
    
        self.barrier_engine.step(self)
        
    
    def steps(self, N, *args, **kwargs):
    
        for _ in range(N):
            self.step(*args, **kwargs)
            

    def update_occupancies(self):
        
        ids = self.positions[self.xp.greater_equal(self.positions, 0)]
        
        self.occupied.fill(False)
        self.occupied[0] = self.occupied[-1] = True

        self.occupied[ids] = True
        
        
    def get_bound_positions(self):

        ids = self.xp.greater_equal(self.positions, 0).all(axis=1)
        bound_positions = self.positions[ids]

        return bound_positions.tolist()
