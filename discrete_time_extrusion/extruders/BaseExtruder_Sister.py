
from . import NullExtruder, EngineFactory

class BaseExtruder_Sister(NullExtruder.NullExtruder):
    
    def __init__(self,
                 number,
                 number_of_sisters, 
                 barrier_engine,
                 birth_prob,
                 death_prob,
                 stalled_death_prob,
                 diffusion_prob,
                 pause_prob,
                 *args, **kwargs):
    
        self.number = number
        self.num_sisters = number_of_sisters

        # Coupling dictionaries 
        self.coupled_to_extruder = {}   # sister_id -> extruder_id (many:1)
        self.coupled_to_sister = {}     # extruder_id -> [sister_ids] (1:many)

        super().__init__(number, barrier_engine)
        
        self.birth_prob = birth_prob
        self.death_prob = death_prob
        self.stalled_death_prob = stalled_death_prob
        self.diffusion_prob = diffusion_prob
        self.pause_prob = pause_prob
        self.stepping_engine = EngineFactory.SteppingEngine
        
        # Sister-specific attributes (separate from extruder positions/states)
        self.sister_positions = None      # Separate position array for sisters
        # self.sister_states = None         # Separate state array for sisters  
        # self.is_sister_active = None      # Which sisters are active
        # Initialize sister management arrays
        # self.is_sister = self.xp.zeros(self.number, dtype=bool)
        # self.sister_coupled = self.xp.zeros(self.number, dtype=bool)  # Which sisters are coupled
        # Initialize sisters randomly on lattice
        self._initialize_sisters()
    
    def _initialize_sisters(self):
        """Initialize sisters as separate entities (not using extruder position arrays)"""
        if self.num_sisters <= 0:
            print("No sisters to initialize")
            return
        
        lattice_size = len(self.occupied)
        print(f"Initializing {self.num_sisters} sisters on lattice of size {lattice_size}")
    
        # ✅ Create separate arrays for sisters
        self.sister_positions = self.xp.zeros(self.num_sisters, dtype=self.xp.int32)
        # self.sister_states = self.xp.ones(self.num_sisters, dtype=self.xp.int32)  # All start active
        # self.is_sister_active = self.xp.ones(self.num_sisters, dtype=bool)
    
        # Randomly place sisters on lattice
        for sister_id in range(self.num_sisters):
            # Find a free position on the lattice
            pos = self._find_free_position_for_sister()
            self.sister_positions[sister_id] = pos
            # Mark lattice position as occupied by sister
            # if pos >= 0:
            #    self.occupied[pos] = True  
        print(f"Initialized {self.num_sisters} sisters at positions: {self.sister_positions}")
    
    def _find_free_position_for_sister(self):
        """Find a free position on the lattice for a sister"""
        lattice_size = len(self.occupied)
        max_attempts = lattice_size
        for _ in range(max_attempts):
            pos = self.xp.random.randint(0, lattice_size)
            if not self.occupied[pos]:
                return pos
        print("WARNING: Could not find free position for sister")
        return -1
    
    def is_sister_coupled(self, sister_id):
        """✅ HELPER: Check if sister is coupled"""
        return sister_id in self.coupled_to_extruder
    
    def check_sister_coupling(self):
        """Check if any sisters have landed on extruder leg positions"""
        # First, uncouple any sisters whose extruders have died
        self._check_extruder_deaths()
        
        # Check for new couplings
        for sister_id in range(self.num_sisters):
            if not self.is_sister_coupled(sister_id):  # Sister is free
                sister_pos = self.sister_positions[sister_id]  
                # Check if sister position matches any extruder leg position
                for extruder_id in range(self.number):
                    if self.states[extruder_id] !=0:  # Extruder is active
                        extruder_pos1 = self.positions[extruder_id, 0]
                        extruder_pos2 = self.positions[extruder_id, 1]
                        
                        # Check if sister is at same position as either extruder leg
                        if sister_pos == extruder_pos1 or sister_pos == extruder_pos2:
                            # Establish coupling
                            self.coupled_to_extruder[sister_id] = extruder_id
                            if extruder_id not in self.coupled_to_sister:
                                self.coupled_to_sister[extruder_id] = []
                            self.coupled_to_sister[extruder_id].append(sister_id)  
                            print(f"Sister {sister_id} coupled to extruder {extruder_id} at position {sister_pos}")
                            break
    
    def _check_extruder_deaths(self):
        """Uncouple sisters whose extruders have died"""
        dead_couplings = []
        
        for sister_id, extruder_id in self.coupled_to_extruder.items():
            if self.states[extruder_id] == 0:  # Extruder died
                dead_couplings.append((sister_id, extruder_id))
        
        # Uncouple dead extruders but keep sisters alive
        for sister_id, extruder_id in dead_couplings:
            del self.coupled_to_extruder[sister_id]
            if extruder_id in self.coupled_to_sister:
                self.coupled_to_sister[extruder_id].remove(sister_id)
                if not self.coupled_to_sister[extruder_id]:  # Empty list
                    del self.coupled_to_sister[extruder_id]
    
    def birth(self, unbound_state_id):
        """Only extruders (non-sisters) can birth"""
        
        free_sites = self.sites[~self.occupied]
        binding_sites = self.xp.random.choice(free_sites, size=self.number, replace=False)

        rng = self.xp.less(self.xp.random.random(self.number), self.birth_prob[binding_sites])
        ids = self.xp.flatnonzero(rng * self.xp.equal(self.states, unbound_state_id))
                
        if len(ids) > 0:
            binding_sites = binding_sites[ids]
            self.positions[ids] = binding_sites[:, None]  # Both legs start at same position
        
            rng_dir = self.xp.less(self.xp.random.random(len(ids)), 0.5)
            rng_stagger = self.xp.less(self.xp.random.random(len(ids)), 0.5) * ~self.occupied[binding_sites+1]
            
            # Second leg can expand if adjacent site is free
            self.positions[ids, 1] = self.xp.where(rng_stagger,
                                                   self.positions[ids, 1] + 1,
                                                   self.positions[ids, 1])
            self.directions[ids] = rng_dir.astype(self.xp.uint32)
                                                           
        return ids
    
    def death(self, bound_state_id):
        """Only extruders (non-sisters) can die"""
        
        death_prob = self.xp.where(self.stalled,
                                   self.stalled_death_prob[self.positions],
                                   self.death_prob[self.positions])
        death_prob = self.xp.max(death_prob, axis=1)
        
        rng = self.xp.less(self.xp.random.random(self.number), death_prob)
        ids = self.xp.flatnonzero(rng * self.xp.equal(self.states, bound_state_id))
        
        return ids
    
    def unload(self, ids_death):
        """Unload dead extruders and uncouple any associated sisters"""
        for extruder_id in ids_death:
            # If this extruder was coupled to a sister, uncouple them
            if extruder_id in self.coupled_to_sister:
                sister_list = self.coupled_to_sister[extruder_id] 
                for sister_id in sister_list:  # ✅ Uncouple all sisters
                    if sister_id in self.coupled_to_extruder:         
                        del self.coupled_to_extruder[sister_id]
                del self.coupled_to_sister[extruder_id]
                print(f"Uncoupled {len(sister_list)} sisters from dead extruder {extruder_id}")
        # Standard unload for extruders only
        self.stalled[ids_death] = False
        self.positions[ids_death] = -1

    def update_states(self, unbound_state_id, bound_state_id):
        
        ids_birth = self.birth(unbound_state_id)
        ids_death = self.death(bound_state_id)
        
        self.states[ids_birth] = bound_state_id
        self.states[ids_death] = unbound_state_id

        self.unload(ids_death)
    
    def update_occupancies(self):
        """Update lattice occupancy including both extruders and sisters"""
        super().update_occupancies()
        
        # Sisters also occupy lattice sites
        # Check if sister attributes are properly initialized
        if hasattr(self, 'is_sister') and self.is_sister is not None:
            for i in range(self.number):
                if self.is_sister[i] and self.states[i] == 1:
                    pos = self.positions[i, 0]
                    if pos >= 0:
                        self.occupied[pos] = True
    
    def get_coupling_status(self):
        """✅ FIXED: Return coupling information using correct data"""
        coupled_pairs = []
        uncoupled_sisters = []
        
        for sister_id in range(self.num_sisters):  # ✅ FIXED: Loop through sisters
            sister_pos = int(self.sister_positions[sister_id])
            
            if self.is_sister_coupled(sister_id):
                extruder_id = self.coupled_to_extruder[sister_id]
                coupled_pairs.append({
                    'sister_id': sister_id,
                    'extruder_id': extruder_id,
                    'sister_pos': sister_pos,
                    'extruder_pos1': int(self.positions[extruder_id, 0]),
                    'extruder_pos2': int(self.positions[extruder_id, 1])
                })
            else:
                uncoupled_sisters.append({
                    'sister_id': sister_id,
                    'sister_pos': sister_pos
                })
        
        return {
            'coupled_pairs': coupled_pairs,
            'uncoupled_sisters': uncoupled_sisters,
            'total_sisters': self.num_sisters
        }
    
    def step(self, mode, unbound_state_id = 0, bound_state_id = 1, active_state_id = 1, **kwargs):
        """Main step function with sister coupling checks"""
        # Standard extruder birth/death updates
        self.update_states(unbound_state_id, bound_state_id)
        
        # Check for new sister-extruder couplings
        self.check_sister_coupling()
        
        # Update occupancies (includes sisters)
        self.update_occupancies()
        
        # Parent class step
        super().step(**kwargs)
        
        # Pass coupling information to stepping engine
        kwargs.update({
            'coupled_to_extruder': self.coupled_to_extruder,
            'coupled_to_sister': self.coupled_to_sister
        })
        
        # Movement step with coupling information
        self.stepping_engine(self, mode, unbound_state_id, active_state_id, **kwargs)
