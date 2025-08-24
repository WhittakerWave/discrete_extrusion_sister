
from . import NullExtruder, EngineFactory

class BaseExtruder_Sister(NullExtruder.NullExtruder):
    
    def __init__(self,
                 number,
                 barrier_engine,
                 birth_prob,
                 death_prob,
                 stalled_death_prob,
                 diffusion_prob,
                 pause_prob,
                 *args, **kwargs):
    
        self.number = number
        # self.num_sisters = kwargs['num_of_sisters']
        self.is_sister = None
        self.sister_coupled = None
        self.coupled_to_extruder = {}   # sister_id -> extruder_id (many:1)
        self.coupled_to_sister = {}     # extruder_id -> [sister_ids] (1:many)

        super().__init__(number, barrier_engine)
        
        self.birth_prob = birth_prob
        self.death_prob = death_prob
        self.stalled_death_prob = stalled_death_prob
        self.diffusion_prob = diffusion_prob
        self.pause_prob = pause_prob
        self.stepping_engine = EngineFactory.SteppingEngine

        # Initialize sister management arrays
        self.is_sister = self.xp.zeros(self.number, dtype=bool)
        self.sister_coupled = self.xp.zeros(self.number, dtype=bool)  # Which sisters are coupled
        
        # Initialize sisters randomly on lattice
        self._initialize_sisters()
    
    def _initialize_sisters(self):
        """Initialize specified number of sisters at random positions on lattice"""
        if self.num_sisters <= 0:
            return
        # Find available particle slots for sisters
        available_ids = self.xp.flatnonzero(self.xp.equal(self.states, 0))  # unbound particles
    
        if len(available_ids) < self.num_sisters:
            print(f"WARNING: Only {len(available_ids)} unbound particles available, need {self.num_sisters}")
            return
            
        # Select random IDs for sisters
        sister_ids = self.xp.random.choice(available_ids, size=self.num_sisters, replace=False)
    
        # Get lattice size dynamically
        lattice_size = len(self.occupied)
        print(lattice_size)
        
        # Initialize each sister
        for sister_id in sister_ids:
            pos = self.xp.random.randint(0, lattice_size)
            
            self.positions[sister_id, 0] = pos  # Sister position
            self.positions[sister_id, 1] = -1   # No second leg
            self.states[sister_id] = 1          # bound state
            self.is_sister[sister_id] = True
            
            if hasattr(self, 'directions'):
                self.directions[sister_id] = 0
    
        print(f"Initialized {self.num_sisters} sisters")
    
    def check_sister_coupling(self):
        """Check if any sisters have landed on extruder leg positions"""
        # First, uncouple any sisters whose extruders have died
        self._check_extruder_deaths()
        
        # Check for new couplings
        for sister_id in range(self.number):
            if (self.is_sister[sister_id] and 
                self.states[sister_id] == 1 and      # Sister is active
                not self.sister_coupled[sister_id]): # Sister not already coupled
                
                sister_pos = self.positions[sister_id, 0]
                
                # Check if sister position matches any extruder leg position
                for extruder_id in range(self.number):
                    if (not self.is_sister[extruder_id] and 
                        self.states[extruder_id] == 1):  # Extruder not already coupled
                        
                        extruder_pos1 = self.positions[extruder_id, 0]
                        extruder_pos2 = self.positions[extruder_id, 1]
                        
                        # Check if sister is at same position as either extruder leg
                        if sister_pos == extruder_pos1 or sister_pos == extruder_pos2:
                            # Establish coupling
                            self.sister_coupled[sister_id] = True
                            self.coupled_to_extruder[sister_id] = extruder_id
                            if extruder_id not in self.coupled_to_sister:
                                self.coupled_to_sister[extruder_id] = []
                            self.coupled_to_sister[extruder_id].append(sister_id)  
                            break
    
    def _check_extruder_deaths(self):
        """Uncouple sisters whose extruders have died"""
        dead_extruders = []
        
        for sister_id, extruder_id in self.coupled_to_extruder.items():
            if self.states[extruder_id] == 0:  # Extruder died
                dead_extruders.append((sister_id, extruder_id))
        
        # Uncouple dead extruders but keep sisters alive
        for sister_id, extruder_id in dead_extruders:
            self.sister_coupled[sister_id] = False
            del self.coupled_to_extruder[sister_id]
            del self.coupled_to_sister[extruder_id]
    
    def birth(self, unbound_state_id):
        """Only extruders (non-sisters) can birth"""
        # Sisters never undergo birth - they're persistent
        eligible_mask = ~self.is_sister
        
        free_sites = self.sites[~self.occupied]
        binding_sites = self.xp.random.choice(free_sites, size=self.number, replace=False)
        rng = self.xp.less(self.xp.random.random(self.number), self.birth_prob[binding_sites])
        
        # Only allow birth for non-sister particles
        ids = self.xp.flatnonzero(rng * self.xp.equal(self.states, unbound_state_id) * eligible_mask)
                
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
        # Sisters never die - they persist even if uncoupled
        eligible_mask = ~self.is_sister
        
        death_prob = self.xp.where(self.stalled,
                                   self.stalled_death_prob[self.positions],
                                   self.death_prob[self.positions])
        death_prob = self.xp.max(death_prob, axis=1)
        
        rng = self.xp.less(self.xp.random.random(self.number), death_prob)
        ids = self.xp.flatnonzero(rng * self.xp.equal(self.states, bound_state_id) * eligible_mask)
        
        return ids
    
    def unload(self, ids_death):
        """Unload dead extruders and uncouple any associated sisters"""
        for extruder_id in ids_death:
            # If this extruder was coupled to a sister, uncouple them
            if extruder_id in self.coupled_to_sister:
                sister_id = self.coupled_to_sister[extruder_id]
                
                for sister_id in sister_list:  # ✅ Uncouple all sisters
                    self.sister_coupled[sister_id] = False
                    del self.coupled_to_extruder[sister_id]
                    
                del self.coupled_to_sister[extruder_id]
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
        """Return information about current couplings"""
        coupled_pairs = []
        uncoupled_sisters = []
        
        for i in range(self.number):
            if self.is_sister[i] and self.states[i] == 1:
                if self.sister_coupled[i]:
                    extruder_id = self.coupled_to_extruder[i]
                    coupled_pairs.append({
                        'sister_id': i,
                        'extruder_id': extruder_id,
                        'sister_pos': self.positions[i, 0],
                        'extruder_pos1': self.positions[extruder_id, 0],
                        'extruder_pos2': self.positions[extruder_id, 1]
                    })
                else:
                    uncoupled_sisters.append({
                        'sister_id': i,
                        'sister_pos': self.positions[i, 0]
                    })
        
        return {
            'coupled_pairs': coupled_pairs,
            'uncoupled_sisters': uncoupled_sisters,
            'total_sisters': len(coupled_pairs) + len(uncoupled_sisters)
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
            'is_sister': self.is_sister,
            'sister_coupled': self.sister_coupled,
            'coupled_to_extruder': self.coupled_to_extruder,
            'coupled_to_sister': self.coupled_to_sister
        })
        
        # Movement step with coupling information
        self.stepping_engine(self, mode, unbound_state_id, active_state_id, **kwargs)
