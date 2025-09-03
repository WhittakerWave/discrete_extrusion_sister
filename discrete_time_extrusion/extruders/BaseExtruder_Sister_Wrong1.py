

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

        # Initialize sister attributes BEFORE calling super().__init__
        self.sister_positions = None
        self.sister_coupled_to = None       # Array: sister_id -> extruder_id (-1 if uncoupled)
        self.extruder_sister_counts = None  # Array: extruder_id -> number of coupled sisters
        
        # Dictionary format for backward compatibility and stepping engine
        self.coupled_to_extruder = {}       # sister_id -> extruder_id [many:1]
        self.coupled_to_sister = {}         # extruder_id -> [sister_ids] [1:many]

        super().__init__(number, barrier_engine)
        
        self.birth_prob = birth_prob
        self.death_prob = death_prob
        self.stalled_death_prob = stalled_death_prob
        self.diffusion_prob = diffusion_prob
        self.pause_prob = pause_prob
        self.stepping_engine = EngineFactory.SteppingEngine
        
        # Initialize sisters after parent constructor completes
        self._initialize_sisters()
    
    def _initialize_sisters(self):
        """Vectorized sister initialization"""
        if self.num_sisters <= 0:
            print("No sisters to initialize")
            return
        
        lattice_size = len(self.occupied)
        print(f"Initializing {self.num_sisters} sisters on lattice of size {lattice_size}")

        # Initialize arrays
        self.sister_positions = self.xp.zeros(self.num_sisters, dtype=self.xp.int32)
        self.sister_coupled_to = self.xp.full(self.num_sisters, -1, dtype=self.xp.int32)
        self.extruder_sister_counts = self.xp.zeros(self.number, dtype=self.xp.int32)

        # Vectorized placement: find all free positions at once
        free_positions = self.xp.where(~self.occupied)[0]
        
        if len(free_positions) >= self.num_sisters:
            selected_positions = self.xp.random.choice(
                free_positions, size=self.num_sisters, replace=False
            )
            self.sister_positions = selected_positions
        else:
            print(f"WARNING: Not enough free positions ({len(free_positions)}) for {self.num_sisters} sisters")
            num_to_place = min(self.num_sisters, len(free_positions))
            self.sister_positions[:num_to_place] = free_positions[:num_to_place]
            self.sister_positions[num_to_place:] = -1
            
        print(f"Initialized {self.num_sisters} sisters at positions: {self.sister_positions}")
    
    def check_sister_coupling(self):
        """Fully vectorized coupling check - NO nested loops"""
        if self.sister_positions is None or self.sister_coupled_to is None:
            return
        
        # First pass: uncouple sisters whose extruders have died (already vectorized)
        self._uncouple_dead_extruders()
        
        # Second pass: find new couplings using vectorized operations
        # Get all uncoupled sisters
        uncoupled_mask = self.sister_coupled_to == -1
        uncoupled_sisters = self.xp.where(uncoupled_mask)[0]
        
        if len(uncoupled_sisters) == 0:
            return
        
        # Get all active extruders and their positions
        active_mask = self.states != 0
        active_extruders = self.xp.where(active_mask)[0]
        
        if len(active_extruders) == 0:
            return
        
        # Get all extruder leg positions (vectorized)
        active_positions = self.positions[active_extruders]  # Shape: (n_active, 2)
        extruder_pos1 = active_positions[:, 0]  # All left legs
        extruder_pos2 = active_positions[:, 1]  # All right legs
        
        # For each uncoupled sister, check against all extruder positions (vectorized)
        sister_positions_uncoupled = self.sister_positions[uncoupled_sisters]
        
        # Create broadcasting arrays to compare all combinations at once
        # sister_pos_broadcast: (n_uncoupled_sisters, 1)
        # extruder_pos_broadcast: (1, n_active_extruders)
        sister_pos_broadcast = sister_positions_uncoupled[:, None]
        
        # Check matches for left legs
        matches_leg1 = sister_pos_broadcast == extruder_pos1[None, :]  # (n_sisters, n_extruders)
        # Check matches for right legs  
        matches_leg2 = sister_pos_broadcast == extruder_pos2[None, :]  # (n_sisters, n_extruders)
        
        # Combined matches (either leg)
        any_matches = matches_leg1 | matches_leg2  # (n_sisters, n_extruders)
        
        # Find which sisters have matches
        sister_indices, extruder_indices = self.xp.where(any_matches)
        
        # Process matches (only take first match per sister to avoid multiple couplings)
        processed_sisters = set()
        
        for i in range(len(sister_indices)):
            sister_idx = sister_indices[i]
            sister_id = uncoupled_sisters[sister_idx]
            
            # Skip if this sister is already processed (multiple matches)
            if int(sister_id) in processed_sisters:
                continue
                
            extruder_idx = extruder_indices[i]
            extruder_id = active_extruders[extruder_idx]
            
            # Establish coupling (update arrays and dicts)
            self.sister_coupled_to[sister_id] = extruder_id
            self.extruder_sister_counts[extruder_id] += 1
            
            # Update dictionaries
            sister_id_int = int(sister_id)
            extruder_id_int = int(extruder_id)
            
            self.coupled_to_extruder[sister_id_int] = extruder_id_int
            if extruder_id_int not in self.coupled_to_sister:
                self.coupled_to_sister[extruder_id_int] = []
            self.coupled_to_sister[extruder_id_int].append(sister_id_int)
            
            processed_sisters.add(sister_id_int)
    
    def _uncouple_dead_extruders(self):
        """Vectorized uncoupling of sisters from dead extruders"""
        if self.sister_coupled_to is None:
            return
    
        # Find all coupled sisters
        coupled_mask = self.sister_coupled_to != -1
        if not self.xp.any(coupled_mask):
            return
            
        coupled_sisters = self.xp.where(coupled_mask)[0]
        extruder_ids = self.sister_coupled_to[coupled_sisters]
        
        # Check which extruders are dead (vectorized)
        dead_mask = self.states[extruder_ids] == 0 
        dead_couplings = coupled_sisters[dead_mask]
        
        if len(dead_couplings) > 0:
            # Get extruder IDs before uncoupling
            dead_extruder_ids = self.sister_coupled_to[dead_couplings]
            
            # Update arrays (vectorized)
            self.extruder_sister_counts[dead_extruder_ids] -= 1
            self.sister_coupled_to[dead_couplings] = -1

            # Update dictionaries
            for i, sister_id in enumerate(dead_couplings):
                sister_id_int = int(sister_id)
                extruder_id = int(dead_extruder_ids[i])
                
                # Remove from coupled_to_extruder
                if sister_id_int in self.coupled_to_extruder:
                    del self.coupled_to_extruder[sister_id_int]
                
                # Remove from coupled_to_sister
                if extruder_id in self.coupled_to_sister:
                    if sister_id_int in self.coupled_to_sister[extruder_id]:
                        self.coupled_to_sister[extruder_id].remove(sister_id_int)
                    if not self.coupled_to_sister[extruder_id]:
                        del self.coupled_to_sister[extruder_id]
    
    def birth(self, unbound_state_id):
        """Vectorized birth function for extruders only"""
        free_sites = self.sites[~self.occupied]
        if len(free_sites) == 0:
            return self.xp.array([])
            
        binding_sites = self.xp.random.choice(free_sites, size=self.number, replace=False)
        rng = self.xp.less(self.xp.random.random(self.number), self.birth_prob[binding_sites])
        ids = self.xp.flatnonzero(rng * self.xp.equal(self.states, unbound_state_id))
                
        if len(ids) > 0:
            binding_sites = binding_sites[ids]
            self.positions[ids] = binding_sites[:, None]
        
            rng_dir = self.xp.less(self.xp.random.random(len(ids)), 0.5)
            rng_stagger = self.xp.less(self.xp.random.random(len(ids)), 0.5) * ~self.occupied[binding_sites+1]
            
            self.positions[ids, 1] = self.xp.where(rng_stagger,
                                                   self.positions[ids, 1] + 1,
                                                   self.positions[ids, 1])
            self.directions[ids] = rng_dir.astype(self.xp.uint32)
                                                           
        return ids
    
    def death(self, bound_state_id):
        """Vectorized death function for extruders only"""
        death_prob = self.xp.where(self.stalled,
                                   self.stalled_death_prob[self.positions],
                                   self.death_prob[self.positions])
        death_prob = self.xp.max(death_prob, axis=1)
        
        rng = self.xp.less(self.xp.random.random(self.number), death_prob)
        ids = self.xp.flatnonzero(rng * self.xp.equal(self.states, bound_state_id))
        
        return ids
    
    def unload(self, ids_death):
        """Optimized unload with proper dict cleanup"""
        if len(ids_death) == 0:
            return
        
        # First: Clean up dictionaries for dying extruders
        for extruder_id in ids_death:
            extruder_id_int = int(extruder_id)
            
            # If this extruder was coupled to sisters, uncouple them
            if extruder_id_int in self.coupled_to_sister:
                sister_list = list(self.coupled_to_sister[extruder_id_int]) 
                
                for sister_id in sister_list:
                    # Remove from coupled_to_extruder dict
                    if sister_id in self.coupled_to_extruder:
                        del self.coupled_to_extruder[sister_id]
                    
                    # Update sister array
                    if sister_id < len(self.sister_coupled_to):
                        self.sister_coupled_to[sister_id] = -1
                
                # Remove extruder from coupled_to_sister dict
                del self.coupled_to_sister[extruder_id_int]
        
        # Vectorized cleanup: ensure all sisters coupled to dying extruders are uncoupled
        if self.sister_coupled_to is not None:
            coupled_to_dead = self.xp.isin(self.sister_coupled_to, ids_death)
            if self.xp.any(coupled_to_dead):
                self.sister_coupled_to[coupled_to_dead] = -1
        
        # Reset sister counts for dying extruders
        self.extruder_sister_counts[ids_death] = 0
        
        # Standard unload for extruders
        self.stalled[ids_death] = False
        self.positions[ids_death] = -1

    def update_states(self, unbound_state_id, bound_state_id):
        """Standard state update"""
        ids_birth = self.birth(unbound_state_id)
        ids_death = self.death(bound_state_id)
        
        self.states[ids_birth] = bound_state_id
        self.states[ids_death] = unbound_state_id

        self.unload(ids_death)
    
    def update_occupancies(self):
        """Vectorized occupancy update including sisters"""
        super().update_occupancies()
        
        # Sisters also occupy lattice sites (vectorized)
        if self.sister_positions is not None:
            valid_sisters = self.sister_positions >= 0
            if self.xp.any(valid_sisters):
                valid_positions = self.sister_positions[valid_sisters]
                self.occupied[valid_positions] = True
    
    def get_coupling_status(self):
        """Fast coupling status using vectorized operations"""
        if self.sister_positions is None or self.sister_coupled_to is None:
            return {
                'coupled_pairs': [],
                'uncoupled_sisters': [],
                'total_sisters': self.num_sisters
            }
            
        coupled_mask = self.sister_coupled_to != -1
        coupled_sisters = self.xp.where(coupled_mask)[0]
        uncoupled_sisters = self.xp.where(~coupled_mask)[0]
        
        coupled_pairs = []
        for sister_id in coupled_sisters:
            extruder_id = int(self.sister_coupled_to[sister_id])
            coupled_pairs.append({
                'sister_id': int(sister_id),
                'extruder_id': extruder_id,
                'sister_pos': int(self.sister_positions[sister_id]),
                'extruder_pos1': int(self.positions[extruder_id, 0]),
                'extruder_pos2': int(self.positions[extruder_id, 1])
            })
        
        uncoupled_sisters_list = []
        for sister_id in uncoupled_sisters:
            uncoupled_sisters_list.append({
                'sister_id': int(sister_id),
                'sister_pos': int(self.sister_positions[sister_id])
            })
        
        return {
            'coupled_pairs': coupled_pairs,
            'uncoupled_sisters': uncoupled_sisters_list,
            'total_sisters': self.num_sisters
        }
    
    def step(self, mode, unbound_state_id = 0, bound_state_id = 1, active_state_id = 1, **kwargs):
        """Main step function with sister coupling"""
        # Standard extruder birth/death updates
        self.update_states(unbound_state_id, bound_state_id)
        
        # Check for sister-extruder couplings
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

    