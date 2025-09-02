

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
        # This prevents AttributeError in update_occupancies()
        self.sister_positions = None
        self.sister_coupled_to = None       # Array: sister_id -> extruder_id (-1 if uncoupled)
        self.extruder_sister_counts = None  # Array: extruder_id -> number of coupled sisters
        
        # Maintain backward compatibility with dictionary format
        self.coupled_to_extruder = {}       # sister_id -> extruder_id [many:1]
        self.coupled_to_sister = {}         # extruder_id -> [sister_ids] [1:many]
        
        # Pre-computed position lookup for faster coupling checks
        self._position_to_extruders = {}    # Cache for position -> extruder mapping
        self._position_cache_valid = False

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
        """Optimized sister initialization using vectorized operations"""
        if self.num_sisters <= 0:
            print("No sisters to initialize")
            return
        
        lattice_size = len(self.occupied)
        print(f"Initializing {self.num_sisters} sisters on lattice of size {lattice_size}")

        # Initialize arrays
        self.sister_positions = self.xp.zeros(self.num_sisters, dtype=self.xp.int32)
        self.sister_coupled_to = self.xp.full(self.num_sisters, -1, dtype=self.xp.int32)
        self.extruder_sister_counts = self.xp.zeros(self.number, dtype=self.xp.int32)

        # Vectorized initialization: find all free positions at once
        free_positions = self.xp.where(~self.occupied)[0]
        
        if len(free_positions) >= self.num_sisters:
            # Randomly select from free positions
            selected_positions = self.xp.random.choice(
                free_positions, size=self.num_sisters, replace=False
            )
            self.sister_positions = selected_positions
        else:
            print(f"WARNING: Not enough free positions ({len(free_positions)}) for {self.num_sisters} sisters")
            # Place as many as possible
            num_to_place = min(self.num_sisters, len(free_positions))
            self.sister_positions[:num_to_place] = free_positions[:num_to_place]
            # Mark excess sisters as inactive
            self.sister_positions[num_to_place:] = -1
            
        print(f"Initialized {self.num_sisters} sisters at positions: {self.sister_positions}")
    
    def _update_position_cache(self):
        ### Build the extruder two leg postions -- id map 
        ### Position to extruder lef and right legs (both are indexed by the extruder ID)
        print(f"=== DEBUG _update_position_cache ===")
        self._position_to_extruders = {}
        active_extruders = []
        for extruder_id in range(self.number):
            if self.states[extruder_id] != 0:  # Extruder is on chromosome -- active 
                active_extruders.append(extruder_id)
                pos1 = int(self.positions[extruder_id, 0])
                pos2 = int(self.positions[extruder_id, 1])
                print(f"Active extruder {extruder_id}: positions ({pos1}, {pos2})")
                # Add both leg positions
                if pos1 not in self._position_to_extruders:
                    self._position_to_extruders[pos1] = []
                if pos2 not in self._position_to_extruders:
                    self._position_to_extruders[pos2] = []

                self._position_to_extruders[pos1].append(extruder_id)
                self._position_to_extruders[pos2].append(extruder_id)
    
        print(f"Total active extruders: {len(active_extruders)}")
        print(f"Position cache built: {self._position_to_extruders}")
        self._position_cache_valid = True
        print("=== END DEBUG ===\n")
   
    def _sync_coupling_dicts(self):
        """Sync array-based coupling data to dictionary format for backward compatibility"""
        self.coupled_to_extruder.clear()
        self.coupled_to_sister.clear()
        
        # Only sync if sisters are initialized
        if self.sister_coupled_to is not None:
            coupled_mask = self.sister_coupled_to != -1
            coupled_sisters = self.xp.where(coupled_mask)[0]
            
            for sister_id in coupled_sisters:
                extruder_id = int(self.sister_coupled_to[sister_id])
                self.coupled_to_extruder[int(sister_id)] = extruder_id
                
                if extruder_id not in self.coupled_to_sister:
                    self.coupled_to_sister[extruder_id] = []
                self.coupled_to_sister[extruder_id].append(int(sister_id))

    def check_sister_coupling(self):
        """Optimized coupling check with immediate dict sync and safe caching"""
        print(f"=== DEBUG check_sister_coupling ===")
        
        if self.sister_positions is None or self.sister_coupled_to is None:
            return
        
        print(f"Sister positions: {self.sister_positions}")
        print(f"Sister coupled_to before: {self.sister_coupled_to}")
        print(f"Coupled_to_extruder dict before: {self.coupled_to_extruder}")
        ## Update position cache for extruders if invalid
        # if not self._position_cache_valid:
        self._update_position_cache()
        print(f"After _update_position_cache: {len(self._position_to_extruders)} positions")
        print(f"Cache contents: {self._position_to_extruders}")
        ## First pass: uncouple sisters whose extruders died
        self._uncouple_dead_extruders()
        ## Second pass: find new couplings
        uncoupled_sisters = self.xp.where(self.sister_coupled_to == -1)[0]
        print(f"Before sister loop: {len(self._position_to_extruders)} positions")
        
        for sister_id in uncoupled_sisters:
            sister_pos = int(self.sister_positions[sister_id])
            print(f"Checking sister {sister_id} at position {sister_pos}")

            if sister_pos in self._position_to_extruders:
                extruder_id = self._position_to_extruders[sister_pos][0]
                print(f"  -> Found extruder {extruder_id} at sister position {sister_pos}")
                ## update the array
                self.sister_coupled_to[sister_id] = extruder_id
                self.extruder_sister_counts[extruder_id] += 1
                print(f"  -> Updated arrays: sister_coupled_to[{sister_id}] = {extruder_id}")

                ## update dictionaries too
                self.coupled_to_extruder[int(sister_id)] = extruder_id
                if extruder_id not in self.coupled_to_sister:
                    self.coupled_to_sister[extruder_id] = []
                self.coupled_to_sister[extruder_id].append(int(sister_id))
                print(f"  -> Updated dicts: coupled_to_extruder = {self.coupled_to_extruder}")
                print(f"  -> Updated dicts: coupled_to_sister = {self.coupled_to_sister}")
            else:
                print(f"  -> No extruder found at position {sister_pos}")

        print(f"Final coupled_to_extruder: {self.coupled_to_extruder}")
        print(f"Final coupled_to_sister: {self.coupled_to_sister}")
        print("=== END DEBUG ===\n")
        # ✅ Keep dicts consistent immediately
        # self._sync_coupling_dicts()

    def _uncouple_dead_extruders(self):
        """Vectorized uncoupling of sisters from dead extruders"""
        # Safety check
        if self.sister_coupled_to is None:
            return
    
        coupled_mask = self.sister_coupled_to != -1
        if not self.xp.any(coupled_mask):
            return
            
        coupled_sisters = self.xp.where(coupled_mask)[0]
        extruder_ids = self.sister_coupled_to[coupled_sisters]
        
        # Check which extruders are dead (vectorized), if 0 dead, if 1 is active 
        dead_mask = self.states[extruder_ids] == 0 
        dead_couplings = coupled_sisters[dead_mask]
        
        if len(dead_couplings) > 0:
            # Uncouple dead extruders (vectorized)
            dead_extruder_ids = self.sister_coupled_to[dead_couplings]
            self.extruder_sister_counts[dead_extruder_ids] -= 1
            self.sister_coupled_to[dead_couplings] = -1

            for sister_id in dead_couplings:
                sister_id_int = int(sister_id)
                extruder_id = int(dead_extruder_ids[sister_id == dead_couplings][0])
                
                # Remove from dicts
                if sister_id_int in self.coupled_to_extruder:
                    del self.coupled_to_extruder[sister_id_int]
                if extruder_id in self.coupled_to_sister:
                    if sister_id_int in self.coupled_to_sister[extruder_id]:
                        self.coupled_to_sister[extruder_id].remove(sister_id_int)
                    if not self.coupled_to_sister[extruder_id]:
                        del self.coupled_to_sister[extruder_id]
    
    def birth(self, unbound_state_id):
        """Optimized birth function - unchanged but benefits from improved coupling"""
        self._position_cache_valid = False  # Invalidate cache
        
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
        """Optimized death function"""
        death_prob = self.xp.where(self.stalled,
                                   self.stalled_death_prob[self.positions],
                                   self.death_prob[self.positions])
        death_prob = self.xp.max(death_prob, axis=1)
        
        rng = self.xp.less(self.xp.random.random(self.number), death_prob)
        ids = self.xp.flatnonzero(rng * self.xp.equal(self.states, bound_state_id))
        
        return ids
    
    def unload(self, ids_death):
        """Optimized unload using vectorized operations"""
        if len(ids_death) == 0:
            return
        # Vectorized uncoupling: find all sisters coupled to dying extruders
        coupled_sisters_death = self.xp.where(
            self.xp.isin(self.sister_coupled_to, ids_death))[0]
        
        if len(coupled_sisters_death) > 0:
            # Reset sister coupling status to -1 for unload coupled sisters
            self.sister_coupled_to[coupled_sisters_death] = -1
        # Reset sister counts for dying extruders
        self.extruder_sister_counts[ids_death] = 0
        # Standard unload for extruders
        self.stalled[ids_death] = False
        self.positions[ids_death] = -1
        
        # Invalidate position cache
        # self._position_cache_valid = False

    def update_states(self, unbound_state_id, bound_state_id):
        ids_birth = self.birth(unbound_state_id)
        ids_death = self.death(bound_state_id)
        
        self.states[ids_birth] = bound_state_id
        self.states[ids_death] = unbound_state_id

        self.unload(ids_death)
    
    def update_occupancies(self):
        """Optimized occupancy update with safety checks"""
        super().update_occupancies()
        # Only update sister occupancies if sisters are initialized
        if self.sister_positions is not None:
            # Vectorized sister occupancy update
            valid_sisters = self.sister_positions >= 0
            if self.xp.any(valid_sisters):
                valid_positions = self.sister_positions[valid_sisters]
                self.occupied[valid_positions] = True
    
    def get_coupling_status(self):
        """Optimized coupling status using vectorized operations"""
        # Safety check for initialization
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
        if len(coupled_sisters) > 0:
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
        """Optimized step function"""
        print(f"\n=== DEBUG STEP START ===")
        print(f"Before update_states - Active extruders: {self.xp.sum(self.states != 0)}")
        # Standard extruder birth/death updates
        self.update_states(unbound_state_id, bound_state_id)
        print(f"After update_states - Active extruders: {self.xp.sum(self.states != 0)}")
        
        # Check coupling dictionaries before check_sister_coupling
        print(f"Before check_sister_coupling: coupled_to_extruder = {self.coupled_to_extruder}")
        
        # Optimized sister-extruder coupling checks
        self.check_sister_coupling()  
        
        # Check coupling dictionaries after check_sister_coupling
        print(f"After check_sister_coupling: coupled_to_extruder = {self.coupled_to_extruder}")
    
        # Update occupancies (vectorized)
        self.update_occupancies()
    
        # Parent class step
        super().step(**kwargs)
        
        # Update kwargs with coupling info
        kwargs.update({
            'coupled_to_extruder': self.coupled_to_extruder,
            'coupled_to_sister': self.coupled_to_sister})
    
        print(f"Passing to stepping_engine: coupled_to_extruder = {kwargs['coupled_to_extruder']}")
        print(f"Passing to stepping_engine: coupled_to_sister = {kwargs['coupled_to_sister']}")
    
        # Movement step with coupling information
        self.stepping_engine(self, mode, unbound_state_id, active_state_id, **kwargs)
        print(f"=== DEBUG STEP END ===\n")
    