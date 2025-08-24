
def symmetric_sister_step_cpu(active_state_id,
                        rngs,
                        N,
                        N_min,
                        N_max,
                        states,
                        occupied,
                        stall_left,
                        stall_right,
                        pause_prob,
                        positions,
                        stalled,
                        is_sister=None,
                        sister_coupled=None,
                        coupled_to_extruder=None,
                        coupled_to_sister=None,
                        **kwargs):
    
    # Default values if sister parameters not provided
    if is_sister is None:
        is_sister = np.zeros(N, dtype=bool)
    if sister_coupled is None:
        sister_coupled = np.zeros(N, dtype=bool)
    if coupled_to_extruder is None:
        coupled_to_extruder = {}
    if coupled_to_sister is None:
        coupled_to_sister = {}
    
    # Track which particles have already been processed (to avoid double-processing coupled pairs)
    processed = np.zeros(N, dtype=bool)
    
    for i in range(N):
        if states[i] == active_state_id and not processed[i]:
            
            if is_sister[i]:
                # This is a sister particle
                if sister_coupled[i]:
                    # Sister is coupled to an extruder - they move together
                    extruder_id = coupled_to_extruder[i]
                    process_coupled_movement(i, extruder_id, rngs, positions, stalled, 
                                           occupied, pause_prob, stall_left, stall_right,
                                           is_sister, processed)
                else:
                    # Uncoupled sister don't move 
                    process_sister_movement(i, rngs, positions, stalled, occupied, 
                                          pause_prob, processed)
            
            else:
                # This is an extruder particle
                if i in coupled_to_sister:
                    # Extruder is coupled to a sister - they move together
                    sister_id = coupled_to_sister[i]
                    process_coupled_movement(sister_id, i, rngs, positions, stalled,
                                           occupied, pause_prob, stall_left, stall_right,
                                           is_sister, processed)
                else:
                   # uncoupled extruder move independently 
                    process_extruder_movement(i, rngs, positions, stalled, occupied,
                                            pause_prob, stall_left, stall_right, processed)


def process_extruder_movement(extruder_id, rngs, positions, stalled, occupied, 
                            pause_prob, stall_left, stall_right, processed):
    """Handle movement for uncoupled extruder (two legs) - original logic"""
    cur1 = positions[extruder_id, 0]
    cur2 = positions[extruder_id, 1]
    
    # Check for stalling
    stall1 = stall_left[cur1]
    stall2 = stall_right[cur2]
    
    if rngs[extruder_id, 0] < stall1:
        stalled[extruder_id, 0] = True
    if rngs[extruder_id, 1] < stall2:
        stalled[extruder_id, 1] = True
    
    # Leg 1 movement (leftward)
    if not stalled[extruder_id, 0]:
        if not occupied[cur1 - 1]:
            pause1 = pause_prob[cur1]
            if rngs[extruder_id, 2] > pause1:
                positions[extruder_id, 0] = cur1 - 1
    
    # Leg 2 movement (rightward)
    if not stalled[extruder_id, 1]:
        if not occupied[cur2 + 1]:
            pause2 = pause_prob[cur2]
            if rngs[extruder_id, 3] > pause2:
                positions[extruder_id, 1] = cur2 + 1
    
    processed[extruder_id] = True
    

def process_sister_movement(sister_id, rngs, positions, stalled, occupied, pause_prob, processed):
    """Handle movement for uncoupled sister (not coupled to extruders) - sisters remain fixed when uncoupled"""
    # Uncoupled sisters do NOT move - they remain at their current position
    # This is the key difference: sisters only move when coupled to extruders
    processed[sister_id] = True


def process_coupled_movement(sister_id, extruder_id, rngs, positions, stalled, occupied,
                           pause_prob, stall_left, stall_right, is_sister, processed):
    """Handle coordinated movement for coupled sister-extruder"""
    
    # Get current positions
    sister_pos = positions[sister_id, 0]
    extruder_pos1 = positions[extruder_id, 0]
    extruder_pos2 = positions[extruder_id, 1]
    
    # Check extruder stalling
    stall1 = stall_left[extruder_pos1]
    stall2 = stall_right[extruder_pos2]
    
    if rngs[extruder_id, 0] < stall1:
        stalled[extruder_id, 0] = True
    if rngs[extruder_id, 1] < stall2:
        stalled[extruder_id, 1] = True
    
    # Determine which extruder leg the sister is coupled to
    sister_coupled_to_leg1 = (sister_pos == extruder_pos1)
    sister_coupled_to_leg2 = (sister_pos == extruder_pos2)
    
    # Leg 1 movement (leftward) - includes sister if coupled to leg 1
    if not stalled[extruder_id, 0]:
        target_pos = extruder_pos1 - 1
        if not occupied[target_pos]:
            extruder_wants_move = rngs[extruder_id, 2] > pause_prob[extruder_pos1]
            
            if sister_coupled_to_leg1:
                # Sister must also agree to move
                if extruder_wants_move:
                    positions[extruder_id, 0] = target_pos
                    # Sister moves with leg 1
                    positions[sister_id, 0] = target_pos  
    
    # Leg 2 movement (rightward) - includes sister if coupled to leg 2  
    if not stalled[extruder_id, 1]:
        target_pos = extruder_pos2 + 1
        if not occupied[target_pos]:
            extruder_wants_move = rngs[extruder_id, 3] > pause_prob[extruder_pos2]
            
            if sister_coupled_to_leg2:
                # Sister must also agree to move
                if extruder_wants_move: 
                    positions[extruder_id, 1] = target_pos
                    positions[sister_id, 0] = target_pos  # Sister moves with leg 2
        
    # Mark both as processed
    processed[sister_id] = True
    processed[extruder_id] = True

