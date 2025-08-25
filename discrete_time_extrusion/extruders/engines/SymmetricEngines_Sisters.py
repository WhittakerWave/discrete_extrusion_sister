
import numpy as np

def symmetric_sister_step_cpu(active_state_id,
                              rngs,
                              N,  # Number of extruders
                              N_min,
                              N_max,
                              states,  # Extruder states
                              occupied,
                              stall_left,
                              stall_right,
                              pause_prob,
                              positions,  # Extruder positions
                              stalled,
                              sister_positions = None,  # Separate sister position array
                              num_sisters = 0,
                              coupled_to_extruder = None,
                              coupled_to_sister = None,
                              **kwargs):
    """
    Sister-aware stepping function that handles extruders and sisters separately
    """
    
    # Default values if sister parameters not provided
    if coupled_to_extruder is None:
        coupled_to_extruder = {}
    if coupled_to_sister is None:
        coupled_to_sister = {}
    if sister_positions is None:
        sister_positions = np.array([])
    
    # Track which extruders have been processed
    processed_extruders = np.zeros(N, dtype=bool)
    
    # Process extruders
    for extruder_id in range(N):
        if states[extruder_id] == active_state_id and not processed_extruders[extruder_id]:
            
            if extruder_id in coupled_to_sister:
                # This extruder is coupled to sister(s) - move together
                # Get the sister IDs for this extruder
                sister_ids = coupled_to_sister[extruder_id]
                process_coupled_extruder_movement_SIMPLE(extruder_id, sister_ids, rngs, 
                                                 positions, sister_positions,
                                                 stalled, occupied, pause_prob,
                                                 stall_left, stall_right, 
                                                 processed_extruders)
            else:
                # Uncoupled extruder - move independently
                process_extruder_movement(extruder_id, rngs, positions, stalled, 
                                        occupied, pause_prob, stall_left, 
                                        stall_right, processed_extruders)
                  
    # Note: Uncoupled sisters don't move, so no separate processing needed

# Uncoupled extruder - move independently 
def process_extruder_movement(extruder_id, rngs, positions, stalled, occupied, 
                             pause_prob, stall_left, stall_right, processed_extruders):
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
    
    processed_extruders[extruder_id] = True


#  Alternative even simpler approach to implement sister and extruder coupled movement 
def process_coupled_extruder_movement_SIMPLE(extruder_id, sister_ids, rngs, positions, 
                                            sister_positions, stalled, occupied, pause_prob,
                                            stall_left, stall_right, processed_extruders):
    """Even simpler: Just use normal extruder movement, then carry sisters"""
    
    # Store positions before movement
    old_pos1 = positions[extruder_id, 0]
    old_pos2 = positions[extruder_id, 1]
    
    # Move extruder exactly like an uncoupled extruder
    process_extruder_movement(extruder_id, rngs, positions, stalled, occupied, 
                            pause_prob, stall_left, stall_right, processed_extruders)
    
    # Get positions after movement
    new_pos1 = positions[extruder_id, 0]
    new_pos2 = positions[extruder_id, 1]
    
    # Carry any sisters that were on the moving extruder legs
    for sister_id in sister_ids:
        sister_pos = sister_positions[sister_id]
        # if sister positions was the old position of left leg 
        if sister_pos == old_pos1 and new_pos1 != old_pos1:
            sister_positions[sister_id] = new_pos1  # Sister carried by leg 1
        elif sister_pos == old_pos2 and new_pos2 != old_pos2:
            sister_positions[sister_id] = new_pos2  # Sister carried by leg 2


def process_coupled_extruder_movement_CORRECTED(extruder_id, sister_ids, rngs, positions, 
                                               sister_positions, stalled, occupied, pause_prob,
                                               stall_left, stall_right, processed_extruders):
    """CORRECTED: Handle extruder+sister movement properly"""
    
    cur1 = positions[extruder_id, 0]  # Left leg
    cur2 = positions[extruder_id, 1]  # Right leg
    
    # Check extruder stalling
    stall1 = stall_left[cur1]
    stall2 = stall_right[cur2]
    
    if rngs[extruder_id, 0] < stall1:
        stalled[extruder_id, 0] = True
    if rngs[extruder_id, 1] < stall2:
        stalled[extruder_id, 1] = True
    
    # FIXED: Move each leg ONCE, independently
    
    # === LEG 1 MOVEMENT ===
    leg1_moved = False
    new_pos1 = cur1
    
    if not stalled[extruder_id, 0]:
        target1 = cur1 - 1
        if target1 >= 0 and not occupied[target1]:
            if rngs[extruder_id, 2] > pause_prob[cur1]:
                new_pos1 = target1
                leg1_moved = True
    
    # Update leg 1 position
    positions[extruder_id, 0] = new_pos1
    
    # === LEG 2 MOVEMENT ===  
    leg2_moved = False
    new_pos2 = cur2
    
    if not stalled[extruder_id, 1]:
        target2 = cur2 + 1
        if target2 < len(occupied) and not occupied[target2]:
            if rngs[extruder_id, 3] > pause_prob[cur2]:
                new_pos2 = target2
                leg2_moved = True
    
    # Update leg 2 position
    positions[extruder_id, 1] = new_pos2
    
    # FIXED: Carry sisters AFTER legs have moved
    for sister_id in sister_ids:
        sister_pos = sister_positions[sister_id]
        
        # Carry sister if it was on a leg that moved
        if sister_pos == cur1 and leg1_moved:
            sister_positions[sister_id] = new_pos1
            # print(f"Sister {sister_id} carried by leg 1: {cur1} → {new_pos1}")
        elif sister_pos == cur2 and leg2_moved:
            sister_positions[sister_id] = new_pos2
            # print(f"Sister {sister_id} carried by leg 2: {cur2} → {new_pos2}")
    
    processed_extruders[extruder_id] = True
