
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
                              sister_positions=None,  # Separate sister position array
                              num_sisters=0,
                              coupled_to_extruder=None,
                              coupled_to_sister=None,
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
                sister_ids = coupled_to_sister[extruder_id]
                process_coupled_extruder_movement(extruder_id, sister_ids, rngs, 
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


def process_coupled_extruder_movement(extruder_id, sister_ids, rngs, positions, 
                                     sister_positions, stalled, occupied, pause_prob,
                                     stall_left, stall_right, processed_extruders):
    """Handle coordinated movement for extruder coupled to sister(s)"""
    
    cur1 = positions[extruder_id, 0]
    cur2 = positions[extruder_id, 1]
    
    # Check extruder stalling
    stall1 = stall_left[cur1]
    stall2 = stall_right[cur2]
    
    if rngs[extruder_id, 0] < stall1:
        stalled[extruder_id, 0] = True
    if rngs[extruder_id, 1] < stall2:
        stalled[extruder_id, 1] = True
    
    # Process each coupled sister
    for sister_id in sister_ids:
        sister_pos = sister_positions[sister_id]
        
        # Determine which extruder leg the sister is coupled to
        sister_coupled_to_leg1 = (sister_pos == cur1)
        sister_coupled_to_leg2 = (sister_pos == cur2)
        
        # Leg 1 movement (leftward) - includes sister if coupled to leg 1
        if not stalled[extruder_id, 0] and sister_coupled_to_leg1:
            target_pos = cur1 - 1
            if not occupied[target_pos]:
                if rngs[extruder_id, 2] > pause_prob[cur1]:
                    positions[extruder_id, 0] = target_pos
                    sister_positions[sister_id] = target_pos  # Sister moves with leg 1
        
        # Leg 2 movement (rightward) - includes sister if coupled to leg 2
        elif not stalled[extruder_id, 1] and sister_coupled_to_leg2:
            target_pos = cur2 + 1
            if not occupied[target_pos]:
                if rngs[extruder_id, 3] > pause_prob[cur2]:
                    positions[extruder_id, 1] = target_pos
                    sister_positions[sister_id] = target_pos  # Sister moves with leg 2
    
    # Handle uncoupled leg movement (if extruder has legs not coupled to sisters)
    # Leg 1 movement (if no sister coupled to leg 1)
    if not stalled[extruder_id, 0]:
        leg1_has_sister = any(sister_positions[sid] == cur1 for sid in sister_ids)
        if not leg1_has_sister:
            target_pos = cur1 - 1
            if not occupied[target_pos]:
                if rngs[extruder_id, 2] > pause_prob[cur1]:
                    positions[extruder_id, 0] = target_pos
    
    # Leg 2 movement (if no sister coupled to leg 2)
    if not stalled[extruder_id, 1]:
        leg2_has_sister = any(sister_positions[sid] == cur2 for sid in sister_ids)
        if not leg2_has_sister:
            target_pos = cur2 + 1
            if not occupied[target_pos]:
                if rngs[extruder_id, 3] > pause_prob[cur2]:
                    positions[extruder_id, 1] = target_pos
    
    processed_extruders[extruder_id] = True


