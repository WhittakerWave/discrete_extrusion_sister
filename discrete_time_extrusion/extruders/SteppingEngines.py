def _symmetric_step(active_state_id,
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
                    stalled):
					
	for i in range(N):
		if states[i] == active_state_id:
			cur1 = positions[i, 0]
			cur2 = positions[i, 1]
			
			stall1 = stall_left[cur1]
			stall2 = stall_right[cur2]
									
			if rngs[i, 0] < stall1:
				stalled[i, 0] = True
				
			if rngs[i, 1] < stall2:
				stalled[i, 1] = True

			if not stalled[i, 0]:
				if (not occupied[cur1-1] and (not occupied[cur1-2] if cur1>N_min+2 else True)):
					pause1 = pause_prob[cur1]
					
					if rngs[i, 2] > pause1:
						positions[i, 0] = cur1 - 1
						
			if not stalled[i, 1]:
				if (not occupied[cur2+1] and (not occupied[cur2+2] if cur2<N_max-2 else True)):
					pause2 = pause_prob[cur2]
					
					if rngs[i, 3] > pause2:
						positions[i, 1] = cur2 + 1
