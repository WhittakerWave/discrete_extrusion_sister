extern "C"
{
__global__ void _symmetric_step(
        const int active_state_id,
        const float* rngs,
        const unsigned int N,
        const unsigned int N_min,
        const unsigned int N_max,
        const int* states,
        const bool* occupied,
        const float* stall_left,
        const float* stall_right,
        const float* pause_prob,
        int* positions,
        bool* stalled) {

    unsigned int i = (unsigned int) (blockDim.x * blockIdx.x + threadIdx.x);

    if (i >= N)
        return;
        
    if (states[i] != active_state_id)
        return;

    uint2* stall = (uint2*) stalled;
    int2* position = (int2*) positions;
    
    float4* rng = (float4*) rngs;

    unsigned int cur1 = (unsigned int) position[i].x;
    unsigned int cur2 = (unsigned int) position[i].y;

    float stall1 = stall_left[cur1];
    float stall2 = stall_right[cur2];
							
    if (rng[i].w < stall1)
        stall[i].x = 1;
        
    if (rng[i].x < stall2)
        stall[i].y = 1;
				 	
    if (stall[i].x == 0) {
        if ( (!occupied[cur1-1]) && (cur1>N_min+2 ? !occupied[cur1-2] : true) ) {
            float pause1 = pause_prob[cur1];
			
            if (rng[i].y > pause1)
                position[i].x = (int) cur1-1;
        }
    }
				
    if (stall[i].y == 0) {
        if ( (!occupied[cur2+1]) && (cur2<N_max-2 ? !occupied[cur2+2] : true) ) {
            float pause2 = pause_prob[cur2];
			
            if (rng[i].z > pause2)
                position[i].y = (int) cur2+1;
        }
    }
}
}
