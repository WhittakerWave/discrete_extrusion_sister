import warnings

from .SymmetricEngines import _symmetric_step_cpu, _symmetric_step_gpu

try:
    import cupy as cp
    symmetric_step_cuda = cp.RawKernel(_symmetric_step_gpu(), '_symmetric_step_gpu')

except:
    pass
    
try:
    import numba as nb
    use_numba = True

    symmetric_step_numba = nb.njit(fastmath=True)(_symmetric_step_cpu)

except ImportError:
    use_numba = False

    symmetric_step = _symmetric_step_cpu

	
def SteppingEngine(sim, mode, active_state_id, threads_per_block=256, **kwargs):

	if mode == "symmetric":
		rngs = sim.xp.random.random((sim.number, 4)).astype(sim.xp.float32)
		
		args = tuple([active_state_id,
			          rngs,
			          sim.number,
			          0, sim.lattice_size,
			          sim.states,
			          sim.occupied,
			          sim.barrier_engine.stall_left,
			          sim.barrier_engine.stall_right,
			          sim.pause_prob,
			          sim.positions,
			          sim.stalled])
		
		if sim.xp.__name__ == 'cupy':
			num_blocks = (sim.number+threads_per_block-1) // threads_per_block
				
			warnings.warn("Running lattice extrusion on the GPU")
			symmetric_step_cuda((num_blocks,), (threads_per_block,), args)
				
		elif sim.xp.__name__ == 'numpy':
			if use_numba:
				warnings.warn("Running lattice extrusion on the CPU using Numba")
				symmetric_step_numba(*args)

			else:
				warnings.warn("Running lattice extrusion on the CPU using pure Python backend")
				symmetric_step(*args)
	
	else:
		raise RuntimeError("Unsupported mode '%s'" % mode)
