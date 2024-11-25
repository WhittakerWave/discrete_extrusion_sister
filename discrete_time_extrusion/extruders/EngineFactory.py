import warnings

from .SymmetricEngines import _symmetric_step_cpu, _symmetric_step_gpu

try:
    import cupy as xp
    use_cuda = xp.cuda.is_available()
    
    if not use_cuda:
        raise ImportError
        
    symmetric_step_gpu = xp.RawKernel(_symmetric_step_gpu(), '_symmetric_step_gpu')

except:
    import numpy as xp
    use_cuda = False
    
try:
    import numba as nb
    use_numba = True

    symmetric_step_numba = nb.njit(fastmath=True)(_symmetric_step_cpu)

except ImportError:
    use_numba = False
    
	
def SteppingEngine(sim, mode, active_state_id, **kwargs):

	if mode == "symmetric":
		rngs = xp.random.random((sim.number, 4)).astype(xp.float32)
		
		args = tuple([active_state_id, rngs,
			      sim.number, 
			      0, sim.lattice_size,
			      sim.states, 
			      sim.occupied,
			      sim.barrier_engine.stall_left, sim.barrier_engine.stall_right,
			      sim.pause_prob,
			      sim.positions, 
			      sim.stalled])
					  
		if kwargs['backend'] == "Numba":
			if use_cuda:
				raise RuntimeError("Numba mode incompatible with CuPy arrays")
				
			if use_numba:
				warnings.warn("Running lattice extrusion using Numba on the CPU")
				symmetric_step_numba(*args)
				
			else:
				raise RuntimeError("Could not load Numba library")

		elif kwargs['backend'] == "CuPy":
			if use_cuda:
				threads_per_block = kwargs['threads_per_block']
				num_blocks = (sim.number+threads_per_block-1) // threads_per_block
				
				warnings.warn("Running lattice extrusion on the GPU")
				symmetric_step_gpu((num_blocks,), (threads_per_block,), args)
										
			else:
				raise RuntimeError("Could not load CuPy library")

		else:
			warnings.warn("Running lattice extrusion with pure Python backend")
			_symmetric_step_cpu(*args)
	
	else:
		raise RuntimeError("Unsupported mode '%s'" % mode)
