import os
import warnings

from . import SteppingEngines

try:
    import cupy as xp
    
    dpath = os.path.dirname(os.path.abspath(__file__))
    use_cuda = xp.cuda.is_available()
    
    if not use_cuda:
        raise ImportError
    
    with open(f'{dpath}/kernels/SteppingEngines.cuh', 'r') as cuda_file:
        cuda_code = cuda_file.read()
        cuda_module = xp.RawModule(code=cuda_code)
    
    _symmetric_step_gpu = cuda_module.get_function('_symmetric_step')

except:
    import numpy as xp
    use_cuda = False
    
try:
    import numba as nb
    use_numba = True

    _symmetric_step_numba = nb.njit(fastmath=True)(SteppingEngines._symmetric_step)

except ImportError:
    use_numba = False
    
	
def SteppingEngine(sim, active_state_id, mode='Symmetric', **kwargs):

	if mode == 'Symmetric':
		rngs = xp.random.random((sim.number, 4)).astype(xp.float32)
		
		args = tuple([active_state_id, rngs,
					  sim.number, 0, sim.lattice_size,
					  sim.states, sim.occupied,
					  sim.barrier_engine.stall_left, sim.barrier_engine.stall_right,
					  sim.pause_prob,
					  sim.positions, sim.stalled])
					  
		if kwargs['backend'] == "Numba":
			if use_cuda:
				raise RuntimeError("Numba mode incompatible with CuPy arrays")
				
			if use_numba:
				warnings.warn("Running lattice extrusion using Numba on the CPU")
				_symmetric_step_numba(*args)
				
			else:
				raise RuntimeError("Could not load Numba library")

		elif kwargs['backend'] == "CuPy":
			if use_cuda:
				threads_per_block = kwargs['threads_per_block']
				num_blocks = (sim.number+threads_per_block-1) // threads_per_block
				
				warnings.warn("Running lattice extrusion on the GPU using %d threads per block" % kwargs['threads_per_block'])
				_symmetric_step_gpu((num_blocks,), (threads_per_block,), args)
										
			else:
				raise RuntimeError("Could not load CuPy library")

		else:
			warnings.warn("Running lattice extrusion with pure Python backend")
			SteppingEngines._symmetric_step(*args)
	
	else:
		raise RuntimeError("Unsupported mode '%s'" % mode)
