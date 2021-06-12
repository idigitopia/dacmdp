from .experiments import *
from .exp_track_helper import Sweep

from munch import Munch



MySweeps = {}
#################################################################


# Maze2d open sweep creation. 
sweep_meta = Munch()
sweep_meta.id =  "maze2d_open_sweep_1"
sweep_meta.description = "Sweeping up Maze2d open for correct hyperparameters, stochastic MDP"
            

sweep_params = Munch()
sweep_params.c_flags = ["Stch"]
sweep_params.env_name = ["maze2d-open-v0"]
sweep_params.tt_count = ["-tt1-","-tt5-","-tt10-","-tt20-"]
sweep_params.p_beta = ["-p0-","-p1-", "-p10-", "-p100-", "-p1000-"]
sweep_params.ns_count = ["-ns1", "-ns5", "-ns10", "-ns20"]

maze2dsweep = Sweep(id = sweep_meta.id, 
                    sweep_params = sweep_params, 
                    metadata = sweep_meta)

for exp in ExpPool.expPool:
    if all([ any([a in exp for a in flags]) for flags in sweep_params.values()]):
        maze2dsweep.exp_ids.append(exp)

MySweeps[maze2dsweep.id] = maze2dsweep
