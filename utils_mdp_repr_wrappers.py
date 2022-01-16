from dacmdp.core.args import BaseConfig as MDPBaseConfig
from dacmdp.core.mdp_agents.disc_agent import DACAgentBase 
from dacmdp.core.mdp_agents.cont_agent import DACAgentContNNBaseline, DACAgentContNNBDeltaPred,  DACAgentThetaDynamics, DACAgentContNNBSARepr
from dacmdp.core.mdp_agents.pi_agent import DACAgentThetaDynamicsBiasPi, DACAgentThetaDynamicsPlusPi,  DACAgentThetaDynamicsPlusPiWithOODEval
from dacmdp.core.repr_nets import DummyNet, LatentDynamicsNet, LatentPolicyNetObs, LatentPolicyNetState
from orepr.td3_wrapper import load_td3_repr

def get_agent_model_class(config, dac_build):
    """
    Holds the map of dac_build and dac agent class. 
    Returns the correct DACMDP Agent Class 
    input: config - current context and arguments. 
    input: dac_build - name of the dac build for the dac agent class.  
    """
    if dac_build == "DACAgentBase": return DACAgentBase;
    elif dac_build == "DACAgentContNNBaseline": return DACAgentContNNBaseline;
    elif dac_build == "DACAgentContNNBDeltaPred": return DACAgentContNNBDeltaPred;
    else: assert False, "Agent Model Not Found"
    

# elif dac_build == "StochasticAgent": return DACAgent;
# elif dac_build == "StochasticAgent_o": return DACAgent;
# elif dac_build == "StochasticAgent_s": return DACAgent;
# elif dac_build == "StochasticAgent_sa": return DACAgentSARepr;
# elif dac_build == "StochasticAgentWithDelta_o": return DACAgentDelta;
# elif dac_build == "StochasticAgentWithDelta_s": return DACAgentDelta;
# elif dac_build == "StochasticAgentWithParametricPredFxn_o": return DACAgentThetaDynamics;
# elif dac_build == "StochasticAgentWithParametricPredFxn_s": return DACAgentThetaDynamics;
# elif dac_build == "StchExtendedAgent_o": return DACAgentThetaDynamicsPlusPi;
# elif dac_build == "StchExtendedAgent_s": return DACAgentThetaDynamicsPlusPi;
# elif dac_build == "StchExtendedAgentSafeBase_o":return DACAgentThetaDynamicsPlusPiWithOODEval;
# elif dac_build == "StchExtendedAgentSafeBase_s":return DACAgentThetaDynamicsPlusPiWithOODEval;
# elif dac_build == "PIAgent_o" : return DACAgentThetaDynamicsBiasPi;
# elif dac_build == "PIAgent_s" : return DACAgentThetaDynamicsBiasPi;


    
def get_repr_model(config, repr_build):
    if repr_build == "identity": return DummyNet();
    elif repr_build == "td3_bc": 
        return  load_td3_repr(config.envArgs.env_name,
                               config.envArgs.seed, 
                               config.reprArgs.repr_save_dir, device = "cpu")
    else: 
        print("Agent Class Not Found","returning dummy representation")
        assert False, "Agent representation Not Found";

        
# elif repr_build == "DeterministicAgent_o": return DummyNet();
# elif repr_build == "DeterministicAgent_s": return LatentDynamicsNet(config.reprArgs.dynamics_model, config.dataArgs.buffer_device);
# elif repr_build == "StochasticAgent": return DummyNet();
# elif repr_build == "StochasticAgent_o": return DummyNet();
# elif repr_build == "StochasticAgent_s": return LatentDynamicsNet(config.reprArgs.dynamics_model, config.dataArgs.buffer_device);
# elif repr_build == "StochasticAgent_sa": return LatentDynamicsNet(config.reprArgs.dynamics_model, config.dataArgs.buffer_device);
# elif repr_build == "StochasticAgentWithDelta_o": return DummyNet();
# elif repr_build == "StochasticAgentWithDelta_s": return LatentDynamicsNet(config.reprArgs.dynamics_model, config.dataArgs.buffer_device);
# elif repr_build == "StochasticAgentWithParametricPredFxn_o": return LatentPolicyNetObs(config.reprArgs.dynamics_model, config.reprArgs.policy_model, config.dataArgs.buffer_device);
# elif repr_build == "StochasticAgentWithParametricPredFxn_s": return LatentPolicyNetState(config.reprArgs.dynamics_model, config.reprArgs.policy_model, config.dataArgs.buffer_device);
# elif repr_build == "StchExtendedAgent_o": return LatentPolicyNetObs(config.reprArgs.dynamics_model, config.reprArgs.policy_model, config.dataArgs.buffer_device);
# elif repr_build == "StchExtendedAgent_s": return LatentPolicyNetState(config.reprArgs.dynamics_model, config.reprArgs.policy_model, config.dataArgs.buffer_device);
# elif repr_build == "StchExtendedAgentSafeBase_o":return LatentPolicyNetObs(config.reprArgs.dynamics_model, config.reprArgs.policy_model, config.dataArgs.buffer_device);
# elif repr_build == "StchExtendedAgentSafeBase_s":return LatentPolicyNetState(config.reprArgs.dynamics_model, config.reprArgs.policy_model, config.dataArgs.buffer_device);
# elif repr_build == "PIAgent_o" : return LatentPolicyNetObs(config.reprArgs.dynamics_model, config.reprArgs.policy_model, config.dataArgs.buffer_device);
# elif repr_build == "PIAgent_s" : return LatentPolicyNetState(config.reprArgs.dynamics_model, config.reprArgs.policy_model, config.dataArgs.buffer_device);