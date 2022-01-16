from .cont_agent import *


class DACAgentSADynamicsPlcySw8(DACAgentContBase):
    """
    This class extends the DACAgentContBase such that it can perform policy switching. 
    a list of repr_model shall be passed one of which will be selected for state representation. 
    the candidate actions will be selected from each of the provided repr_model.| repr model will also have a policy embedded inside it. 
    the candidate Dynamics will be approximated using the SA representation given by the same selected repr_model. 
        - The target state will be approximated by the next_state of the nn transition.
        - This means that get_candidate_predictions and get_candidate_predictions_knn_dicts may be combined here. 
    the candidate rewards will be approximated using the SA representation given by the same selected repr_model. 
        - The reward will be approximated by the reward of the nn transition. 
     

    """
    
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)

        # Assert that a list of repr_models are provided. 

        # Assert that each of the repr_model has action selection method implemented.


        # Main Functions | Override to change the nature of the MDP
    def get_candidate_actions(self, parsed_states):
        raise NotImplementedError ("Subclasses should implement")
        """ return candidate actiosn for all parsed_states  | numpy array with shape  [state_count, action_count, action_vec_size]"""
        self.v_print("Getting Candidate Actions [Start]"); st = time.time()

        parsed_s_candidate_actions = []

        batch_iterator = v_iter(iter_batch(range(len(self.cache_buffer)), _batch_size), 
                                          self._verbose,
                                          "Getting Candidate Actions from observations")

        # Populate self.parsed_transitions
        for idxs in batch_iterator:
            o_batch, _, _, _, _ = self.cache_buffer.sample_indices(idxs)
            parsed_s_candidate_actions.extend(self.repr_model.sample_action_batch(batch_ob))

        self.v_print("Getting Candidate Actions [Complete],  Time Elapsed: {} \n".format(time.time() - st))

        return np.array(parsed_s_candidate_actions).astype(np.float32)

    # for sa repreesntation only 
    def get_candidate_predictions(self, parsed_states, candidate_actions):
        """ return the predictions for all candidate actions | numpy array with shape  [state_count, action_count, state_vec_size] """
        raise NotImplementedError ("Subclasses should implement")
        # self.v_print("Getting predictions for given Candidate Actions"); st = time.time()
        # parsed_s_candidate_predictions = [[self._query_ns_from_D(nn_s)  for nn_s,d in self.items4tt(knn_dict)]
        #         for knn_dict in self.parsed_s_nn_dicts]

        # self.v_print("Getting predictions for given Candidate Actions [Complete],  Time Elapsed: {} \n\n".format(time.time() - st))
        # return np.array(parsed_s_candidate_predictions).astype(np.float32)

       
    def get_candidate_predictions_knn_dicts(self, parsed_s_candidate_predictions):
        raise NotImplementedError ("Subclasses should implement")
        # return self.s_kdTree.get_knn_sub_batch(parsed_s_candidate_predictions.reshape(-1, self.state_vec_size),
        #                                 self.build_args.mdp_build_k,
        #                                 batch_size=256, verbose=self.verbose,
        #                                 message="Calculating NN for all predicted states.")


    def get_candidate_rewards(self, parsed_states, candidate_actions):
        """ return the predictions for all candidate actions | numpy array with shape  [state_count, action_count] """
        
        raise NotImplementedError ("Subclasses should implement")

        # parsed_s_candidate_rewards = [[self._query_r_from_D(nn_s)  for nn_s,d in self.items4tt(knn_dict)]
        #         for knn_dict in self.parsed_s_nn_dicts]
        # return np.av_printrray(parsed_s_candidate_rewards).astype(np.float32)


    def get_candidate_actions_dist(self, parsed_states, candidate_actions):
        """ return dists of all candidate actions  | numpy array with shape [state_count, action_count]"""
        raise NotImplementedError ("Subclasses should implement")
        self.v_print("Getting Candidate Action Distances"); st = time.time()
        
        parsed_s_candidate_action_dists = [] 
        candidate_actions = np.array(candidate_actions)

        batch_iterator = v_iter(iter_batch(range(len(self.cache_buffer)), _batch_size), 
                                          self._verbose,
                                          "Getting Candidate Actions from observations")

        # Make a uncertainty Estimating Model 
        # - Get obs action embedding for seen actions 
        # - for caclulating uncertainty on prediction of each candidate action. 

        # Parse observation and candidate actions so that we can query the uncertainty model. 
        # - Might be just concatenation for true uncertainty model. 
        # - for now this will be the observation candidate action representation. 


        # Query for the uncertainty. 
        # - find the distance to nearest state action pair for each 

        # use the action embedding  of seen actioons 
        # to calculating the distance to candidate action. 
        # distance is the proxy to the uncertainty attached with each predciction. 

        # Return the distance. 

        # Populate self.parsed_transitions
        for idxs in batch_iterator:
            o_batch, _, _, _, _ = self.cache_buffer.sample_indices(idxs)
            cand_a_batch = candidate_actions[idxs]
            for a in cand_a_batch: 
                parsed_s_candidate_actions.extend(self.repr_model.sample_action_batch(batch_ob))


        self.v_print("Getting Candidate Action Distances [Complete],  Time Elapsed: {} \n".format(time.time() - st))
        return np.array(parsed_s_candidate_action_dists).astype(np.float32)




class DACAgentThetaDynamicsPlusPi(DACAgentThetaDynamics):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        
    # Main Functions | Override to change the nature of the MDP
    def get_candidate_actions(self, parsed_states):
        """ return candidate actiosn for all parsed_states  | numpy array with shape  [state_count, action_count, action_vec_size]"""
        self.v_print("Getting Candidate Actions [Start]"); st = time.time()
        
        _batch_size = 256
        batch_iterator = verbose_iterator(iterator = iter_batch(range(len(self.cache_buffer)), _batch_size), 
                                          verbose = self._verbose, 
                                         message = "Getting actions using policy model")
        parsed_pred_actions = [self.repr_model.predict_action_batch(self.cache_buffer.sample_indices(idxs)[0])
                                        for idxs in batch_iterator]
        parsed_s_candidate_actions_P = np.expand_dims(np.concatenate(parsed_pred_actions), 1)
        
        
        self.v_print("Combining the action spaces")
        if len(self.tran_types)>1:
            parsed_s_nn_dict_iterator = verbose_iterator(iterator = self.parsed_s_nn_dicts,   verbose = self._verbose, 
                                         message = "Getting actions using the dataset")
            parsed_s_candidate_actions_D = [[self._query_action_from_D(nn_s) for nn_s,d in self.items4tt(knn_dict)][:-1]
                                      for knn_dict in parsed_s_nn_dict_iterator]
        
            parsed_s_candidate_actions = np.concatenate([parsed_s_candidate_actions_D , parsed_s_candidate_actions_P] ,axis = 1) 
        else:
            parsed_s_candidate_actions = np.array(parsed_s_candidate_actions_P)
        
        
        ################ Make KD Tree ###################
        self.v_print("Making new kD tree for new action space")
        self.parsed_actions = np.unique(parsed_s_candidate_actions.reshape(-1,self.action_len),axis=0)
        self.a_kdTree = MyKDTree(self.parsed_actions)
        
        self.v_print("Getting Candidate Actions [Complete],  Time Elapsed: {} \n".format(time.time() - st))
        return np.array(parsed_s_candidate_actions).astype(np.float32)
        
    def get_candidate_actions_dist(self, parsed_states, candidate_actions):
        self.v_print("Getting distanceds for the given candidate actions");
        st = time.time()
        self.v_print("Getting Candidate Action Distances [Complete],  Time Elapsed: {} \n".format(time.time() - st))
        return np.zeros(self.parsed_s_candidate_actions.shape[:-1]).astype(np.float32)

class DACAgentThetaDynamicsPlusPiWithOODEval(DACAgentThetaDynamicsPlusPi):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def base_policy(self, obs):
        base_policy_action = self.repr_model.predict_action_single(obs)
        return np.array(base_policy_action).reshape(-1,)
    
    def test_for_ood(self, obs):
        base_policy_action = self.repr_model.predict_action_single(obs)
        state = self.repr_model.encode_state_single(obs)
        base_policy_ns = self.repr_model.predict_next_state_single(state, base_policy_action)
        nn_hs, eval_dist = list(self.s_kdTree.get_knn(base_policy_ns, k = 1).items())[0]
        self.online_eval_nn_distances.append(eval_dist)
        
        override_flag = eval_dist > self.eval_threshold
        if override_flag:
            self.policy_calls["base_pi"] += 1
        else:
            self.policy_calls["mdp_pi"] += 1
            
        return override_flag
    
    def seed_policies(self, eval_threshold= float("inf")):
        self.online_eval_nn_distances = []
        self.policy_calls = {"base_pi":0, "mdp_pi":0}
        
        nn_dists = [list(nn_d.values())[-1] for nn_d in self.parsed_s_nn_dicts]
        self.eval_threshold = np.quantile(nn_dists, self.eval_args.eval_threshold_quantile)
        
        
        self.policies = {"optimal":  lambda obs: self.base_policy(obs)  if  self.test_for_ood(obs) else self.opt_policy(obs),
                         "random": self.random_policy,
                         "eps_optimal":  lambda obs: self.base_policy(obs)  if  self.test_for_ood(obs) else self.eps_optimal_policy(obs),
                         "safe":  lambda obs: self.base_policy(obs)  if  self.test_for_ood(obs) else self.safe_policy(obs)}
        

class DACAgentThetaDynamicsBiasPi(DACAgentThetaDynamicsPlusPiWithOODEval):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        
    # Main Functions | Override to change the nature of the MDP
    # def get_candidate_actions(self, parsed_states):
    #     """ return candidate actiosn for all parsed_states  | numpy array with shape  [state_count, action_count, action_vec_size]"""
    #     self.v_print("Getting Candidate Actions [Start]"); st = time.time()
    #
    #     _batch_size = 256
    #     batch_iterator = verbose_iterator(iterator = iter_batch(range(len(self.cache_buffer)), _batch_size),
    #                                       verbose = self._verbose,
    #                                      message = "Getting actions using policy model")
    #     parsed_pred_actions = [self.repr_model.predict_action_batch(self.cache_buffer.sample_indices(idxs)[0])
    #                                     for idxs in batch_iterator]
    #     parsed_s_candidate_actions_P = np.expand_dims(np.concatenate(parsed_pred_actions), 1)
    #
    #
    #     self.v_print("Combining the action spaces")
    #     if len(self.tran_types)>1:
    #         parsed_s_nn_dict_iterator = verbose_iterator(iterator = self.parsed_s_nn_dicts,   verbose = self._verbose,
    #                                      message = "Getting actions using the dataset")
    #         parsed_s_candidate_actions_D = [[self._query_action_from_D(nn_s) for nn_s,d in self.items4tt(knn_dict)][:-1]
    #                                   for knn_dict in parsed_s_nn_dict_iterator]
    #
    #         parsed_s_candidate_actions = np.concatenate([parsed_s_candidate_actions_D , parsed_s_candidate_actions_P] ,axis = 1)
    #     else:
    #         parsed_s_candidate_actions = np.array(parsed_s_candidate_actions_P)
    #
    #
    #     ################ Make KD Tree ###################
    #     self.v_print("Making new kD tree for new action space")
    #     self.parsed_actions = np.unique(parsed_s_candidate_actions.reshape(-1,self.action_len),axis=0)
    #     self.a_kdTree = MyKDTree(self.parsed_actions)
    #
    #     self.v_print("Getting Candidate Actions [Complete],  Time Elapsed: {} \n".format(time.time() - st))
    #     return np.array(parsed_s_candidate_actions).astype(np.float32)
    #
    # def get_candidate_actions_dist(self, parsed_states, candidate_actions):
    #     """ return dists of all candidate actions  | numpy array with shape [state_count, action_count]"""
    #     self.v_print("Getting distanceds for the given candidate actions"); st = time.time()
    #     self.v_print("Getting Candidate Action Distances [Complete],  Time Elapsed: {} \n".format(time.time() - st))
    #     return np.zeros(self.parsed_s_candidate_actions.shape[:-1]).astype(np.float32)
    
        
    # Step 3
    def intialize_dac_dynamics(self):
        """ Populates tC and rC based on the parsed transitions """
        
        # HouseKeeping
        self.v_print(f"----  Initializing Stochastic Dynamics  NS Count {self.build_args.MAX_NS_COUNT}----"); st = time.time()
        self.v_print("Step 3 [Populate Dynamics]: Running"); st = time.time()
        
        if self.build_args.mdp_build_k != self.build_args.MAX_NS_COUNT:
            print("Warning Number of available ns slots and mdp build k is not the same. behavior is undefined")
        
        # Declare and Initialize helper variables
        self.stt2a_idx_matrix = np.zeros((len(self.s_kdTree.s2i), self.build_args.tran_type_count)).astype(np.int32)
        self.orig_tD, self.orig_rD = defaultdict(init2zero_def_dict), defaultdict(init2zero_def_dict)
        self.tC = defaultdict(init2zero_def_def_dict)
        self.rC = defaultdict(init2zero_def_def_dict)
        self.cC = defaultdict(init2zero_def_def_dict) 
        
        # seed for end_state transitions
        for tt in self.tran_types:
            self.tC[self.end_state_vector][tt][self.end_state_vector] = 1
            self.rC[self.end_state_vector][tt][self.end_state_vector] = 0
            self.cC[self.end_state_vector][tt][self.end_state_vector] = 0
            
        # activates _query_action_from_D, and _query_ns_from_D
        for s, a, ns, r, d in self.parsed_transitions:
            self.orig_tD[s][a] = ns if not d else self.end_state_vector
            self.orig_rD[s][a] = r 

        
        # calculate k for nearest neighbor lookups and helper functions
        nn_k = max(self.build_args.tran_type_count, self.build_args.mdp_build_k)
        
        # New NN variables
        self.parsed_states = list(zip(*self.parsed_transitions))[0]
        self.parsed_actions = list(zip(*self.parsed_transitions))[1]
        self.parsed_s_nn_dicts = self.s_kdTree.get_knn_sub_batch(self.parsed_states, nn_k, 
                                                                 batch_size = 256, verbose = self.verbose, 
                                                                 message= "NN for all parsed states")
        
        # candidate actions and predictions
        self.parsed_s_candidate_actions = self.get_candidate_actions(self.parsed_states) #  [state_count, action_count, action_vec_size] 
        self.parsed_s_candidate_action_dists = self.get_candidate_actions_dist(self.parsed_states, self.parsed_s_candidate_actions) # [state_count, action_count]
        self.parsed_s_candidate_predictions = self.get_candidate_predictions(self.parsed_states, self.parsed_s_candidate_actions) #  [state_count, action_count, state_vec_size]
        self.parsed_s_candidate_rewards = self.get_candidate_rewards(self.parsed_states, self.parsed_s_candidate_actions) 
        
        # Sanity Check
        assert len(self.parsed_s_candidate_actions[0]) == self.build_args.tran_type_count
        assert len(self.parsed_s_candidate_predictions[0]) == self.build_args.tran_type_count
        assert len(self.parsed_s_candidate_action_dists[0]) == self.build_args.tran_type_count
        assert len(self.parsed_s_candidate_rewards[0]) == self.build_args.tran_type_count
        
        ac_size, acd_size, sp_size = self.parsed_s_candidate_actions.shape, self.parsed_s_candidate_action_dists.shape, self.parsed_s_candidate_predictions.shape
        self.parsed_s_candidate_predictions_knn_dicts = self.s_kdTree.get_knn_sub_batch(self.parsed_s_candidate_predictions.reshape(-1,self.state_vec_size),  self.build_args.mdp_build_k, 
                                                                               batch_size = 256, verbose = self.verbose,
                                                                              message = "NN for all predicted states.")

        
        policy_action_idx = len(self.tran_types)-1
        
        for s_idx, (s, a, ns, r, d) in verbose_iterator(enumerate(self.parsed_transitions),self._verbose, "Calculating DAC Dynamics"):    
            candidate_actions = self.parsed_s_candidate_actions[s_idx]
            candidate_action_dists = self.parsed_s_candidate_action_dists[s_idx]
            candidate_rewards = self.parsed_s_candidate_rewards[s_idx]
            
            pi_ns_idx = s_idx * len(self.tran_types) + policy_action_idx
            policy_action_distance = list(self.parsed_s_candidate_predictions_knn_dicts[pi_ns_idx].values())[0]
            
            for a_idx, (tt, cand_a, cand_d, cand_r) in enumerate(zip(self.tran_types, candidate_actions, candidate_action_dists,candidate_rewards)):
                
                # tt to action map 
                self.stt2a_idx_matrix[self.s_kdTree.s2i[s]][self.tt2i[tt]] = self.a_kdTree.s2i[tuple(cand_a)]
                
                preD_ns_idx = s_idx * len(self.tran_types) + a_idx
                pred_ns_nn_dict = {nn_s: d + cand_d for nn_s,d in self.parsed_s_candidate_predictions_knn_dicts[preD_ns_idx].items()}
                pred_ns_probs = kernel_probs(pred_ns_nn_dict, delta=self.build_args.knn_delta,
                                                norm_by_dist = self.build_args.normalize_by_distance)
                    
                # We are only concerned with transition counts in this phase. 
                # All transition counts will be properly converted to tran prob while inserting in MDP
                # Reward can be a function of distance of the nn of the prediction, or can also be accounted for individually. 
                if a_idx == policy_action_idx:
                    penalty_distance = 0
                    cand_c = reward_logic(penalty_distance, self.build_args.penalty_beta)
                else:
                    penalty_distance = policy_action_distance + list(pred_ns_nn_dict.values())[0]
                    cand_c = reward_logic(penalty_distance, self.build_args.penalty_beta)

                for dist, (pred_ns, prob) in zip(pred_ns_nn_dict.values(), pred_ns_probs.items()):
                    # reward discounted by the distance to state used for tt->a mapping. 
                    self.tC[s][tt][pred_ns] = int(prob*100)
                    self.rC[s][tt][pred_ns] = cand_r*int(prob*100)
                    self.cC[s][tt][pred_ns] = cand_c*int(prob*100)
                    
            
        self.v_print("Step 3 [Populate Dynamics]: Complete,  Time Elapsed: {} \n\n".format(time.time() - st))


class DACAgentThetaDynamicsEvalPi(DACAgentThetaDynamics):
    def __init__(self, *args,**kwargs):
        super().__init__(*args,**kwargs)
        print("In StchPlcyEvalAgent")
        
        # Encoding Functions
        if self.build_args.tran_type_count != 1:
            print("Warining: tran_type_count must be 1 for policy Evaluation")

    # Main Functions | Override to change the nature of the MDP
    def get_candidate_actions(self, parsed_states):
        """ return candidate actiosn for all parsed_states  | numpy array with shape  [state_count, action_count, action_vec_size]"""
        _batch_size = 256
        batch_iterator = verbose_iterator(iterator = iter_batch(range(len(self.cache_buffer)), _batch_size), 
                                          verbose = self._verbose, 
                                         message = "Getting actions using policy model")
        parsed_pred_actions = [self.repr_model.predict_action_batch(self.cache_buffer.sample_indices(idxs)[0])
                                        for idxs in batch_iterator]
        parsed_s_candidate_actions = np.expand_dims(np.concatenate(parsed_pred_actions), 1)
        
        self.parsed_actions = np.unique(parsed_s_candidate_actions.reshape(-1,self.action_len),axis=0)
        self.a_kdTree = MyKDTree(self.parsed_actions)
        
        return parsed_s_candidate_actions
        
    def get_candidate_actions_dist(self, parsed_states, candidate_actions):
        print("Dummy action distance implemented: May need some further consideration")
        return np.zeros(self.parsed_s_candidate_actions.shape[:-1]).astype(np.float32)