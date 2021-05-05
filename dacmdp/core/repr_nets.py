import torch
import numpy as np
class DummyNet():
    def __init__(self, dummy_params = None):
        self.dummy_params = None
        self.encode_state_single = lambda o: self.encode_state_batch([o])[0]
        self.encode_action_single = lambda a: self.encode_action_batch([a])[0]
        self.predict_action_single = lambda s: self.predict_action_batch([s])[0]
        self.predict_next_state_single = lambda s,a: self.predict_next_state_batch([s],[a])[0]
    

    def encode_state_batch(self, o_batch):
        o_batch = o_batch.cpu().numpy() if torch.is_tensor(o_batch) else o_batch
        return [tuple(np.array(o).astype(np.float32)) for o in o_batch]
    
    def encode_action_batch(self,a_batch):
        a_batch = a_batch.cpu().numpy() if torch.is_tensor(a_batch) else a_batch
        return [tuple(np.array(a).astype(np.float32)) for a in a_batch]

    def predict_action_batch(self, o_batch):
        assert False, "Not Implemented Error"
        
    def predict_next_state_batch(self, s_batch, a_btach):
        assert False, "Not Implemented Error"

class LatentDynamicsNet(DummyNet):
    def __init__(self, dynamics_model, device, *args, **kwargs):
        super().__init__(*args,**kwargs)
        
        self.dynamics_model = dynamics_model.get_dynamics(0).to(device)
        self.device = device

    def encode_state_batch(self, o_batch):
        o_batch = (o_batch if torch.is_tensor(o_batch) else torch.FloatTensor(o_batch)).to(self.device)
        latent_batch = self.dynamics_model.encoder(o_batch).detach().cpu().numpy().astype(np.float32)
        return [tuple(l) for l in latent_batch]

    def encode_action_batch(self, a_batch):
        assert False, "Not Implemented Error"
        return [tuple(a) for a in a_batch]
    
class LatentPolicyNet(DummyNet):
    def __init__(self, dynamics_model,policy_model, device, *args, **kwargs):
        super().__init__(*args,**kwargs)
        
        self.dynamics_model = dynamics_model.to(device)
        self.policy_model = policy_model.to("cpu")
        self.device = device

    # Encoding Functions. 
    def encode_state_batch(self, o_batch):
        o_batch = o_batch.to(self.device) if torch.is_tensor(o_batch) else torch.FloatTensor(o_batch).to(self.device)
        latent_batch = self.dynamics_model.encoder(o_batch).detach().cpu().numpy().astype(np.float32)
        return [tuple(l) for l in latent_batch]
    
    # Prediction candidate action for a state
    def predict_action_batch(self, s_batch):
        s_batch = s_batch.to("cpu") if torch.is_tensor(s_batch) else torch.FloatTensor(s_batch).to("cpu") 
        action_dist = self.policy_model.actor(s_batch)
        a_batch = action_dist.mean.data.detach().cpu().numpy().astype(np.float32)
        return [tuple(a) for a in a_batch]
        
    def predict_next_state_batch(self, s_batch, a_batch):
        s_batch = s_batch.to(self.device) if torch.is_tensor(s_batch) else torch.FloatTensor(s_batch).to(self.device)
        a_batch = a_batch.to(self.device) if torch.is_tensor(a_batch) else torch.FloatTensor(a_batch).to(self.device)
        ns_batch = self.dynamics_model.transition.rnn(a_batch, s_batch).detach().cpu().numpy().astype(np.float32)
        return [tuple(s) for s in ns_batch]
        
class BCQEncoderNet():
    def __init__(self, bcq_net, device):
        self.bcq_net = bcq_net.to(device)
        self.device = device

    def encode_state_single(self, o):
        o = o if torch.is_tensor(o) else torch.FloatTensor([o])
        o = o if o.device == self.device else o.to(self.device)
        latent = self.dynamics_model.encoder(o).reshape(-1).detach().cpu().numpy().astype(np.float32)
        return tuple(latent)

    def encode_state_batch(self, o_batch):
        o_batch = o_batch if torch.is_tensor(o_batch) else torch.FloatTensor(o_batch).to(self.device)
        latent_batch = self.dynamics_model.encoder(o_batch).detach().cpu().numpy().astype(np.float32)
        return [tuple(l) for l in latent_batch]

    def encode_action_single(self, a):
        raise "Non Implemented Error"
        return tuple(np.array(a).astype(np.float32))

    def encode_action_batch(self, a_batch):
        a_batch = a_batch if torch.is_tensor(a_batch) else torch.FloatTensor(a_batch)
        a_batch = a_batch.cpu().numpy().astype(np.float32)
        return [tuple(a) for a in a_batch]

    def predict_single_transition(self, o, a):
        assert False, "Not Implemented Error"

    def predict_batch_transition(self, o_batch, a_batch):
        assert False, "Not Implemented Error"
