import flwr as fl
import torch
from collections import OrderedDict
from typing import Dict, Tuple
from flwr.common import NDArrays, Scalar
from model import Net, train, test

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, trainloader, valloader, num_classes) -> None:
        super().__init__()

        self.trainloader = trainloader
        self.valloader = valloader

        #pass in parameters for detection here

        self.model = Net(num_classes)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    def set_parameters(self, parameters):
        """Receive parameters and apply them to the local model."""
        # state_dict() returns current state of the model
        param_dict = zip(self.model.state_dict().keys(), parameters)

        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in param_dict})

        # load_state_dict() function takes a dictionary object, NOT a path to a saved object.
        # used to load parameters into a model
        self.model.load_state_dict(state_dict, strict=True)

    def get_parameters(self, config: Dict[str, Scalar]):
        """Extract model parameters and return them as a list of numpy arrays."""
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]


    def fit(self,parameters, config):
        """Train model received by the server (parameters) using the data.

        that belongs to this client. Then, send it back to the server.
        """
        self.set_parameters(parameters)
        lr = config["lr"]
        momentum = config["momentum"]
        epochs = config["local_epochs"]

        optim = torch.optim.SGD(self.model.parameters(), lr=lr, momentum=momentum)

        train(self.model, self.trainloader,optim, epochs, self.device)

        #check in this step if there are any concept drfit
        #if detected reset the model and dont send update to server

        return self.get_parameters({}), len(self.trainloader), {}
    
    def evaluate(self, parameters:NDArrays,config: Dict[str , Scalar]):
        
        self.set_parameters(parameters)

        loss, accuracy =  test(self.model, self.valloader, self.device)

        return float(loss), len(self.valloader), {'accurcay':accuracy}
    

def get_client_fn (trainloaders, valloaders, num_classes):
    def client_fn(cid: str):
        return FlowerClient(trainloader=trainloaders[int(cid)],
                            valloader=valloaders[int(cid)],
                            num_classes= num_classes,).to_client()
    return client_fn