from omegaconf import DictConfig
from model import Net, test
import torch
from typing import OrderedDict
from typing import List, Tuple
from flwr.common import Metrics

def get_fit_config(config: DictConfig):
    def fit_config_fn(server_round: int):
        return{"lr":config.lr, "momentum":config.momentum, 
               "local_epochs":config.local_epochs}
    
    return fit_config_fn
    
def get_evaluate_fn(num_classes: int, vallloader):
    def evaluate_fn(server_round:int, parameters, config):

        model = Net(num_classes)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        param_dict = zip(model.state_dict().keys(), parameters)

        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in param_dict})

        # load_state_dict() function takes a dictionary object, NOT a path to a saved object.
        # used to load parameters into a model
        model.load_state_dict(state_dict, strict=True)

        loss, accuracy = test(model, vallloader, device)

        return loss, {"server_accuracy": accuracy}
    
    return evaluate_fn

def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    return {"weighted_server_accuracy": sum(accuracies) / sum(examples)}

