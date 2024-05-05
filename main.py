import pickle
import hydra
from pathlib import Path
import flwr as fl
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from dataset import dataset_preprocessing
from client import get_client_fn
from server import get_fit_config, get_evaluate_fn, weighted_average

@hydra.main(config_path="conf", config_name="hyperparameters", 
            version_base=None)
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    save_path = HydraConfig.get().runtime.output_dir


    trainloaders, validationloaders, testloaders = dataset_preprocessing(
       cfg.num_clients, cfg.batch_size
    )

    print(len(trainloaders), len(trainloaders[0].dataset), 
          len(validationloaders[0].dataset))
    
    
    client_fn = get_client_fn(trainloaders, testloaders, cfg.num_classes)


    fedlearn_strategy = fl.server.strategy.FedAvg(fraction_fit=1.0,
                                                  min_fit_clients=cfg.num_clients_per_round_fit,
                                                  fraction_evaluate=1.0,
                                                  min_evaluate_clients=cfg.num_clients_per_round_eval,
                                                  min_available_clients=cfg.num_clients,
                                                  on_fit_config_fn=get_fit_config(cfg.config_fit),
                                                  evaluate_metrics_aggregation_fn=weighted_average,
                                                  evaluate_fn=get_evaluate_fn(cfg.num_classes,
                                                                              validationloaders),)
    
    history = fl.simulation.start_simulation(
       client_fn = client_fn,
       num_clients = cfg.num_clients,
       config = fl.server.ServerConfig(num_rounds=cfg.num_rounds),
       strategy = fedlearn_strategy,
       client_resources={
            "num_cpus": 2,
            "num_gpus": 0.0,
            },  
    )

    # optional stuff

    results_path = Path(save_path) / "results.pkl"


    results = {"history": history, "anythingelse": "here"}



    with open(str(results_path), "wb") as h:
       pickle.dump(results, h, protocol=pickle.HIGHEST_PROTOCOL)





if __name__ == "__main__":
  main()