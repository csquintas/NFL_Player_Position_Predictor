
import yaml

class TrainConfig:
    """ Work in progress """
    def __init__(self, config_path: str | None = None):
        
        self.cfg = self.load_config(config_path) if config_path else {}
        self.train_cfg = self.process_config()
    
    def load_config(self, config_path: str):

        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)

        return cfg
    
    def process_config(self):

        if "learning_rate" not in self.cfg:
            self.cfg["learning_rate"] = 0.001
        if "batch_size" not in self.cfg:
            self.cfg["batch_size"] = 32
        if "num_epochs" not in self.cfg:
            self.cfg["num_epochs"] = 10
        if "device" not in self.cfg:
            self.cfg["device"] = "cpu"
        
        return self.cfg