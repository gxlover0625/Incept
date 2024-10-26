from Incept.utils import seed_everything

class ExpManager:
    def __init__(self, config, trainer, train_dataset, val_dataset = None):
        self.config = config
        self.trainer = trainer
        self.train_dataset = train_dataset
        if val_dataset is not None:
            self.val_dataset = val_dataset
    
    def run(self, n_runs = 1, pretrain = True):
        seed_everything()
        # self.trainer.train(self.train_dataset)
        self.trainer.setup()

        if pretrain:
            self.trainer.train(self.train_dataset, self.val_dataset)
        else:
            self.trainer.train(self.train_dataset)
