import Incept
import Incept.models
from Incept.utils.data import CommonDataset

# step 1, load config
config = Incept.configs.load_exp_config("DEC", "MNIST")
# step 2, load trainer
# trainer = Incept.models.DECTrainer(config, pretrained = True)
trainer = Incept.models.DECPretrainer(config)
# step 3, load dataset
train_dataset = CommonDataset(
    config.dataset_name, config.data_dir, True,
    trainer.img_transform, trainer.target_transform,
)
val_dataset = CommonDataset(
    config.dataset_name, config.data_dir, False,
    trainer.img_transform, trainer.target_transform,
)
# step 4, run experiment
exp = Incept.exp.ExpManager(config, trainer, train_dataset, val_dataset)
exp.run()