import Incept
from Incept.utils.data import CommonDataset
from Incept.utils import seed_everything
from Incept.exp import ExpManager

pretrain_mode = False

# step 1, load config
config = Incept.configs.load_exp_config("DEC", "MNIST")

# step 2, load trainer
if pretrain_mode:
    trainer = Incept.models.DECPretrainer(config)
else:
    trainer = Incept.models.DECTrainer(config, pretrained = True)

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
exp = ExpManager(config, trainer, train_dataset, val_dataset)
if pretrain_mode:
    exp.run(pretrain=True)
else:
    exp.run(pretrain=False)