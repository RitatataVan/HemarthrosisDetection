import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint

from hyperparameters import parameters as params
from dataset import NovoDataModule
from network import Net

pl.seed_everything(params['seed'])

wandb_logger = WandbLogger(project="Qianyu-Augmented-Blood")

# setup data
dataset = NovoDataModule()
dataset.setup()

# Callbacks
checkpoint_callback = ModelCheckpoint(
    monitor='train/loss',
    dirpath='../weights/',
    filename='adtblood-aug-{epoch:02d}',
    mode='min',
)

trainer = pl.Trainer(
    logger=wandb_logger,    # W&B integration
    log_every_n_steps=2,   # set the logging frequency
    min_epochs=params['min_epochs'],
    max_epochs=params['epochs'],
    precision=params['precision'],
    accumulate_grad_batches=params['accumulate_grad_batches'],
    gpus=-1,                # use all GPUs
    deterministic=True,      # keep it deterministic
    callbacks=[checkpoint_callback]
)

# setup model
model = Net() #.load_from_checkpoint('./pretrained_adteff_weights.ckpt')
#model.freeze_weights()

# fit the model
trainer.fit(model, dataset)

# evaluate the model on a test set
trainer.test(datamodule=dataset, ckpt_path=checkpoint_callback.best_model_path)
