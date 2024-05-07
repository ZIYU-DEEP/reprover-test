from pytorch_lightning.cli import LightningCLI
from loguru import logger
import os

from generator.model import RetrievalAugmentedGenerator
from generator.datamodule import GeneratorDataModule
from common import set_logger

class CustomCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        super().add_arguments_to_parser(parser)
        
        # New arguments to start with a ckpt
        parser.add_argument("--init_ckpt_path", type=str, default=None,
                            help="Path to the checkpoint file to init the model.")
           
        # Linking arguments
        parser.link_arguments("model.model_name", "data.model_name")
        parser.link_arguments("data.max_inp_seq_len", "model.max_inp_seq_len")
        parser.link_arguments("data.max_oup_seq_len", "model.max_oup_seq_len")

    def before_fit(self):
        # Logging the config
        logger.info(f'Config: {self.config}')
        init_ckpt_path = self.config.fit.init_ckpt_path

        if init_ckpt_path:
            # Set the device
            device = self.config.fit.trainer.devices
            if isinstance(device, list): device = f'cuda:{device[0]}' 
            elif device: device = 'cuda'
            else: device = 'cpu' 

            self.model = RetrievalAugmentedGenerator.load(
                ckpt_path=init_ckpt_path,
                device=device,
                freeze=False
            )
            logger.info(f"Model loaded from checkpoint: {init_ckpt_path}")
        else:
            logger.info("No checkpoint provided; starting training from scratch.")

def main():
    # Set the logger
    set_logger(verbose=True)
    logger.info(f"PID: {os.getpid()}.")
    logger.info(f"Starting the training process.")
    
    # Set the client and train
    cli = CustomCLI(model_class=RetrievalAugmentedGenerator, 
                    datamodule_class=GeneratorDataModule,
                    run=True)
    logger.info(f"Configuration loaded and training is complete.")


if __name__ == "__main__":
    main()
