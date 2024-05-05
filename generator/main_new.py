"""Script for training the tactic generator."""

import os
from loguru import logger
from pytorch_lightning.cli import LightningCLI

from generator.datamodule import GeneratorDataModule
from generator.model import RetrievalAugmentedGenerator


# class CLI(LightningCLI):
    # def add_arguments_to_parser(self, parser) -> None:
    #     parser.link_arguments("model.model_name", "data.model_name")
    #     parser.link_arguments("data.max_inp_seq_len", "model.max_inp_seq_len")
    #     parser.link_arguments("data.max_oup_seq_len", "model.max_oup_seq_len")

class CLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        # Base setting
        super().add_arguments_to_parser(parser)
        parser.add_argument("--checkpoint_path", type=str, default=None, 
                            help="Path to the checkpoint file from which to load the model.")
        # Ensure the parser recognizes 'checkpoint_path' as a valid configuration key
        parser.set_defaults({
            'checkpoint_path': None  # Setting a default value that LightningCLI can recognize
        })
        
        # Linking arguments from model to data
        parser.link_arguments("model.model_name", "data.model_name")
        parser.link_arguments("data.max_inp_seq_len", "model.max_inp_seq_len")
        parser.link_arguments("data.max_oup_seq_len", "model.max_oup_seq_len")

    def before_instantiate_classes(self):
        super().before_instantiate_classes()
        # Use the custom load method if a checkpoint path is provided
        checkpoint_path = self.config.get('checkpoint_path')
        seed_everything = self.config.get('seed_everything')
        
        if seed_everything: logger.info('Using seed!')
        else: logger.info('No seed!')
        
        if checkpoint_path:
            logger.info(f"Loading model from checkpoint: {checkpoint_path}")
            
            # Set the device
            devices = self.config['trainer']['devices']
            device = f'cuda:{devices[0]}' if isinstance(devices, list) else 'cuda' if devices > 0 else 'cpu'
            
            # Load the model from checkpoint
            model = RetrievalAugmentedGenerator.load(
                ckpt_path=checkpoint_path,
                device=device,
                freeze=False,
            )
            self.config_init['model'] = model
            logger.info(f"Model successfully loaded from {checkpoint_path}")
            if hasattr(model, 'summary'):
                model.summary()
        else:
            logger.info("No checkpoint path provided; initializing model from scratch.")


def main() -> None:
    # Start
    logger.info(f"PID: {os.getpid()}")
    logger.info("Starting the training process.")
    cli = CLI(RetrievalAugmentedGenerator, GeneratorDataModule)
    
    # Set the config
    logger.info("Configuration: \n", cli.config)
    logger.info("Configuration loaded and CLI setup is complete.")


if __name__ == "__main__":
    main()
