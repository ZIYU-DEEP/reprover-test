"""Script for training the tactic generator."""

import os
from loguru import logger
from pytorch_lightning.cli import LightningCLI

from generator.datamodule import GeneratorDataModule
from generator.model import RetrievalAugmentedGenerator


class CLI(LightningCLI):
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
        
        # Linking arguments from model to data
        parser.link_arguments("model.model_name", "data.model_name")
        parser.link_arguments("data.max_inp_seq_len", "model.max_inp_seq_len")
        parser.link_arguments("data.max_oup_seq_len", "model.max_oup_seq_len")

    def before_instantiate_classes(self):
        # Use the custom load method if a checkpoint path is provided
        checkpoint_path = self.config.get('checkpoint_path')
        
        if checkpoint_path:
            # Set the device
            devices = self.config['trainer']['devices']
            device = f'cuda:{devices[0]}' if isinstance(devices, list) else 'cuda' if devices > 0 else 'cpu'
            
            # Load the model from checkpoint
            model = RetrievalAugmentedGenerator.load(
                ckpt_path=checkpoint_path,
                device=device,
                freeze=False 
            )
            self.config_init['model'] = model

    def before_instantiate_classes(self):
        # Use the custom load method if a checkpoint path is provided
        if 'checkpoint_path' in self.config and self.config['checkpoint_path']:
            self.config_init['model'] = RetrievalAugmentedGenerator.load(
                ckpt_path=self.config['checkpoint_path'],
                device=self.config['trainer']['devices'],  # Ensure you specify the device correctly based on your config
                freeze=False  # or True, based on your need
            )


def main() -> None:
    logger.info(f"PID: {os.getpid()}")
    cli = CLI(RetrievalAugmentedGenerator, GeneratorDataModule)
    print("Configuration: \n", cli.config)


if __name__ == "__main__":
    main()
