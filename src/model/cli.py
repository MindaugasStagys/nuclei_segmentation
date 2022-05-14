from os.path import abspath, join
from pytorch_lightning.utilities.cli import LightningCLI
from datasets.data_loaders import PannukeData
from models.model import UNetSharp


class MyLightningCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        root = abspath(join(__file__, '..', '..'))
        parser.set_defaults({
            'data.data_dir': join(root, 'data'),
            'model.test_dir': join(root, 'saved', 'preds'),
            'trainer.default_root_dir': join(root, 'saved')
        })


if __name__ == '__main__':
    root = abspath(join(__file__, '..', '..'))
    default_config = join(root, 'configs', 'config.yaml')
    cli = MyLightningCLI(
        model_class=UNetSharp,
        datamodule_class=PannukeData,
        auto_registry=True,
        parser_kwargs={
            'fit': {'default_config_files': [default_config]},
            'test': {'default_config_files': [default_config]},
            'predict': {'default_config_files': [default_config]}
        }
    )
