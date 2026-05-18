import logging
import os
from typing import TYPE_CHECKING, Any, Optional, TypeVar

import lightning as L
import torch
from lightning.fabric.loggers import CSVLogger, TensorBoardLogger
try:
    from lightning.fabric.loggers import WandbLogger
    HAS_WANDB_LOGGER = True
except ImportError:
    try:
        from lightning.pytorch.loggers import WandbLogger
        HAS_WANDB_LOGGER = True
    except ImportError:
        HAS_WANDB_LOGGER = False

from lightning.fabric.utilities.rank_zero import rank_zero_only
from omegaconf import DictConfig, OmegaConf

if TYPE_CHECKING:
    import lightning.fabric.loggers.tensorboard

log = logging.getLogger(__name__)

TensorOrModule = TypeVar("TensorOrModule", torch.Tensor, torch.nn.Module, Any)


class LightningFabricMixin:
    """
    A mixin class for integrating Lightning Fabric into a project.

    This class provides methods to initialize and manage a Lightning Fabric instance for distributed computing,
    including setup with optional logging, device management for tensors and modules, and hyperparameter logging.
    It leverages the Lightning framework to facilitate distributed training and inference across multiple devices
    and nodes, with support for custom logging via TensorBoard.

    Attributes:
    - _fabric (L.Fabric): The Lightning Fabric instance used for distributed computing.

    Note:
    This mixin is designed to be used with classes that require distributed computing capabilities and wish to
    leverage the Lightning Fabric for this purpose. It assumes the presence of a `config` attribute or parameter
    in the consuming class for configuration.
    """

    _fabric: L.Fabric = None

    def setup_lightning_fabric(self, config: DictConfig):
        """
        Initializes and launches the Lightning Fabric with optional logging.

        This method sets up the Lightning Fabric for distributed computing based on the provided configuration. If a fabric
        configuration is not found, it logs a warning and exits. Optionally, if a fabric logger configuration is provided,
        it initializes a logger with the specified settings.

        Expected configuration keys:
        - fabric: The configuration for the Lightning Fabric.
        - fabric_logger: The configuration for the logger (TensorBoardLogger or WandbLogger).
        """
        if self._fabric is None:
            if config.get("fabric", None) is None:
                log.warning("No fabric configuration found. use default settings.")
                self._fabric = L.Fabric()
            else:
                if config.get("fabric_logger", None) is not None:
                    fabric_logger_cfg = config.fabric_logger
                    if "_target_" in fabric_logger_cfg:
                        from hydra.utils import instantiate

                        if "WandbLogger" in fabric_logger_cfg._target_:
                            if HAS_WANDB_LOGGER:
                                # Manually instantiate WandbLogger to avoid Hydra's _target_ resolution error
                                # if the path is incorrect for the current Lightning version.
                                kwargs = OmegaConf.to_container(
                                    fabric_logger_cfg, resolve=True
                                )
                                kwargs.pop("_target_")
                                logger = WandbLogger(**kwargs)
                            else:
                                log.warning(
                                    "WandbLogger not found. Falling back to CSVLogger."
                                )
                                logger = CSVLogger(
                                    root_dir=fabric_logger_cfg.get(
                                        "save_dir", "outputs/logs"
                                    )
                                )
                        else:
                            logger = instantiate(fabric_logger_cfg)
                    elif "root_dir" in fabric_logger_cfg:
                        logger = TensorBoardLogger(**fabric_logger_cfg)
                    else:
                        log.warning(
                            f"Invalid fabric_logger configuration: {fabric_logger_cfg}. No logger will be used."
                        )
                        logger = None
                else:
                    logger = None
                log.info("Launching Lightning Fabric")
                self._fabric = L.Fabric(**config.fabric, loggers=logger)
            self._fabric.launch()
            # Set the log directory in config if it is not already set
            if (
                self.log_dir is not None
                and hasattr(config, "log_dir")
                and config.get("log_dir", None) is None
            ):
                if self._fabric.is_global_zero:
                    log.info(f"Setting log_dir to {self.log_dir}")
                config.log_dir = self.log_dir

    @property
    def fabric(self):
        if self._fabric is None:
            self.setup_lightning_fabric(getattr(self, "config", DictConfig({})))
        return self._fabric

    def _active_fabric_logger(self):
        fabric = self.fabric
        if fabric is None:
            return None
        loggers = getattr(fabric, "_loggers", None)
        if isinstance(loggers, (list, tuple)):
            return loggers[0] if loggers else None
        try:
            return fabric.logger
        except IndexError:
            return None

    @property
    def log_dir(self):
        """
        Retrieves the log directory from the fabric's logger.
        """
        logger = self._active_fabric_logger()
        if logger is not None:
            log_dir = logger.log_dir
            if (
                log_dir is not None
                and self.fabric.is_global_zero
                and not os.path.exists(log_dir)
            ):
                os.makedirs(log_dir, exist_ok=True)
            return log_dir
        else:
            return None

    def to_device(self, obj: TensorOrModule) -> TensorOrModule:
        """
        Moves a tensor or module to the proper device.

        Args:
            obj (TensorOrModule): The tensor or module to move to the device.

        Returns:
            TensorOrModule: the same type of object as the input, moved to the device.
        """
        return self.fabric.to_device(obj)

    @rank_zero_only
    def log_hyperparams(
        self,
        config: Optional[DictConfig] = None,
        save_dir: Optional[str] = None,
        filename: str = "config.yaml",
    ):
        R"""
        Logs the hyperparameters and saves the configuration to a YAML file.
        The YAML file is saved in the log directory by default with the name `config.yaml`, or in the specified save directory `save_dir`.

        Args:
            config (Optional[DictConfig]): The configuration to log and save. If not provided, the class's `config` attribute is used.
            save_dir (Optional[str]): The directory in which to save the configuration file. If not provided, the log directory is used.
            filename (str): The name of the configuration file. Default is `config.yaml`.
        """
        if config is None:
            config = self.config
        if save_dir is None:
            save_dir = self.log_dir
        logger = self._active_fabric_logger()
        if logger is not None:
            logger.log_hyperparams(
                OmegaConf.to_container(config, resolve=True, enum_to_str=True)
            )
        if save_dir is None:
            return
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
        OmegaConf.save(
            config,
            os.path.join(self.log_dir if save_dir is None else save_dir, filename),
        )

    @property
    def tensorboard_summarywriter(
        self,
    ) -> "lightning.fabric.loggers.tensorboard.SummaryWriter":
        logger = self._active_fabric_logger()
        if isinstance(logger, TensorBoardLogger):
            return logger.experiment
        else:
            raise AttributeError("the logger is not a TensorBoardLogger.")

    @property
    def is_debug_mode(self):
        if hasattr(self, "config") and self.config.get("fast_dev_run", False):
            return True
        elif hasattr(self, "_program") and self._program.config.get(
            "fast_dev_run", False
        ):
            return True
        else:
            return False
