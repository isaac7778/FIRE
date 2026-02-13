from typing import Dict

from omegaconf import OmegaConf

import wandb


class SimpleTrainerLogger(object):
    """Logger with the same interface as WandbTrainerLogger but no wandb (e.g. when disabled or no access)."""

    def __init__(self, cfg: Dict):
        self.cfg = cfg
        self.reset()

    def update_metric(self, **kwargs) -> None:
        for k, v in kwargs.items():
            if isinstance(v, float) or isinstance(v, int):
                self.average_meter_dict.update(k, v)
            else:
                self.media_dict[k] = v

    def log_metric(self, step: int) -> Dict:
        log_data = {}
        log_data.update(self.average_meter_dict.averages())
        log_data.update(self.media_dict)
        if log_data:
            print(f"[step {step}] {log_data}")
        return log_data

    def reset(self) -> None:
        self.average_meter_dict = AverageMeterDict()
        self.media_dict = {}


class WandbTrainerLogger(object):
    def __init__(self, cfg: Dict):
        self.cfg = cfg
        self._use_wandb = True
        dict_cfg = OmegaConf.to_container(cfg, throw_on_missing=True)

        if getattr(cfg, "disable_wandb", False):
            self._use_wandb = False
        else:
            try:
                wandb.init(
                    project=cfg.project_name,
                    entity=cfg.entity_name,
                    group=cfg.group_name,
                    config=dict_cfg,
                )
            except wandb.errors.CommError as e:
                if "403" in str(e) or "Forbidden" in str(e):
                    self._use_wandb = False
                    print(
                        "wandb init failed (e.g. no permission to entity/project). "
                        "Logging locally only. Set disable_wandb: true in config to silence."
                    )
                else:
                    raise

        self.reset()

    def update_metric(self, **kwargs) -> None:
        for k, v in kwargs.items():
            if isinstance(v, float) or isinstance(v, int):
                self.average_meter_dict.update(k, v)
            else:
                self.media_dict[k] = v

    def log_metric(self, step: int) -> Dict:
        log_data = {}
        log_data.update(self.average_meter_dict.averages())
        log_data.update(self.media_dict)
        if self._use_wandb:
            wandb.log(log_data, step=step)
        elif log_data:
            print(f"[step {step}] {log_data}")

    def reset(self) -> None:
        self.average_meter_dict = AverageMeterDict()
        self.media_dict = {}


class AverageMeterDict(object):
    """
    Manages a collection of AverageMeter instances,
    allowing for grouped tracking and averaging of multiple metrics.
    """

    def __init__(self, meters=None):
        self.meters = meters if meters else {}

    def __getitem__(self, key):
        if key not in self.meters:
            meter = AverageMeter()
            meter.update(0)
            return meter
        return self.meters[key]

    def update(self, name, value, n=1) -> None:
        if name not in self.meters:
            self.meters[name] = AverageMeter()
        self.meters[name].update(value, n)

    def reset(self) -> None:
        for meter in self.meters.values():
            meter.reset()

    def values(self, format_string="{}"):
        return {
            format_string.format(name): meter.val for name, meter in self.meters.items()
        }

    def averages(self, format_string="{}"):
        return {
            format_string.format(name): meter.avg for name, meter in self.meters.items()
        }


class AverageMeter(object):
    """
    Tracks and calculates the average and current values of a series of numbers.
    """

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        # TODO: description for using n
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __format__(self, format):
        return "{self.val:{format}} ({self.avg:{format}})".format(
            self=self, format=format
        )
