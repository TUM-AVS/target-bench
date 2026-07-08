"""Backbone registry for target-bench evaluation."""

from importlib import import_module


BACKBONES = {
    "vggt": "evaluation.backbones.vggt:VGGTBackbone",
    "da3": "evaluation.backbones.da3:DA3Backbone",
    "vggt_omega": "evaluation.backbones.vggt_omega:VGGTOmegaBackbone",
    "vggt-omega": "evaluation.backbones.vggt_omega:VGGTOmegaBackbone",
}


def canonical_backbone_name(name):
    if name == "vggt-omega":
        return "vggt_omega"
    return name


def create_backbone(name, project_root, config_path=None, model_id=None, checkpoint_path=None):
    """Instantiate a backbone runner by name."""
    name = canonical_backbone_name(name)
    if name not in BACKBONES:
        raise ValueError(f"Unsupported backbone: {name}. Supported: {sorted(BACKBONES)}")

    module_name, class_name = BACKBONES[name].split(":")
    module = import_module(module_name)
    runner_class = getattr(module, class_name)
    return runner_class(
        project_root=project_root,
        config_path=config_path,
        model_id=model_id,
        checkpoint_path=checkpoint_path,
    )
