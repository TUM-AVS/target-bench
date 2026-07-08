from typing_extensions import Literal, TypeAlias

from ..models.wan_video_animate_adapter import WanAnimateAdapter
from ..models.wan_video_dit import WanModel
from ..models.wan_video_dit_s2v import WanS2VModel
from ..models.wan_video_image_encoder import WanImageEncoder
from ..models.wan_video_motion_controller import WanMotionControllerModel
from ..models.wan_video_text_encoder import WanTextEncoder
from ..models.wan_video_vace import VaceWanModel
from ..models.wan_video_vae import WanVideoVAE, WanVideoVAE38
from ..models.wav2vec import WanS2VAudioEncoder


model_loader_configs = [
    # Wan video DiT variants.
    (None, "9269f8db9040a9d860eaca435be61814", ["wan_video_dit"], [WanModel], "civitai"),
    (None, "aafcfd9672c3a2456dc46e1cb6e52c70", ["wan_video_dit"], [WanModel], "civitai"),
    (None, "6bfcfb3b342cb286ce886889d519a77e", ["wan_video_dit"], [WanModel], "civitai"),
    (None, "6d6ccde6845b95ad9114ab993d917893", ["wan_video_dit"], [WanModel], "civitai"),
    (None, "349723183fc063b2bfc10bb2835cf677", ["wan_video_dit"], [WanModel], "civitai"),
    (None, "efa44cddf936c70abd0ea28b6cbe946c", ["wan_video_dit"], [WanModel], "civitai"),
    (None, "3ef3b1f8e1dab83d5b71fd7b617f859f", ["wan_video_dit"], [WanModel], "civitai"),
    (None, "70ddad9d3a133785da5ea371aae09504", ["wan_video_dit"], [WanModel], "civitai"),
    (None, "26bde73488a92e64cc20b0a7485b9e5b", ["wan_video_dit"], [WanModel], "civitai"),
    (None, "ac6a5aa74f4a0aab6f64eb9a72f19901", ["wan_video_dit"], [WanModel], "civitai"),
    (None, "b61c605c2adbd23124d152ed28e049ae", ["wan_video_dit"], [WanModel], "civitai"),
    (None, "1f5ab7703c6fc803fdded85ff040c316", ["wan_video_dit"], [WanModel], "civitai"),
    (None, "5b013604280dd715f8457c6ed6d6a626", ["wan_video_dit"], [WanModel], "civitai"),
    (None, "2267d489f0ceb9f21836532952852ee5", ["wan_video_dit"], [WanModel], "civitai"),
    (None, "47dbeab5e560db3180adf51dc0232fb1", ["wan_video_dit"], [WanModel], "civitai"),
    (None, "cb104773c6c2cb6df4f9529ad5c60d0b", ["wan_video_dit"], [WanModel], "diffusers"),
    (None, "966cffdcc52f9c46c391768b27637614", ["wan_video_dit"], [WanS2VModel], "civitai"),

    # Optional Wan control/adapter modules.
    (None, "a61453409b67cd3246cf0c3bebad47ba", ["wan_video_dit", "wan_video_vace"], [WanModel, VaceWanModel], "civitai"),
    (None, "7a513e1f257a861512b1afd387a8ecd9", ["wan_video_dit", "wan_video_vace"], [WanModel, VaceWanModel], "civitai"),
    (None, "dbd5ec76bbf977983f972c151d545389", ["wan_video_motion_controller"], [WanMotionControllerModel], "civitai"),
    (None, "06be60f3a4526586d8431cd038a71486", ["wans2v_audio_encoder"], [WanS2VAudioEncoder], "civitai"),
    (None, "31fa352acb8a1b1d33cd8764273d80a2", ["wan_video_dit", "wan_video_animate_adapter"], [WanModel, WanAnimateAdapter], "civitai"),

    # Wan shared components.
    (None, "9c8818c2cbea55eca56c7b447df170da", ["wan_video_text_encoder"], [WanTextEncoder], "civitai"),
    (None, "5941c53e207d62f20f9025686193c40b", ["wan_video_image_encoder"], [WanImageEncoder], "civitai"),
    (None, "1378ea763357eea97acdef78e65d6d96", ["wan_video_vae"], [WanVideoVAE], "civitai"),
    (None, "ccc42284ea13e1ad04693284c7a09be6", ["wan_video_vae"], [WanVideoVAE], "civitai"),
    (None, "e1de6c02cdac79f8b739f4d3698cd216", ["wan_video_vae"], [WanVideoVAE38], "civitai"),
]

huggingface_model_loader_configs = []
patch_model_loader_configs = []

Preset_model_id: TypeAlias = Literal["Wan2.2-TI2V-5B"]

preset_models_on_huggingface = {}
preset_models_on_modelscope = {}
