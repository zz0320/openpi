import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def make_qingloong_example() -> dict:
    """Creates a random input example for the QingLoong policy."""
    return {
        "observation/state": np.random.rand(16),
        "observation/images/front": np.random.randint(256, size=(480, 640, 3), dtype=np.uint8),
        "observation/images/left": np.random.randint(256, size=(480, 640, 3), dtype=np.uint8),
        "observation/images/right": np.random.randint(256, size=(480, 640, 3), dtype=np.uint8),
        "prompt": "do something",
    }


def _parse_image(image) -> np.ndarray:
    """Parse image to correct format for the model."""
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


def _resize_image(image: np.ndarray, target_size: tuple = (224, 224)) -> np.ndarray:
    """Resize image to target size for model input."""
    from PIL import Image
    if isinstance(image, np.ndarray):
        image_pil = Image.fromarray(image)
        resized_image = image_pil.resize(target_size)
        return np.array(resized_image)
    return image


@dataclasses.dataclass(frozen=True)
class QingLoongInputs(transforms.DataTransformFn):
    """
    This class is used to convert inputs to the model to the expected format. It is used for both training and inference.
    
    Adapted for QingLoong dataset which uses:
    - 16-dimensional observation state (14 joint positions + 2 gripper positions)
    - 16-dimensional actions (14 joint actions + 2 gripper actions)
    - Three camera views: front, left, right
    - Original image size: 480x640, resized to 224x224 for model
    """
    # Determines which model will be used.
    # Do not change this for your own dataset.
    action_dim: int
    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        # QingLoong stores state in "observation.state" with 16 dimensions
        # state = transforms.pad_to_dim(data["observation.state"], self.action_dim)
        state = data["observation.state"]

        # Parse and resize images from QingLoong format
        # QingLoong stores images as (C,H,W) video format, need to convert to (H,W,C) and resize
        front_image = _parse_image(data["observation.images.front"])
        left_image = _parse_image(data["observation.images.left"])
        right_image = _parse_image(data["observation.images.right"])
        
        # Resize images from 480x640 to 224x224 for model input===============
        # front_image = _resize_image(front_image, (224, 224))
        # left_image = _resize_image(left_image, (224, 224))
        # right_image = _resize_image(right_image, (224, 224))

        # Create inputs dict. Map QingLoong camera views to pi0 expected format
        inputs = {
            "state": state,
            "image": {
                # Use front camera as base view
                "base_0_rgb": front_image,
                # Map left and right cameras to wrist views
                "left_wrist_0_rgb": left_image,
                "right_wrist_0_rgb": right_image,
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.True_,  # All three cameras available in QingLoong
            },
        }

        # Pad actions to the model action dimension. Keep this for your own dataset.
        # Actions are only available during training.
        if "actions" in data:  # 修正：repack transform 后键名变为 "actions"
            # QingLoong stores actions in "actions" key with 16 dimensions (after repack)
            # actions = transforms.pad_to_dim(data["actions"], self.action_dim)
            # inputs["actions"] = actions
            inputs["actions"] = data["actions"]

        # Pass the prompt (aka language instruction) to the model.
        # QingLoong dataset may not have explicit prompts, so provide a default
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class QingLoongOutputs(transforms.DataTransformFn):
    """
    This class is used to convert outputs from the model back to the QingLoong dataset specific format. 
    It is used for inference only.
    """

    def __call__(self, data: dict) -> dict:
        # Return the full 16-dimensional actions for QingLoong format
        # (14 joint positions + 2 gripper positions)
        return {"actions": np.asarray(data["actions"][:, :16])}
