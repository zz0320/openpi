import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


# Astribot S1 action dimensions
ASTRIBOT_ACTION_DIM = 25  # Full action space with chassis
ASTRIBOT_ACTION_DIM_NO_CHASSIS = 22  # Without chassis


def make_astribot_example() -> dict:
    """Creates a random input example for the Astribot S1 policy."""
    return {
        "observation.state": np.random.rand(25).astype(np.float32),
        "observation.images.head": np.random.randint(256, size=(720, 1280, 3), dtype=np.uint8),
        "observation.images.wrist_left": np.random.randint(256, size=(360, 640, 3), dtype=np.uint8),
        "observation.images.wrist_right": np.random.randint(256, size=(360, 640, 3), dtype=np.uint8),
        "prompt": "clear up the desktop",
    }


def _parse_image(image) -> np.ndarray:
    """Parse image to correct format for the model."""
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class AstribotInputs(transforms.DataTransformFn):
    """
    Astribot S1 input transform.
    
    Astribot S1 robot configuration:
    - 25-dimensional state:
        - arm_left: 7 joint positions (joint_0 to joint_6)
        - arm_right: 7 joint positions (joint_0 to joint_6)
        - gripper_left: 1 position
        - gripper_right: 1 position
        - head: 2 positions (pitch, yaw)
        - torso: 4 positions
        - chassis: 3 positions (x, y, theta)
    - 25-dimensional actions (same structure as state)
    - 4 cameras:
        - head (720x1280) - main view
        - wrist_left (360x640) - left wrist camera
        - wrist_right (360x640) - right wrist camera
        - torso (720x1280) - torso camera (optional, not used in this config)
    """
    model_type: _model.ModelType
    action_dim: int = ASTRIBOT_ACTION_DIM  # Default to 25 dimensions

    def __call__(self, data: dict) -> dict:
        # Get state (25 dimensions)
        state = data["observation.state"]
        
        # Parse images from Astribot format
        head_image = _parse_image(data["observation.images.head"])
        wrist_left_image = _parse_image(data["observation.images.wrist_left"])
        wrist_right_image = _parse_image(data["observation.images.wrist_right"])
        
        # Create inputs dict mapping Astribot cameras to pi0 expected format
        inputs = {
            "state": state,
            "image": {
                "base_0_rgb": head_image,  # head camera as base view
                "left_wrist_0_rgb": wrist_left_image,
                "right_wrist_0_rgb": wrist_right_image,
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.True_,
            },
        }
        
        # Actions (only available during training)
        if "actions" in data:
            inputs["actions"] = data["actions"]
        
        # Prompt / language instruction
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]
        
        return inputs


@dataclasses.dataclass(frozen=True)
class AstribotOutputs(transforms.DataTransformFn):
    """
    Convert model outputs back to Astribot S1 format.
    
    Returns 25-dimensional actions:
    - arm_left: 7 joint actions      [0:7]
    - arm_right: 7 joint actions     [7:14]
    - gripper_left: 1 action         [14]
    - gripper_right: 1 action        [15]
    - head: 2 actions                [16:18]
    - torso: 4 actions               [18:22]
    - chassis: 3 actions             [22:25]
    """
    action_dim: int = ASTRIBOT_ACTION_DIM  # Default to 25 dimensions

    def __call__(self, data: dict) -> dict:
        # Return the first action_dim dimensions (Astribot action space)
        return {"actions": np.asarray(data["actions"][:, :self.action_dim])}
