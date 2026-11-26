"""Use the time-varying sphere contact location for no-slip derivatives."""
from full_body_preview_collision_separating import FullBodyPreview as BasePreview
class FullBodyPreview(BasePreview):
    surface_contact_kinematics = True
