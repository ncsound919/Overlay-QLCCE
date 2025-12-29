"""
Configuration routes for QLCCE API
"""
from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()


class DefaultConfig(BaseModel):
    """Default configuration values"""
    lattice_size: int = 64
    field_mass: float = 0.1
    field_coupling: float = 1.0
    chaos_system: str = "lorenz"
    log_bases: list[str] = ["e", "10", "2", "golden"]
    constraints: list[str] = ["boundary", "quantum"]
    n_steps: int = 500


@router.get("/default")
async def get_default_config():
    """Get default configuration"""
    return DefaultConfig()


@router.get("/options")
async def get_config_options():
    """Get available configuration options"""
    return {
        "chaos_systems": ["lorenz", "rossler", "logistic"],
        "log_bases": ["e", "10", "2", "golden", "silver", "bronze", "pi", "sqrt2"],
        "constraints": ["boundary", "symmetry", "conservation", "quantum"],
        "lattice_size_range": {"min": 16, "max": 128, "step": 16},
        "field_mass_range": {"min": 0.01, "max": 2.0, "step": 0.01},
        "field_coupling_range": {"min": 0.1, "max": 5.0, "step": 0.1},
        "n_steps_range": {"min": 100, "max": 2000, "step": 100},
    }
