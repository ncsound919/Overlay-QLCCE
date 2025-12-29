"""
Analysis routes for QLCCE API
"""
import sys
import os
import logging
from typing import Optional
import numpy as np
import base64
from io import BytesIO

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

# Configure logging
logger = logging.getLogger(__name__)

# Add parent directory to import QLCCE
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from qlcce_engine import QLCCE_Engine, MultiLogTransformer, BenfordQuantumAnalyzer

router = APIRouter()

# Store running analyses (Note: In production, use Redis or database for persistence)
analyses_store: dict = {}


class AnalysisConfig(BaseModel):
    """Configuration for QLCCE analysis"""
    lattice_size: int = Field(default=64, ge=16, le=128, description="Size of the quantum field lattice")
    field_mass: float = Field(default=0.1, ge=0.01, le=2.0, description="Field mass parameter")
    field_coupling: float = Field(default=1.0, ge=0.1, le=5.0, description="Field coupling constant")
    chaos_system: str = Field(default="lorenz", description="Type of chaotic system")
    log_bases: list[str] = Field(default=["e", "10", "2", "golden"], description="Logarithmic bases to use")
    constraints: list[str] = Field(default=["boundary", "quantum"], description="Constraints to apply")
    n_steps: int = Field(default=500, ge=100, le=2000, description="Number of simulation steps")


class AnalysisResult(BaseModel):
    """Response model for analysis results"""
    id: str
    status: str
    field_properties: Optional[dict] = None
    benford_analysis: Optional[dict] = None
    lyapunov_exponents: Optional[list] = None
    log_periodicity: Optional[dict] = None
    visualization: Optional[str] = None  # Base64 encoded image


class QuickAnalysisRequest(BaseModel):
    """Request for quick analysis"""
    data: list[float] = Field(..., description="Data to analyze")
    analysis_type: str = Field(default="benford", description="Type of analysis: benford, log_transform, chaos")


def run_analysis(analysis_id: str, config: dict):
    """Background task to run QLCCE analysis"""
    try:
        analyses_store[analysis_id]["status"] = "running"
        
        engine = QLCCE_Engine(config)
        results = engine.run_full_analysis(n_steps=config.get('n_steps', 500))
        
        # Create visualization
        fig = plt.figure(figsize=(16, 12))
        
        # 1. Quantum field
        ax1 = fig.add_subplot(2, 3, 1)
        if 'field' in results:
            im = ax1.imshow(results['field'], cmap='RdBu_r', aspect='auto')
            plt.colorbar(im, ax=ax1, shrink=0.7)
            ax1.set_title('Quantum Scalar Field')
        
        # 2. Benford's law
        ax2 = fig.add_subplot(2, 3, 2)
        if 'benford' in results:
            benford = results['benford']
            digits = np.arange(1, 10)
            ax2.bar(digits - 0.2, benford['observed_distribution'], width=0.4, 
                   label='Observed', alpha=0.7)
            ax2.bar(digits + 0.2, benford['expected_distribution'], width=0.4, 
                   label='Expected', alpha=0.7)
            ax2.set_title(f"Benford's Law (score: {benford['compliance_score']:.3f})")
            ax2.legend()
        
        # 3. Chaotic trajectory
        ax3 = fig.add_subplot(2, 3, 3, projection='3d')
        if 'chaos_trajectory' in results:
            trajectory = results['chaos_trajectory']
            if trajectory.shape[1] >= 3:
                ax3.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], 
                        'b-', alpha=0.6, linewidth=0.5)
                ax3.set_title('Chaotic Attractor')
        
        # 4. Log divergence matrix
        ax4 = fig.add_subplot(2, 3, 4)
        if 'log_divergence' in results:
            divergence_data = results['log_divergence']
            # Handle NaN/Inf values with logging
            has_invalid = np.any(~np.isfinite(divergence_data))
            if has_invalid:
                logger.warning(f"Analysis {analysis_id}: Log divergence contains NaN/Inf values, replacing with defaults")
            divergence_clean = np.nan_to_num(divergence_data, nan=0.0, posinf=1.0, neginf=0.0)
            im = ax4.imshow(divergence_clean, cmap='viridis')
            plt.colorbar(im, ax=ax4, shrink=0.7)
            ax4.set_title('Log Transform Divergence')
        
        # 5. Lyapunov exponents
        ax5 = fig.add_subplot(2, 3, 5)
        if 'lyapunov_exponents' in results:
            lexp = results['lyapunov_exponents']
            # Filter out NaN/Inf values with logging
            has_invalid = np.any(~np.isfinite(lexp))
            if has_invalid:
                logger.warning(f"Analysis {analysis_id}: Lyapunov exponents contain NaN/Inf values, replacing with defaults")
            lexp_clean = np.nan_to_num(lexp, nan=0.0, posinf=0.0, neginf=0.0)
            ax5.bar(range(len(lexp_clean)), lexp_clean, color='green', alpha=0.7)
            ax5.axhline(y=0, color='r', linestyle='--')
            ax5.set_title('Lyapunov Exponents')
        
        # 6. Field histogram
        ax6 = fig.add_subplot(2, 3, 6)
        if 'field' in results:
            field_flat = results['field'].flatten()
            # Filter out NaN values
            field_flat = field_flat[np.isfinite(field_flat)]
            if len(field_flat) > 0:
                ax6.hist(field_flat, bins=50, density=True, alpha=0.7)
            ax6.set_title('Field Distribution')
        
        plt.tight_layout()
        
        # Convert to base64
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()
        
        # Helper function to convert numpy types to Python types
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.integer, np.int32, np.int64)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(i) for i in obj]
            return obj
        
        # Store results
        field_props = convert_numpy(results.get('field_properties'))
        
        analyses_store[analysis_id].update({
            "status": "completed",
            "field_properties": field_props,
            "benford_analysis": {
                "observed_distribution": results['benford']['observed_distribution'].tolist() if 'benford' in results else None,
                "expected_distribution": results['benford']['expected_distribution'].tolist() if 'benford' in results else None,
                "compliance_score": float(results['benford']['compliance_score']) if 'benford' in results else None,
                "chi2_p_value": float(results['benford']['chi2_p_value']) if 'benford' in results else None,
            } if 'benford' in results else None,
            "lyapunov_exponents": [float(x) for x in results['lyapunov_exponents']] if 'lyapunov_exponents' in results else None,
            "log_periodicity": {
                "periods": results['log_periodicity']['periods'],
                "significance": float(results['log_periodicity']['significance']),
                "base": results['log_periodicity']['base'],
            } if 'log_periodicity' in results else None,
            "visualization": img_base64,
        })
        
    except Exception as e:
        analyses_store[analysis_id]["status"] = "failed"
        analyses_store[analysis_id]["error"] = str(e)


@router.post("/run", response_model=dict)
async def start_analysis(config: AnalysisConfig, background_tasks: BackgroundTasks):
    """Start a new QLCCE analysis"""
    import uuid
    analysis_id = str(uuid.uuid4())
    
    config_dict = {
        'lattice_size': config.lattice_size,
        'field_mass': config.field_mass,
        'field_coupling': config.field_coupling,
        'chaos_system': config.chaos_system,
        'log_bases': config.log_bases,
        'constraints': config.constraints,
        'n_steps': config.n_steps,
    }
    
    analyses_store[analysis_id] = {
        "id": analysis_id,
        "status": "pending",
        "config": config_dict,
    }
    
    background_tasks.add_task(run_analysis, analysis_id, config_dict)
    
    return {"id": analysis_id, "status": "pending", "message": "Analysis started"}


@router.get("/status/{analysis_id}")
async def get_analysis_status(analysis_id: str):
    """Get the status of an analysis"""
    if analysis_id not in analyses_store:
        raise HTTPException(status_code=404, detail="Analysis not found")
    
    return analyses_store[analysis_id]


@router.get("/result/{analysis_id}", response_model=AnalysisResult)
async def get_analysis_result(analysis_id: str):
    """Get the result of a completed analysis"""
    if analysis_id not in analyses_store:
        raise HTTPException(status_code=404, detail="Analysis not found")
    
    analysis = analyses_store[analysis_id]
    
    if analysis["status"] != "completed":
        raise HTTPException(status_code=400, detail=f"Analysis status: {analysis['status']}")
    
    return AnalysisResult(**analysis)


@router.post("/quick")
async def quick_analysis(request: QuickAnalysisRequest):
    """Run a quick analysis on provided data"""
    data = np.array(request.data)
    
    if request.analysis_type == "benford":
        analyzer = BenfordQuantumAnalyzer()
        result = analyzer.analyze_dataset(data)
        
        if result is None:
            raise HTTPException(status_code=400, detail="Could not analyze data - ensure positive values")
        
        return {
            "type": "benford",
            "observed_distribution": result['observed_distribution'].tolist(),
            "expected_distribution": result['expected_distribution'].tolist(),
            "compliance_score": float(result['compliance_score']),
            "chi2_p_value": float(result['chi2_p_value']),
            "ks_p_value": float(result['ks_p_value']),
        }
    
    elif request.analysis_type == "log_transform":
        transformer = MultiLogTransformer()
        # Filter positive values
        positive_data = data[data > 0]
        if len(positive_data) < 10:
            raise HTTPException(status_code=400, detail="Need at least 10 positive values")
        
        results, divergence = transformer.multi_base_transform(positive_data)
        
        return {
            "type": "log_transform",
            "transforms": {k: v.tolist() for k, v in results.items()},
            "divergence_matrix": divergence.tolist(),
        }
    
    else:
        raise HTTPException(status_code=400, detail=f"Unknown analysis type: {request.analysis_type}")


@router.get("/list")
async def list_analyses():
    """List all analyses"""
    return [
        {"id": k, "status": v["status"]} 
        for k, v in analyses_store.items()
    ]
