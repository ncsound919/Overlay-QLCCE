"""
QLCCE FastAPI Backend - Main Application
"""
import sys
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Add parent directory to path to import QLCCE
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from app.routes import analysis, config


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler"""
    yield


app = FastAPI(
    title="QLCCE API",
    description="Quantum-Logarithmic Chaotic Constraint Engine API",
    version="1.0.0",
    lifespan=lifespan,
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173", "http://127.0.0.1:3000", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(analysis.router, prefix="/api/analysis", tags=["analysis"])
app.include_router(config.router, prefix="/api/config", tags=["config"])


@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "QLCCE API is running", "version": "1.0.0"}


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}
