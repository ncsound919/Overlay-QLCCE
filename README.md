# QLCCE - Quantum-Logarithmic Chaotic Constraint Engine

A full-stack research environment for studying quantum field theory simulations, Benford's law compliance testing, multi-base logarithmic transformations, and chaotic systems with adaptive constraints.

![QLCCE Dashboard](https://github.com/user-attachments/assets/de2b7f7d-17e0-4d5a-ac63-ae0703e5073a)

## Features

- **Quantum Field Simulations**: Lattice-based quantum scalar field generation using Metropolis-Hastings algorithm
- **Benford's Law Analysis**: Statistical testing for digit distribution compliance across scales
- **Multi-base Logarithmic Transforms**: Analysis with various bases (e, 10, 2, golden ratio, etc.)
- **Chaotic Systems**: Lorenz, Rössler, and logistic map simulations with adaptive constraints
- **Log-Periodic Pattern Detection**: Detection of discrete scale invariance signatures
- **Real-time Visualization**: Interactive charts and comprehensive matplotlib visualizations

## Architecture

```
├── backend/               # FastAPI Python backend
│   ├── app/
│   │   ├── main.py       # Application entry point
│   │   └── routes/       # API endpoints
│   ├── qlcce_engine.py   # Core QLCCE engine
│   └── requirements.txt  # Python dependencies
├── frontend/             # React TypeScript frontend
│   ├── src/
│   │   ├── App.tsx       # Main application
│   │   └── components/   # React components
│   └── package.json      # Node dependencies
└── run.py                # Standalone launcher script
```

## Quick Start

### Prerequisites

- **Python 3.9+** 
- **Node.js 18+** and npm

### One-Command Launch

Simply run the launcher script to start both backend and frontend:

```bash
python run.py
```

This will:
1. Install all Python dependencies automatically
2. Install all Node.js dependencies automatically
3. Start the backend API server on port 8000
4. Start the frontend development server on port 5173
5. Display access URLs when ready

Access the application:
- **Frontend**: http://localhost:5173
- **Backend API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs

Press `Ctrl+C` to stop all services.

### Manual Setup (Alternative)

If you prefer to run services separately:

#### Backend

```bash
cd backend
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

#### Frontend

```bash
cd frontend
npm install
npm run dev
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/analysis/run` | POST | Start a new QLCCE analysis |
| `/api/analysis/status/{id}` | GET | Get analysis status |
| `/api/analysis/result/{id}` | GET | Get completed analysis results |
| `/api/analysis/quick` | POST | Quick analysis on provided data |
| `/api/analysis/list` | GET | List all analyses |
| `/api/config/default` | GET | Get default configuration |
| `/api/config/options` | GET | Get available options |

## Configuration Options

| Parameter | Description | Range |
|-----------|-------------|-------|
| `lattice_size` | Quantum field lattice size | 16 - 128 |
| `field_mass` | Field mass parameter | 0.01 - 2.0 |
| `field_coupling` | Coupling constant | 0.1 - 5.0 |
| `n_steps` | Simulation steps | 100 - 2000 |
| `chaos_system` | Type of chaotic system | lorenz, rossler, logistic |
| `log_bases` | Logarithmic bases | e, 10, 2, golden, silver, bronze, pi, sqrt2 |
| `constraints` | Applied constraints | boundary, symmetry, conservation, quantum |

## Analysis Results

![Analysis Results](https://github.com/user-attachments/assets/74aec403-6bfc-4f94-bfa6-691b5cbbee1a)

The analysis provides:
- **Field Properties**: Mean, variance, skewness, kurtosis, entropy
- **Benford's Law**: Compliance score, chi-square p-value, digit distributions
- **Lyapunov Exponents**: Chaos indicators
- **Log-Periodic Patterns**: Detected periodicities and significance
- **Visualizations**: Quantum field, chaotic attractor, power spectrum

## License

See [LICENSE](LICENSE) file for details.
