import sys

modules = ["numpy", "pandas", "scipy", "sklearn", "xgboost", "mlflow", "fastapi", "uvicorn", "pydantic", "streamlit", "plotly", "pptx", "yaml", "pytest"]
results = {}

for m in modules:
    try:
        mod = __import__(m)
        ver = getattr(mod, "__version__", "installed")
        results[m] = f"OK ({ver})"
    except ImportError as e:
        results[m] = f"MISSING ({e})"

print("PYTHON INTERPRETER:", sys.executable)
print("PYTHON VERSION:", sys.version.split()[0])
print("\n--- DEPENDENCY STATUS ---")
for k, v in results.items():
    print(f"  {k:15}: {v}")
