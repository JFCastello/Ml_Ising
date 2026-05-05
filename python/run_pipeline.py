"""
Pipeline orchestrator — called automatically by ./main after sample generation.
Executes both analysis notebooks and reports status.
"""
import subprocess
import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR   = os.path.dirname(SCRIPT_DIR)

def run_notebook(nb_path: str) -> bool:
    abs_path = os.path.join(SCRIPT_DIR, nb_path)
    print(f"  Executing {nb_path}...")
    import shutil
    nbconvert_cmd = shutil.which("jupyter-nbconvert") or shutil.which("jupyter")
    if nbconvert_cmd is None:
        print("  ERROR: jupyter-nbconvert not found. Install with: pip install nbconvert")
        return False

    # jupyter-nbconvert is a direct binary; jupyter needs the subcommand
    cmd = (
        [nbconvert_cmd, "--to", "notebook", "--execute", "--inplace",
         "--ExecutePreprocessor.timeout=600", abs_path]
        if "jupyter-nbconvert" in nbconvert_cmd
        else [nbconvert_cmd, "nbconvert", "--to", "notebook", "--execute",
              "--inplace", "--ExecutePreprocessor.timeout=600", abs_path]
    )
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  ERROR running {nb_path}:")
        print(result.stderr[-2000:])
        return False
    print(f"  {nb_path} — done.")
    return True

def main():
    os.makedirs(os.path.join(ROOT_DIR, "plots"), exist_ok=True)

    print("\n[Pipeline] Step 1: Energy & Magnetization Distributions")
    ok1 = run_notebook("Distributions.ipynb")

    print("\n[Pipeline] Step 2: ML Model Training & Tc Estimation")
    ok2 = run_notebook("ML_Model.ipynb")

    if ok1 and ok2:
        print("\n[Pipeline] All notebooks executed successfully.")
        print(f"[Pipeline] Plots saved to {os.path.join(ROOT_DIR, 'plots')}/")
    else:
        print("\n[Pipeline] One or more notebooks failed. Check output above.")
        sys.exit(1)

if __name__ == "__main__":
    main()
