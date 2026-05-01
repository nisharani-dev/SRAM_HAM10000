import subprocess

def run_openram(config_path):
    try:
        result = subprocess.run(
            ["python", "sram_compiler.py", config_path],
            capture_output=True,
            text=True
        )
        return result.stdout, result.stderr
    except Exception as e:
        return None, str(e)