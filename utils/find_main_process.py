import subprocess
import psutil
import re

def find_python_processes():
    python_procs = []
    for proc in psutil.process_iter(attrs=["pid", "name", "cmdline"]):
        try:
            if "python" in proc.info["name"].lower():
                python_procs.append((proc.info["pid"], " ".join(proc.info["cmdline"])))
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return python_procs

def check_py_spy(pid):
    try:
        output = subprocess.check_output(
            ["py-spy", "dump", "--pid", str(pid)],
            stderr=subprocess.DEVNULL,
            timeout=5,
            text=True
        )
        return output
    except subprocess.TimeoutExpired:
        return f"PID {pid}: timeout"
    except subprocess.CalledProcessError:
        return f"PID {pid}: permission error or not attachable"
    except Exception as e:
        return f"PID {pid}: error {e}"

def main():
    targets = find_python_processes()
    print(f"Found {len(targets)} python processes\n")

    for pid, cmdline in targets:
        print(f"Checking PID {pid} ({cmdline})...")
        dump = check_py_spy(pid)

        if any(keyword in dump for keyword in ["main.py", "rqa_analysis", "ProcessPoolExecutor", "as_completed"]):
            print(f"\n✅ Potential MAIN process found (PID {pid}):")
            print("="*60)
            print(dump)
            print("="*60)
        else:
            print(f"❌ PID {pid} doesn't look like main logic\n")

if __name__ == "__main__":
    main()
