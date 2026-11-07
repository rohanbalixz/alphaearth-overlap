"""Show recent Earth Engine export tasks with state and compute used (EECU-hours).
Run: .venv/bin/python gee/task_status.py <gcp-project>"""
import sys
import warnings

warnings.filterwarnings("ignore")
import ee  # noqa: E402

ee.Initialize(project=sys.argv[1])
for op in ee.data.listOperations()[:12]:
    md = op.get("metadata", {})
    eecu_h = float(md.get("batchEecuUsageSeconds", 0)) / 3600
    print(f"{md.get('description', '?'):22s} {md.get('state', '?'):10s} progress {float(md.get('progress', 0)):.0%}  "
          f"EECU-hours {eecu_h:7.2f}  started {md.get('startTime', '-')}  updated {md.get('updateTime', '-')}")
    if md.get("state") == "FAILED":
        print("   error:", op.get("error", {}).get("message"))
