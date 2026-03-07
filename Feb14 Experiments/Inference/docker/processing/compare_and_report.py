import os
import json
from pathlib import Path
import pandas as pd
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors

EVAL_ROOT = Path("/opt/ml/processing/eval")
OUT_DIR = Path("/opt/ml/processing/output")
RUN_DATE = os.getenv("RUN_DATE", "unknown")

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = []

    for p in EVAL_ROOT.rglob("metrics.json"):
        with p.open("r", encoding="utf-8") as f:
            m = json.load(f)

        cutoff = "UNKNOWN"
        for part in p.parts:
            if part.startswith("cutoff="):
                cutoff = part.split("=", 1)[1]

        m["cutoff"] = m.get("cutoff", cutoff)
        rows.append(m)

    if not rows:
        raise RuntimeError("No metrics.json files found")

    df = pd.DataFrame(rows)

    df = df.sort_values(["r2", "rmse"], ascending=[False, True]).reset_index(drop=True)
    df["rank"] = df.index + 1

    df.to_csv(OUT_DIR / "cutoff_comparison.csv", index=False)
    df.to_parquet(OUT_DIR / "cutoff_comparison.parquet", index=False)

    summary = {
        "run_date": RUN_DATE,
        "best_cutoff": df.loc[0, "cutoff"],
        "table": df.to_dict("records")
    }

    (OUT_DIR / "cutoff_comparison.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8"
    )

    # PDF
    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate(str(OUT_DIR / "cutoff_comparison.pdf"))
    story = []

    story.append(Paragraph("GeoAI Yield Forecast - Cutoff Comparison", styles["Title"]))
    story.append(Spacer(1, 12))
    story.append(Paragraph(f"Run Date: {RUN_DATE}", styles["Normal"]))
    story.append(Spacer(1, 12))

    best = df.iloc[0]
    story.append(Paragraph(f"<b>Best Cutoff:</b> {best['cutoff']}", styles["Heading2"]))
    story.append(Paragraph(
        f"R²={best['r2']:.4f}, RMSE={best['rmse']:.4f}, MAE={best['mae']:.4f}",
        styles["Normal"]
    ))
    story.append(Spacer(1, 12))

    table_data = [list(df.columns)] + df.values.tolist()

    t = Table(table_data)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.lightgrey),
        ("GRID", (0,0), (-1,-1), 0.5, colors.grey)
    ]))

    story.append(t)
    doc.build(story)

    print("Comparison + report generated")

if __name__ == "__main__":
    main()