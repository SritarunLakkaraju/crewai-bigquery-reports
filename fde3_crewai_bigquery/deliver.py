"""Run the weekly report and deliver it. This is the entry point Cloud Scheduler triggers (as a Cloud Run job):

    gcloud run jobs deploy weekly-report --source . --command python --args -m,fde3_crewai_bigquery.deliver
    gcloud scheduler jobs create http weekly-report-trigger --schedule "0 8 * * MON" \
        --uri "https://run.googleapis.com/v2/projects/$PROJECT/locations/$REGION/jobs/weekly-report:run" \
        --http-method POST --oauth-service-account-email $SA

Delivery channels (any that are configured):
  * always      - outbox/<week>/report.md + charts + report.eml (a ready-to-send email with attachments)
  * SLACK_WEBHOOK_URL         - posts the headline section to Slack
  * SMTP_HOST/SMTP_USER/SMTP_PASSWORD/REPORT_TO - sends the email
"""
from __future__ import annotations

import os
import smtplib
from email.message import EmailMessage

from .crew import run_weekly_report


def build_email(report: str, out_dir: str, charts: list[str], to: str = "leadership@example.com") -> EmailMessage:
    msg = EmailMessage()
    msg["Subject"] = report.splitlines()[0].lstrip("# ").strip()
    msg["From"] = os.environ.get("SMTP_USER", "reports@example.com")
    msg["To"] = to
    msg.set_content(report)
    for c in charts:
        with open(os.path.join(out_dir, c), "rb") as f:
            msg.add_attachment(f.read(), maintype="image", subtype="png", filename=c)
    return msg


def deliver(out_dir: str | None = None) -> dict:
    res = run_weekly_report(out_dir)
    if res["unverified_numbers"]:
        raise RuntimeError(f"refusing to send: unverified numbers {res['unverified_numbers']}")
    msg = build_email(res["report"], res["out_dir"], res["charts"], os.environ.get("REPORT_TO", "leadership@example.com"))
    eml = os.path.join(res["out_dir"], "report.eml")
    with open(eml, "wb") as f:
        f.write(bytes(msg))
    sent = ["outbox"]
    if url := os.environ.get("SLACK_WEBHOOK_URL"):
        import httpx

        headline = res["report"].split("## Revenue by category")[0]
        httpx.post(url, json={"text": headline}, timeout=10).raise_for_status()
        sent.append("slack")
    if os.environ.get("SMTP_HOST"):
        with smtplib.SMTP(os.environ["SMTP_HOST"], int(os.environ.get("SMTP_PORT", 587))) as s:
            s.starttls()
            s.login(os.environ["SMTP_USER"], os.environ["SMTP_PASSWORD"])
            s.send_message(msg)
        sent.append("email")
    return {**res, "eml": eml, "delivered_to": sent}


if __name__ == "__main__":
    r = deliver()
    print(f"report for {r['out_dir']} delivered to {r['delivered_to']}; {r['queries']} queries, "
          f"{r['bytes_scanned']:,} bytes scanned, unverified numbers: {r['unverified_numbers']}")
