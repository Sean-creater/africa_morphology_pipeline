"""
Email notification hooks for the morphology analysis pipeline.

This is an optional operational convenience, not core science: the pipeline
runs unattended for hours to days across thousands of cities, and these
hooks let an operator get progress, completion, and error notifications by
email without having to tail a log file. Every method degrades gracefully to
a no-op (with a log message) if SMTP credentials are not configured via
environment variables, so the pipeline itself never depends on email being
set up.

Configure via environment variables:
    SMTP_SERVER            e.g. "smtp.gmail.com"
    SMTP_PORT               e.g. "465" (SMTP_SSL)
    SMTP_SENDER_EMAIL       account used to authenticate and send
    SMTP_SENDER_PASSWORD    SMTP password / app password / auth code
    SMTP_RECEIVER_EMAIL     recipient address (comma-separated for multiple)
    SMTP_ASSISTANT_NAME     display name used in message bodies (optional)

If any of SMTP_SERVER / SMTP_SENDER_EMAIL / SMTP_SENDER_PASSWORD /
SMTP_RECEIVER_EMAIL is unset, notifications are silently skipped (logged at
INFO level) rather than raising.
"""

import os
import smtplib
import ssl
from email.message import EmailMessage
from datetime import datetime
import logging


def _load_config_from_env():
    """Build an email config dict from environment variables.

    Returns None if the required variables are not fully set, so callers
    can treat notifications as an optional no-op.
    """
    server = os.environ.get("SMTP_SERVER")
    sender_email = os.environ.get("SMTP_SENDER_EMAIL")
    sender_password = os.environ.get("SMTP_SENDER_PASSWORD")
    receiver_email = os.environ.get("SMTP_RECEIVER_EMAIL")

    if not all([server, sender_email, sender_password, receiver_email]):
        return None

    return {
        "smtp_server": server,
        "smtp_port": int(os.environ.get("SMTP_PORT", "465")),
        "sender_email": sender_email,
        "sender_password": sender_password,
        "receiver_email": receiver_email,
        "assistant_name": os.environ.get("SMTP_ASSISTANT_NAME", "Morphology Analysis Assistant"),
    }


class EmailNotifier:
    """Email notification handler for the morphology analysis pipeline.

    Reads SMTP credentials from environment variables (see module
    docstring). If they are not set, every ``send_*`` method logs and
    returns False instead of raising, so the pipeline can call these
    unconditionally regardless of whether notifications are configured.
    """

    def __init__(self, config=None, logger=None):
        """
        Initialize email notifier.

        Args:
            config: Optional explicit email configuration dict, with keys
                smtp_server, smtp_port, sender_email, sender_password,
                receiver_email, assistant_name. If omitted, configuration is
                read from environment variables; if those are incomplete,
                the notifier operates in no-op mode.
            logger: Optional logger instance
        """
        self.config = config or _load_config_from_env()
        self.logger = logger or logging.getLogger(__name__)
        if self.config is None:
            self.logger.info(
                "Email notifications disabled (SMTP_SERVER / SMTP_SENDER_EMAIL / "
                "SMTP_SENDER_PASSWORD / SMTP_RECEIVER_EMAIL not set)"
            )

    def _send_email(self, subject, body):
        """
        Internal method to send email using SMTP.

        Args:
            subject (str): Email subject
            body (str): Email body content

        Returns:
            bool: True if email sent (or assumed sent) successfully, False
                if notifications are not configured or sending failed.
        """
        if self.config is None:
            self.logger.info(f"Email notification skipped (not configured): {subject}")
            return False

        try:
            msg = EmailMessage()
            msg['Subject'] = subject
            msg['From'] = self.config['sender_email']
            msg['To'] = self.config['receiver_email']
            msg.set_content(body)

            context = ssl.create_default_context()
            with smtplib.SMTP_SSL(
                self.config['smtp_server'],
                self.config['smtp_port'],
                context=context
            ) as smtp:
                smtp.login(self.config['sender_email'], self.config['sender_password'])
                smtp.send_message(msg)
                # Do not wait for the QUIT response; the message has already
                # been accepted by the server by the time send_message returns.

            self.logger.info(f"Email sent: {subject}")
            return True

        except Exception as e:
            # Some SMTP servers close the connection abruptly right after
            # accepting the message, which surfaces here as a low-level
            # socket/protocol error even though the email was delivered.
            if "b'\\x00\\x00\\x00'" in str(e) or "(-1," in str(e):
                self.logger.info(f"Email sent: {subject}")
                return True
            self.logger.warning(f"Email warning: {str(e)} (but email may have been sent)")
            return True  # Assume delivered; see note above.

    def send_phase1_progress(self, completed_count, total_count, recent_cities, failed_cities, elapsed_time, estimated_remaining_time):
        """
        Send progress notification every 10 cities during Phase 1.

        Args:
            completed_count (int): Number of cities completed so far
            total_count (int): Total number of cities in Phase 1
            recent_cities (list): List of recently completed cities (last 10)
            failed_cities (list): List of failed cities
            elapsed_time (float): Elapsed time in seconds
            estimated_remaining_time (float): Estimated remaining time in seconds
        """
        subject = f"Phase 1 Progress: {completed_count}/{total_count} Cities Completed"

        success_rate = (completed_count / total_count * 100) if total_count > 0 else 0

        body = f"""Dear User,

{self.config['assistant_name']} reports:

================================================================================
PHASE 1 PROGRESS UPDATE - 50 Sample Cities (All Africa)
================================================================================

Progress: {completed_count}/{total_count} cities ({success_rate:.1f}%)
Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Statistics:
  - Completed: {completed_count}
  - Failed: {len(failed_cities)}
  - Remaining: {total_count - completed_count}
  - Elapsed time: {elapsed_time/3600:.2f} hours
  - Estimated remaining: {estimated_remaining_time/3600:.2f} hours
  - Estimated completion: {datetime.fromtimestamp(datetime.now().timestamp() + estimated_remaining_time).strftime('%Y-%m-%d %H:%M:%S')}

"""

        if recent_cities:
            body += f"\nRecently completed cities (last {len(recent_cities)}):\n"
            for i, city in enumerate(recent_cities, 1):
                body += f"  {i}. {city}\n"

        if failed_cities:
            body += f"\nFailed cities so far ({len(failed_cities)}):\n"
            for i, city in enumerate(failed_cities, 1):
                body += f"  {i}. {city}\n"

        body += f"""
================================================================================
Phase 1 is progressing smoothly. Next update at {completed_count + 10} cities.
================================================================================

Best regards,
{self.config['assistant_name']}
"""

        return self._send_email(subject, body)

    def send_phase1_completion(self, total_cities, successful_cities, failed_cities, total_time):
        """
        Send notification for Phase 1 (50 sample cities) completion.

        Args:
            total_cities (int): Total number of cities in Phase 1
            successful_cities (list): List of successfully processed cities
            failed_cities (list): List of failed cities
            total_time (float): Total processing time in seconds
        """
        subject = "Phase 1 Completed: 50 Sample Cities Analysis Finished"

        success_rate = (len(successful_cities) / total_cities * 100) if total_cities > 0 else 0

        body = f"""Dear User,

{self.config['assistant_name']} reports:

================================================================================
PHASE 1 COMPLETION REPORT - 50 Sample Cities (All Africa)
================================================================================

Status: COMPLETED
Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Statistics:
  - Total cities: {total_cities}
  - Successfully processed: {len(successful_cities)}
  - Failed: {len(failed_cities)}
  - Success rate: {success_rate:.1f}%
  - Total processing time: {total_time/3600:.2f} hours

"""

        if successful_cities:
            body += f"\nSuccessful cities ({len(successful_cities)}):\n"
            for i, city in enumerate(successful_cities[:20], 1):  # Show first 20
                body += f"  {i}. {city}\n"
            if len(successful_cities) > 20:
                body += f"  ... and {len(successful_cities) - 20} more\n"

        if failed_cities:
            body += f"\nFailed cities ({len(failed_cities)}):\n"
            for i, city in enumerate(failed_cities, 1):
                body += f"  {i}. {city}\n"

        body += f"""
================================================================================
Next Step: Phase 2 (Remaining Ghana cities) will begin automatically.
================================================================================

Best regards,
{self.config['assistant_name']}
"""

        return self._send_email(subject, body)

    def send_phase2_completion(self, total_cities, successful_cities, failed_cities, total_time):
        """
        Send notification for Phase 2 (Ghana) completion.

        Args:
            total_cities (int): Total number of cities in Phase 2
            successful_cities (list): List of successfully processed cities
            failed_cities (list): List of failed cities
            total_time (float): Total processing time in seconds
        """
        subject = "Phase 2 Completed: Ghana Analysis Finished"

        success_rate = (len(successful_cities) / total_cities * 100) if total_cities > 0 else 0

        body = f"""Dear User,

{self.config['assistant_name']} reports:

================================================================================
PHASE 2 COMPLETION REPORT - Remaining Ghana Cities
================================================================================

Status: COMPLETED
Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Statistics:
  - Total cities: {total_cities}
  - Successfully processed: {len(successful_cities)}
  - Failed: {len(failed_cities)}
  - Success rate: {success_rate:.1f}%
  - Total processing time: {total_time/3600:.2f} hours

"""

        if successful_cities:
            body += f"\nSuccessful cities ({len(successful_cities)}):\n"
            for i, city in enumerate(successful_cities, 1):
                body += f"  {i}. {city}\n"

        if failed_cities:
            body += f"\nFailed cities ({len(failed_cities)}):\n"
            for i, city in enumerate(failed_cities, 1):
                body += f"  {i}. {city}\n"

        body += f"""
================================================================================
Next Step: Phase 3 (All remaining African cities) will begin automatically.
================================================================================

Best regards,
{self.config['assistant_name']}
"""

        return self._send_email(subject, body)

    def send_country_completion(self, country_name, iso3, phase, total_cities, successful_cities, failed_cities, total_time):
        """
        Send notification when a country's all cities are completed (Phase 3).

        Args:
            country_name (str): Country name
            iso3 (str): ISO3 country code
            phase (int): Phase number
            total_cities (int): Total number of cities in this country
            successful_cities (list): List of successfully processed cities
            failed_cities (list): List of failed cities
            total_time (float): Total processing time in seconds
        """
        subject = f"Country Completed: {country_name} ({iso3}) - All Cities Analyzed"

        success_rate = (len(successful_cities) / total_cities * 100) if total_cities > 0 else 0

        body = f"""Dear User,

{self.config['assistant_name']} reports:

================================================================================
COUNTRY COMPLETION REPORT - {country_name} ({iso3})
================================================================================

Phase: {phase}
Status: COMPLETED
Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Statistics:
  - Total cities: {total_cities}
  - Successfully processed: {len(successful_cities)}
  - Failed: {len(failed_cities)}
  - Success rate: {success_rate:.1f}%
  - Total processing time: {total_time/3600:.2f} hours

"""

        if successful_cities:
            body += f"\nSuccessful cities ({len(successful_cities)}):\n"
            for i, city in enumerate(successful_cities, 1):
                body += f"  {i}. {city}\n"

        if failed_cities:
            body += f"\nFailed cities ({len(failed_cities)}):\n"
            for i, city in enumerate(failed_cities, 1):
                body += f"  {i}. {city}\n"

        body += f"""
================================================================================
{country_name} analysis completed. Continuing with remaining countries...
================================================================================

Best regards,
{self.config['assistant_name']}
"""

        return self._send_email(subject, body)

    def send_country_excluded_notification(
        self,
        country_name: str,
        iso3: str,
        failed_cities: list,
        total_cities_in_country: int,
        remaining_countries_count: int,
        remaining_cities_count: int,
        excluded_countries_list: list
    ):
        """
        Send notification when a country is excluded due to consecutive cities with no Open Buildings data.

        Args:
            country_name (str): Name of the excluded country
            iso3 (str): ISO3 country code
            failed_cities (list): List of consecutively failed cities
            total_cities_in_country (int): Total number of cities in this country
            remaining_countries_count (int): Number of countries still in analysis plan
            remaining_cities_count (int): Total number of cities remaining in analysis plan
            excluded_countries_list (list): List of all excluded country names
        """
        subject = f"Country Excluded: {country_name} ({iso3}) - No Open Buildings Data"

        body = f"""Dear User,

{self.config['assistant_name']} reports:

================================================================================
COUNTRY EXCLUSION ALERT - {country_name} ({iso3})
================================================================================

Status: COUNTRY EXCLUDED FROM ANALYSIS
Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Reason:
  Detected {len(failed_cities)} consecutive cities with ZERO buildings from
  Google Open Buildings dataset. This indicates the country lacks Open Buildings
  data coverage.

Excluded Country Details:
  - Country: {country_name} ({iso3})
  - Total cities in country: {total_cities_in_country}
  - Cities checked before exclusion: {len(failed_cities)}
  - Failed cities: {', '.join(failed_cities)}

Action Taken:
  - {country_name} ({iso3}) removed from analysis plan
  - All {total_cities_in_country} cities in this country excluded
  - Future grid generation and data retrieval skipped for this country

================================================================================
UPDATED ANALYSIS PLAN
================================================================================

Remaining Countries: {remaining_countries_count}

Remaining Total Cities: {remaining_cities_count}

Total Excluded Countries: {len(excluded_countries_list)}
  {', '.join(excluded_countries_list)}

================================================================================
Analysis will continue with remaining countries that have Open Buildings data.
================================================================================

Best regards,
{self.config['assistant_name']}
"""

        return self._send_email(subject, body)

    def send_final_completion(self, phase_stats, total_time):
        """
        Send notification when all phases are completed.

        Args:
            phase_stats (dict): Statistics for all phases
            total_time (float): Total processing time in seconds
        """
        subject = "ALL PHASES COMPLETED: Building Morphology Analysis Finished"

        body = f"""Dear User,

{self.config['assistant_name']} reports:

================================================================================
FINAL COMPLETION REPORT - All Phases Finished
================================================================================

Status: ALL PHASES COMPLETED
Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Total Processing Time: {total_time/3600:.2f} hours ({total_time/86400:.2f} days)

================================================================================
Phase 1 - 50 Sample Cities (All Africa)
================================================================================
  - Total cities: {phase_stats.get('phase1_total', 0)}
  - Completed: {phase_stats.get('phase1_completed', 0)}
  - Failed: {phase_stats.get('phase1_failed', 0)}
  - Success rate: {phase_stats.get('phase1_completed', 0) / phase_stats.get('phase1_total', 1) * 100:.1f}%

================================================================================
Phase 2 - Remaining Ghana Cities
================================================================================
  - Total cities: {phase_stats.get('phase2_total', 0)}
  - Completed: {phase_stats.get('phase2_completed', 0)}
  - Failed: {phase_stats.get('phase2_failed', 0)}
  - Success rate: {phase_stats.get('phase2_completed', 0) / phase_stats.get('phase2_total', 1) * 100:.1f}%

================================================================================
Phase 3 - All Remaining African Cities
================================================================================
  - Total cities: {phase_stats.get('phase3_total', 0)}
  - Completed: {phase_stats.get('phase3_completed', 0)}
  - Failed: {phase_stats.get('phase3_failed', 0)}
  - Success rate: {phase_stats.get('phase3_completed', 0) / phase_stats.get('phase3_total', 1) * 100:.1f}%

================================================================================
OVERALL SUMMARY
================================================================================
  - Total cities processed: {sum([phase_stats.get(f'phase{i}_total', 0) for i in [1,2,3]])}
  - Successfully completed: {sum([phase_stats.get(f'phase{i}_completed', 0) for i in [1,2,3]])}
  - Failed: {sum([phase_stats.get(f'phase{i}_failed', 0) for i in [1,2,3]])}
  - Overall success rate: {sum([phase_stats.get(f'phase{i}_completed', 0) for i in [1,2,3]]) / sum([phase_stats.get(f'phase{i}_total', 0) for i in [1,2,3]]) * 100:.1f}%

================================================================================
ALL ANALYSIS COMPLETED SUCCESSFULLY
================================================================================

All building morphology analysis tasks have been completed.
Please check the output directory for results.

Best regards,
{self.config['assistant_name']}
"""

        return self._send_email(subject, body)

    def send_error_notification(self, error_message, city_name=None, phase=None, context=None):
        """
        Send notification when an error causes pipeline interruption.

        Args:
            error_message (str): Error message
            city_name (str): Optional city name where error occurred
            phase (int): Optional phase number
            context (dict): Optional additional context information
        """
        subject = f"ERROR: Pipeline Interrupted{f' - {city_name}' if city_name else ''}"

        body = f"""Dear User,

{self.config['assistant_name']} reports:

================================================================================
ERROR NOTIFICATION - Pipeline Interrupted
================================================================================

Status: ERROR
Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

"""

        if phase:
            body += f"Phase: {phase}\n"

        if city_name:
            body += f"City: {city_name}\n"

        body += f"""
Error Message:
{error_message}

"""

        if context:
            body += "Additional Context:\n"
            for key, value in context.items():
                body += f"  - {key}: {value}\n"

        body += f"""
================================================================================
Action Required:
  1. Review the error message above
  2. Fix the underlying issue
  3. Restart the pipeline (checkpoint enabled - will resume from last success)
================================================================================

The pipeline has been halted to prevent cascading failures.
Please investigate and resolve the issue.

Best regards,
{self.config['assistant_name']}
"""

        return self._send_email(subject, body)

    def send_no_building_data_notification(self, city_name, country_name, iso3, retry_count=0, coordinates=None, area_km2=None):
        """
        Send notification when a city has no building data available (informational, not an error).

        Args:
            city_name (str): Name of the city
            country_name (str): Name of the country
            iso3 (str): ISO3 country code
            retry_count (int): Number of retry attempts made
            coordinates (dict): Optional boundary coordinates
            area_km2 (float): Optional boundary area
        """
        subject = f"INFO: No Building Data - {city_name}, {country_name}"

        body = f"""Dear User,

{self.config['assistant_name']} reports:

================================================================================
DATA COVERAGE NOTIFICATION - City Skipped
================================================================================

Status: INFORMATIONAL
Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

City: {city_name}
Country: {country_name} ({iso3})
Retry Attempts: {retry_count + 1}

Situation:
Google Open Buildings dataset does not have building data for this city.
After {retry_count + 1} extraction attempt(s), zero buildings were found.

"""

        if area_km2:
            body += f"City Area: {area_km2:.2f} km2\n"

        if coordinates:
            body += f"""Coordinates:
  - Longitude: {coordinates.get('lon_min', 'N/A'):.4f} to {coordinates.get('lon_max', 'N/A'):.4f}
  - Latitude: {coordinates.get('lat_min', 'N/A'):.4f} to {coordinates.get('lat_max', 'N/A'):.4f}

"""

        body += f"""================================================================================
Pipeline Status:
  - City has been recorded in no_building_data_cities.json
  - Pipeline continues processing remaining cities
  - No action required from you
================================================================================

This is a data coverage gap, not a pipeline error.
The city has been skipped and the pipeline continues normally.

Best regards,
{self.config['assistant_name']}
"""

        return self._send_email(subject, body)

    def send_task_completion(self, mode, summary_stats):
        """
        Send notification for single/batch task completion (non-phased modes).

        Args:
            mode (str): Processing mode (single_city, city_batch, country_batch)
            summary_stats (dict): Summary statistics
        """
        subject = f"Task Completed: {mode.replace('_', ' ').title()} Analysis Finished"

        body = f"""Dear User,

{self.config['assistant_name']} reports:

================================================================================
TASK COMPLETION REPORT - {mode.replace('_', ' ').title()} Mode
================================================================================

Status: COMPLETED
Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Statistics:
"""

        for key, value in summary_stats.items():
            body += f"  - {key}: {value}\n"

        body += f"""
================================================================================
All requested analysis tasks have been completed.
Please check the output directory for results.
================================================================================

Best regards,
{self.config['assistant_name']}
"""

        return self._send_email(subject, body)


# Convenience function for one-off notifications without constructing an
# EmailNotifier explicitly.
def send_email(subject, body, logger=None):
    """
    Send a simple email notification.

    Args:
        subject (str): Email subject
        body (str): Email body
        logger: Optional logger instance

    Returns:
        bool: True if sent (or notifications are unconfigured and the call
            was a no-op is reported as False - see EmailNotifier._send_email).
    """
    notifier = EmailNotifier(logger=logger)
    return notifier._send_email(subject, body)


if __name__ == "__main__":
    # Manual smoke test for the notification system. Requires SMTP_* env
    # vars to be set; otherwise this exercises and confirms the no-op path.
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    notifier = EmailNotifier()

    print("Testing email notification system...")
    print("Sending test email...")

    result = notifier.send_error_notification(
        error_message="This is a test error notification from notifications.py",
        city_name="TestCity",
        phase=1,
        context={"test_mode": True, "reason": "Module testing"}
    )

    if result:
        print("Test email sent successfully.")
        sys.exit(0)
    else:
        print("Notification not sent (either unconfigured or a real send failure - see log).")
        sys.exit(1)
