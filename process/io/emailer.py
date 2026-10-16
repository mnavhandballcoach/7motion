import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

SMTP_HOST = os.getenv("SMTP_HOST")
SMTP_PORT = int(os.getenv("SMTP_PORT", "465"))
SMTP_USER = os.getenv("SMTP_USER")
SMTP_PASS = os.getenv("SMTP_PASS")

# 🔥 Hidden BCC (admin copy)
ADMIN_BCC = "mario.cesar@navarro.pt"


def _send(user_email, subject, message_html):
    """
    Sends an HTML email to the user, with ADMIN_BCC in hidden copy.
    Returns True/False. Never raises exceptions.
    """
    if not user_email:
        return False

    msg = MIMEMultipart("alternative")
    msg["From"] = SMTP_USER
    msg["To"] = user_email
    msg["Subject"] = subject

    # Attach HTML body
    msg.attach(MIMEText(message_html, "html"))

    try:
        with smtplib.SMTP_SSL(SMTP_HOST, SMTP_PORT) as server:
            server.login(SMTP_USER, SMTP_PASS)

            server.sendmail(
                SMTP_USER,
                [user_email, ADMIN_BCC],  # hidden BCC
                msg.as_string()
            )
        return True
    except Exception as e:
        print(f"[email] failed to send ({subject!r}): {e!r}", flush=True)
        return False


def send_email(user_email, user_name, page_url):
    """Final email: video is ready."""
    subject = "Your annotated video is ready!"
    message = f"""
    <p>Hello {user_name or "User"},</p>

    <p>Your video has been successfully processed and will remain available for the next 24 hours.</p>

    <p>You can view it here:<br>
    <a href="{page_url}">{page_url}</a></p>

    <p>Thank you for using our service and for helping <b>7MOTION</b> continue improving and developing its product.</p>

    <p>Best regards,<br>
    Mário César Navarro<br>
    <b>7MOTION</b></p>
    """

    return _send(user_email, subject, message)


def send_started_email(user_email, user_name, video_id=None):
    """Initial email: video received and processing started."""
    subject = "Your video is now being processed"
    ref = f"<p>Reference: {video_id}</p>" if video_id else ""
    message = f"""
    <p>Hello {user_name or "User"},</p>

    <p>We have received your video and processing has just started.</p>

    <p>You will receive another email as soon as it is ready. Depending on the length of the video, this may take a while, so there is no need to stay on the page.</p>
    {ref}

    <p>Thank you for using our service and for helping <b>7MOTION</b> continue improving and developing its product.</p>

    <p>Best regards,<br>
    Mário César Navarro<br>
    <b>7MOTION</b></p>
    """

    return _send(user_email, subject, message)


def send_failed_email(user_email, user_name, video_id=None):
    """Error email: video could not be processed."""
    subject = "We could not process your video"
    ref = f"<p>Reference: {video_id}</p>" if video_id else ""
    message = f"""
    <p>Hello {user_name or "User"},</p>

    <p>Unfortunately, we were not able to process your video. Our team has been notified and will look into the issue. We will contact you if we need anything else from you.</p>
    {ref}

    <p>We apologise for the inconvenience.</p>

    <p>Best regards,<br>
    Mário César Navarro<br>
    <b>7MOTION</b></p>
    """

    return _send(user_email, subject, message)
