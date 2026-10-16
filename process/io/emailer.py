import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

SMTP_HOST = os.getenv("SMTP_HOST")
SMTP_PORT = int(os.getenv("SMTP_PORT", "465"))
SMTP_USER = os.getenv("SMTP_USER")
SMTP_PASS = os.getenv("SMTP_PASS")

# 🔥 O teu email BCC (não visível para o utilizador)
ADMIN_BCC = "mario.cesar@navarro.pt"   # <-- mete aqui o teu email


def _send(user_email, subject, message):
    """
    Envia um email em texto simples para o utilizador, com o ADMIN_BCC em
    cópia invisível. Devolve True/False. Nunca levanta exceção (um email
    falhado não pode estragar o processamento), mas o erro fica no log.
    """
    if not user_email:
        return False

    msg = MIMEMultipart()
    msg["From"] = SMTP_USER
    msg["To"] = user_email
    msg["Subject"] = subject

    # ⚠️ Não meter msg["Bcc"] — mantém o BCC invisível
    msg.attach(MIMEText(message, "plain"))

    try:
        with smtplib.SMTP_SSL(SMTP_HOST, SMTP_PORT) as server:
            server.login(SMTP_USER, SMTP_PASS)

            # 🔥 Enviar para o utilizador + BCC
            server.sendmail(
                SMTP_USER,
                [user_email, ADMIN_BCC],   # <-- BCC aqui
                msg.as_string()
            )
        return True
    except Exception as e:
        print(f"[email] falhou o envio ({subject!r}): {e!r}", flush=True)
        return False


def send_email(user_email, user_name, page_url):
    """Email final: o vídeo está pronto."""
    subject = "O seu vídeo anotado está pronto!"
    message = f"""
Hello {user_name or "Utilizador"},

Your video has been successfully processed and wil remain available for the next 24 hours.

You can view it here:
{page_url}

Thank you for using our service and for helping 7MOTION continue improving and developing its product

Best regards,
Mário César Navarro 
7MOTION"""

    return _send(user_email, subject, message)


def send_started_email(user_email, user_name, video_id=None):
    """Email inicial: o vídeo foi recebido e começou a ser processado."""
    subject = "O seu vídeo começou a ser processado"
    ref = f"\nReference: {video_id}\n" if video_id else ""
    message = f"""
Hello {user_name or "Utilizador"},

We have received your video and its processing has just started.

You will receive another email as soon as it is ready. Depending on the length of the video, this can take a while, so there is no need to wait on the page.
{ref}
Thank you for using our service and for helping 7MOTION continue improving and developing its product

Best regards,
Mário César Navarro 
7MOTION"""

    return _send(user_email, subject, message)


def send_failed_email(user_email, user_name, video_id=None):
    """Email de erro: não foi possível processar o vídeo."""
    subject = "Não foi possível processar o seu vídeo"
    ref = f"\nReference: {video_id}\n" if video_id else ""
    message = f"""
Hello {user_name or "Utilizador"},

Unfortunately we were not able to process your video. Our team has been notified and will look into it. We will contact you if we need anything else from you.
{ref}
We apologise for the inconvenience.

Best regards,
Mário César Navarro 
7MOTION"""

    return _send(user_email, subject, message)
