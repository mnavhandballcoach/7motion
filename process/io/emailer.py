import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

SMTP_HOST = os.getenv("SMTP_HOST")
SMTP_PORT = int(os.getenv("SMTP_PORT", "465"))
SMTP_USER = os.getenv("SMTP_USER")
SMTP_PASS = os.getenv("SMTP_PASS")

def send_email(user_email, user_name, video_url):
    if not user_email:
        return

    subject = "O seu vídeo anotado está pronto!"
    message = f"""
Olá {user_name or "Utilizador"},

O seu vídeo foi processado com sucesso.

Pode descarregar o vídeo anotado aqui:
{video_url}

Obrigado por utilizar o nosso serviço!
"""

    msg = MIMEMultipart()
    msg["From"] = SMTP_USER
    msg["To"] = user_email
    msg["Subject"] = subject
    msg.attach(MIMEText(message, "plain"))

    try:
        with smtplib.SMTP_SSL(SMTP_HOST, SMTP_PORT) as server:
            server.login(SMTP_USER, SMTP_PASS)
            server.sendmail(SMTP_USER, user_email, msg.as_string())
    except Exception:
        pass
