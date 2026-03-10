from server import mcp
import smtplib , logging , environ , pathlib
from email.message import EmailMessage


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

base_path = pathlib.Path(__file__).parent.parent.parent
e = environ.Env()
e.read_env(str(base_path / ".env"))

# global email configuration
EMAIL_HOST = e('MAIL_HOST', default='smtp.example.com')
EMAIL_PORT = e('MAIL_PORT', default=587)
MAIL_USERNAME = e('MAIL_USERNAME',default=None)
EMAIL_HOST_PASSWORD = e('MAIL_PASSWORD', default=None)
EMAIL_USE_SSL = e.bool('MAIL_ENCRYPTION', default=True)
MAIL_FROM_NAME = e('MAIL_FROM_NAME', default='Zkaria Gamal')


@mcp.tool()
def send_email(subject: str, body: str, to_email: list[str]):
    """
        Send an email.
        args:
            subject: str
            body: str
            to_email: list[str]
        returns:
            bool
    """
    try:
        msg = EmailMessage()
        msg.set_content(body)
        msg['Subject'] = subject
        msg['From'] = f"{MAIL_FROM_NAME} <{MAIL_USERNAME}>"
        if EMAIL_USE_SSL:
            with smtplib.SMTP_SSL(EMAIL_HOST, EMAIL_PORT , timeout = 5)  as server:
                server.login(MAIL_USERNAME, EMAIL_HOST_PASSWORD)
                logger.info(f"mail data {MAIL_USERNAME}, and server logeded in")
                msg['To'] = ", ".join(to_email)
                server.send_message(msg)
        else:
            with smtplib.SMTP(EMAIL_HOST, EMAIL_PORT, timeout = 5) as server:
                server.starttls()
                server.login(MAIL_USERNAME, EMAIL_HOST_PASSWORD)
                msg['To'] = to_email
                server.send_message(msg)
        logger.info(f"Email sent successfully to {to_email}.")    
        return True
    except Exception as e:
        logger.error(f"Failed to send email: {e}")
        return False
