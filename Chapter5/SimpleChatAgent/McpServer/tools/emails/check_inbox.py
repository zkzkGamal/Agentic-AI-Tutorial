from server import mcp
import smtplib , logging , environ , pathlib
import imaplib
import email



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
def check_inbox(limit: int = 5) -> list:
    """Fetch recent emails from inbox. Returns list of dicts with subject, from, body snippet."""
    try:
        with imaplib.IMAP4_SSL(EMAIL_HOST, EMAIL_PORT) as server:
            if EMAIL_USE_SSL:
                server.starttls()
            server.login(MAIL_USERNAME, EMAIL_HOST_PASSWORD)
            server.select("inbox")
            _ , inbox = server.search(None, "ALL")
            emails = []
            for num in inbox.split()[-limit:]:
                _ , data = server.fetch(num, "(RFC822)")
                email = email.message_from_bytes(data[0][1])
                emails.append({
                    "subject": email.get("Subject"),
                    "from": email.get("From"),
                    "body": email.get("Body"),
                })
            server.close()
            return emails
    except Exception as e:
        logger.error(f"Failed to check inbox: {e}")
        return []