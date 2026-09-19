"""Interactive, one-record OpenReview diagnostic; never persists credentials."""

import argparse
import getpass
import os
import sys
import warnings
from importlib.metadata import version

import requests


class DiagnosticSession(requests.Session):
    """Use requests' no-retry default and bounded network waits."""

    def request(self, method, url, **kwargs):
        kwargs.setdefault("timeout", (10, 30))
        return super().request(method, url, **kwargs)


def make_client(api_version=2):
    import openreview

    # Do not silently use credentials inherited from the shell. Construct without
    # login first, so the bounded session also applies to authentication and MFA.
    keys = ("OPENREVIEW_USERNAME", "OPENREVIEW_PASSWORD")
    saved = {key: os.environ.pop(key) for key in keys if key in os.environ}
    try:
        client = (openreview.api.OpenReviewClient(baseurl="https://api2.openreview.net")
                  if api_version == 2 else openreview.Client(baseurl="https://api.openreview.net"))
    finally:
        os.environ.update(saved)
    client.session.close()
    client.session = DiagnosticSession()
    return client


def report_error(exc, stage):
    """Print only known classifications, never raw server bodies or exceptions."""
    payload = exc.args[0] if exc.args and isinstance(exc.args[0], dict) else {}
    name = payload.get("name")
    print(f"FAILED during {stage}.")
    if name == "ChallengeRequiredError":
        print("ChallengeRequiredError: OpenReview still requires browser verification.")
        print("Stopped without retrying. Authentication does not necessarily clear this challenge.")
    elif isinstance(exc, requests.exceptions.Timeout):
        print("Network timeout. This does not establish whether authentication works.")
    elif name in {"AuthenticationError", "InvalidCredentialsError", "UnauthorizedError"}:
        print("Authentication was rejected. Check your credentials privately.")
    else:
        print("Request failed; raw error details are withheld to protect account information.")
    return 1


def run_check(client, anonymous=False, after_success=None):
    stage = "login"
    try:
        if anonymous:
            print("Anonymous diagnostic: no credentials will be requested or used.")
        else:
            email = input("OpenReview email: ").strip()
            # Fail closed if getpass cannot disable terminal echo.
            with warnings.catch_warnings():
                warnings.simplefilter("error", getpass.GetPassWarning)
                password = getpass.getpass("OpenReview password (hidden): ")
            if not email or not password:
                print("Email and password must both be nonempty. No login attempted.")
                return 1
            try:
                client.login_user(email, password)
            finally:
                del password
            print("Login completed. Testing metadata access separately...")

        stage = "fetching one ICLR 2025 submission"
        notes = client.get_notes(invitation="ICLR.cc/2025/Conference/-/Submission", limit=1)
        if not notes:
            print("Request succeeded but returned zero records; data access is not confirmed.")
            return 2
        print("SUCCESS: fetched one ICLR 2025 submission.")
        print("This confirms one-record access, not full-corpus coverage or bulk-download access.")
        if after_success is not None:
            stage = "downloading public conference metadata"
            return after_success(client)
        return 0
    except (EOFError, KeyboardInterrupt):
        print("\nCancelled. No credentials saved.")
        return 130
    except Exception as exc:
        return report_error(exc, stage)
    finally:
        client.session.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anonymous", action="store_true", help="Test without login or prompts")
    parser.add_argument("--download", action="store_true", help="After login/check, download public metadata in this same session")
    parser.add_argument("--with-decisions", action="store_true", help="Also retrieve public official decision replies (requires --download)")
    parser.add_argument("--years", nargs="+", type=int, default=list(range(2022, 2027)), choices=range(2022, 2027))
    parser.add_argument("--venues", nargs="+", choices=["iclr", "neurips", "icml"], default=["iclr", "neurips", "icml"])
    parser.add_argument("--output", help="New output directory; defaults to DATA_DIR/literature/openreview_<UTC timestamp>")
    args = parser.parse_args(argv)
    if args.with_decisions and not args.download:
        parser.error("--with-decisions requires --download")
    if not args.anonymous and not sys.stdin.isatty():
        print("Run in your interactive terminal; do not pipe credentials into this command.")
        return 2
    try:
        print(f"Official client: openreview-py {version('openreview-py')}")
        print("No credential files, session-token files, or raw error logs are written.")
        client = make_client()
    except Exception as exc:
        return report_error(exc, "client setup")
    callback = None
    if args.download:
        from paper_atlas.openreview_download import download_public_metadata

        callback = lambda active: download_public_metadata(active, args.output, args.venues, args.years,
                                                          with_decisions=args.with_decisions)
    return run_check(client, anonymous=args.anonymous, after_success=callback)


if __name__ == "__main__":
    raise SystemExit(main())
