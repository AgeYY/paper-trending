from unittest.mock import Mock

import pytest
import requests

from paper_atlas import openreview_auth_check as check


@pytest.fixture
def client():
    obj = Mock()
    obj.get_notes.return_value = [object()]
    return obj


@pytest.fixture
def credentials(monkeypatch):
    monkeypatch.setattr("builtins.input", lambda _: "private@example.org")
    monkeypatch.setattr(check.getpass, "getpass", lambda _: "secret-password")


def test_authenticated_success(client, credentials, capsys):
    assert check.run_check(client) == 0
    client.login_user.assert_called_once_with("private@example.org", "secret-password")
    client.get_notes.assert_called_once_with(
        invitation="ICLR.cc/2025/Conference/-/Submission", limit=1
    )
    client.session.close.assert_called_once()
    text = capsys.readouterr().out
    assert "SUCCESS" in text
    assert "secret-password" not in text and "private@example.org" not in text


@pytest.mark.parametrize("at_login", [True, False])
def test_challenge_stops_and_redacts(client, credentials, capsys, at_login):
    error = Exception({"name": "ChallengeRequiredError", "message": "private-token"})
    (client.login_user if at_login else client.get_notes).side_effect = error
    assert check.run_check(client) == 1
    text = capsys.readouterr().out
    assert "ChallengeRequiredError" in text and "private-token" not in text
    assert client.get_notes.call_count == (0 if at_login else 1)
    client.session.close.assert_called_once()


def test_anonymous_never_logs_in(client):
    assert check.run_check(client, anonymous=True) == 0
    client.login_user.assert_not_called()


def test_empty_result_not_success(client):
    client.get_notes.return_value = []
    assert check.run_check(client, anonymous=True) == 2


def test_generic_error_redacted(client, capsys):
    client.get_notes.side_effect = Exception("Bearer secret-token")
    assert check.run_check(client, anonymous=True) == 1
    assert "secret-token" not in capsys.readouterr().out


def test_timeout_is_not_credential_failure(client, capsys):
    client.get_notes.side_effect = requests.exceptions.Timeout("private-details")
    assert check.run_check(client, anonymous=True) == 1
    assert "Network timeout" in capsys.readouterr().out


def test_network_is_bounded(monkeypatch):
    request = Mock(return_value="ok")
    monkeypatch.setattr(requests.Session, "request", request)
    with check.DiagnosticSession() as session:
        assert session.get("https://api2.openreview.net/notes") == "ok"
        assert request.call_args.kwargs["timeout"] == (10, 30)
        assert session.get_adapter("https://").max_retries.total == 0


def test_no_inherited_login(monkeypatch):
    import openreview

    monkeypatch.setenv("OPENREVIEW_USERNAME", "inherited-user")
    monkeypatch.setenv("OPENREVIEW_PASSWORD", "inherited-password")
    login = Mock()
    monkeypatch.setattr(openreview.api.OpenReviewClient, "login_user", login)
    obj = check.make_client()
    login.assert_not_called()
    assert check.os.environ["OPENREVIEW_PASSWORD"] == "inherited-password"
    obj.session.close()


def test_noninteractive_refused(monkeypatch):
    monkeypatch.setattr(check.sys.stdin, "isatty", lambda: False)
    assert check.main([]) == 2


def test_download_reuses_authenticated_client(client, credentials):
    download = Mock(return_value=0)
    assert check.run_check(client, after_success=download) == 0
    download.assert_called_once_with(client)
    client.login_user.assert_called_once()
    client.session.close.assert_called_once()
