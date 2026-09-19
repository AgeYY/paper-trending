# Security and privacy

Never put API keys in browser code, a public repository, screenshots or issues.
The full Python application reads the user's own OpenAI key from an environment
variable or an owner-readable local configuration file. AI is explicitly opt-in.
The GitHub Pages demo contains no key and makes no AI calls.

If a credential has been shared publicly or in a conversation, revoke it at its
provider and create a replacement. Removing a file or rewriting Git history is
not a substitute for revocation. Secret-pattern scans reduce risk but cannot prove
absence of every secret. Review both source and generated artifacts before release.

The Python HTTP server intentionally binds to loopback only. It is not an
authenticated production backend; do not expose it with a public tunnel. Local
AI requests send the entered description to OpenAI under that provider's policy
and consume the user's API budget. No application search history is persisted.

For a potential vulnerability, use GitHub's private vulnerability reporting when
available rather than posting exploit details or secrets in a public issue.
