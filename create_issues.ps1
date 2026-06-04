[Environment]::SetEnvironmentVariable("GITHUB_TOKEN", $null, "Process")

$issues = @(
  @{
    Title = "Critical: Stored XSS via Missing `bleach` Dependency & Weak Regex Fallback"
    Body = "## Description`n`nThe HTML sanitization logic in `/api/upload` (inside `backend/main.py`) attempts to import the `bleach` library to sanitize product titles and descriptions. However, `bleach` is missing from `requirements.txt`.`n`nWhen `bleach` fails to import, the code falls back to a weak custom regex:`n````python`nre.sub(r'<[^>]*>', '', str(value))`n`````n`nThis regex only matches tags that have a closing `>` bracket. An attacker can bypass this by uploading a CSV with an unclosed HTML tag, such as:`n````html`n<img src=x onerror=alert(1) `n`````nBecause the regex fails to match, the payload is persisted to the database. When the frontend renders this data, the browser automatically closes the tag and executes the malicious JavaScript, leading to a Stored Cross-Site Scripting (XSS) vulnerability.`n`n## Impact`nAn attacker can execute arbitrary JavaScript in the browsers of users viewing the infected products, potentially stealing session tokens or performing actions on their behalf.`n`n## Recommended Fix`n1. Add `bleach` to `requirements.txt`.`n2. Remove the insecure regex fallback or replace it with a robust HTML parser (e.g., `html.escape` if full HTML is not needed)."
  },
  @{
    Title = "Critical: Fail-Open Authentication in GitHub Webhook Signature Verification"
    Body = "## Description`n`nThe `_verify_github_signature` function in `backend/main.py` is responsible for authenticating incoming GitHub webhooks by verifying the `X-Hub-Signature-256` header.`n`nHowever, the function contains a fail-open flaw:`n````python`ndef _verify_github_signature(request_body: bytes, signature_header: str | None) -> None:`n    secret = os.environ.get('GITHUB_WEBHOOK_SECRET', '').strip()`n    if not secret:`n        return`n`````nIf the `GITHUB_WEBHOOK_SECRET` environment variable is unset (e.g., due to misconfiguration in production), the function returns early without raising any authentication errors. This completely bypasses the signature verification.`n`n## Impact`nAn unauthenticated attacker can spoof webhook payloads and force the server to execute automated GitHub API actions (via `apply_github_actions`), potentially using the server's `GITHUB_TOKEN` to spam or modify labels/comments on arbitrary repositories.`n`n## Recommended Fix`nChange the logic to fail-closed. If the secret is not configured, the endpoint should raise an `HTTPException(500)` or reject all requests with a `403/401` status."
  },
  @{
    Title = "Critical: Denial of Service (DoS) via Infinite Rate Limiting Map Memory Leak"
    Body = "## Description`n`nThe rate limiting logic in `backend/main.py` (used in `/api/search` and `_apply_rate_limit`) tracks request timestamps in an in-memory dictionary keyed by the `client_ip`:`n````python`nbucket = _rate_limit_buckets.setdefault(client_ip, {'timestamps': []})`nbucket['timestamps'] = [t for t in bucket['timestamps'] if now - t < 60]`n`````nWhile the list of timestamps is pruned for entries older than 60 seconds, the `client_ip` keys themselves are **never removed** from the `_rate_limit_buckets` dictionary.`n`n## Impact`nAn attacker can flood the server with requests from spoofed IPs (if deployed behind a proxy that trusts `X-Forwarded-For` without validation) or via a distributed botnet. Every unique IP address adds a permanent entry to the dictionary, eventually exhausting the server's memory and crashing the application (OOM DoS).`n`n## Recommended Fix`nImplement a garbage collection routine to remove dictionary keys when their associated `timestamps` list becomes empty, or use an established rate-limiting library (like `slowapi`) backed by Redis."
  }
)

foreach ($issue in $issues) {
  $title = $issue.Title
  $body = $issue.Body
  gh issue create --repo "leonagoel/hybrid-recommender" --title $title --body $body
  Start-Sleep -Seconds 2
}
