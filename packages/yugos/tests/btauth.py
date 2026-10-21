"""Check hosted calls against Bittensor's btauth/1 verifier."""

import bittensor as bt


def verify_signed(request, key):
    """Require a valid unbound btauth/1 signature and return the raw body."""
    body = request.content
    caller = bt.http_auth.verify(
        request.headers,
        body,
        method=request.method,
        path=request.url.raw_path.decode(),
        self_hotkey_ss58=key.ss58_address,
        require_receiver=False,
        nonce_store=bt.http_auth.InMemoryNonceStore(),
    )
    assert caller.hotkey_ss58 == key.ss58_address
    assert bt.http_auth.HEADER_RECEIVER not in request.headers
    assert not any(name.lower().startswith("epistula") for name in request.headers)
    return body


def assert_unsigned(headers):
    """Wallet authentication must not ride along on storage or result polls."""
    names = [name.lower() for name in headers]
    assert not any(name.startswith("x-bittensor") or name.startswith("epistula") for name in names)
