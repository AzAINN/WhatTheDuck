"""
Lightweight wrapper around `classiq.authenticate()` that avoids re-registering
a device when a refresh token is already stored. Pass `--overwrite` if you
explicitly want to rotate the token.
"""

import argparse
import classiq
from classiq._internals.authentication import password_manager as pm


def _has_refresh_token() -> bool:
    """Check the first supported password manager for an existing refresh token."""
    for manager_cls in (pm.KeyringPasswordManager, pm.FilePasswordManager):
        if not manager_cls.is_supported():
            continue
        manager = manager_cls()
        if manager.refresh_token:
            return True
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Authenticate with Classiq")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate credentials even if a refresh token already exists.",
    )
    args = parser.parse_args()

    if _has_refresh_token() and not args.overwrite:
        print(
            "Existing Classiq refresh token found; skipping device registration. "
            "Run with --overwrite to rotate it if needed."
        )
        return

    classiq.authenticate(overwrite=args.overwrite)
    print("Classiq authentication complete.")


if __name__ == "__main__":
    main()

    
