#!/usr/bin/env python3
# setup_users.py
"""
═══════════════════════════════════════════════════════════════════
SIGNALBOT USER SETUP — Run this ONCE to initialize user system
═══════════════════════════════════════════════════════════════════

WHAT THIS DOES:
  1. Creates the users/ directory structure
  2. Creates your admin account (god mode)
  3. Migrates your existing SignalBot data into your account
  4. Optionally creates accounts for your kids

RUN:
  python3 setup_users.py

AFTER RUNNING:
  Your data files stay in the root directory as backups.
  The ACTIVE copies are now in users/adam/ (or whatever you pick).
  You can delete the root copies later once you're sure it works.
"""

from user_manager import get_user_manager


def main():
    print("=" * 60)
    print("SIGNALBOT USER SYSTEM SETUP")
    print("=" * 60)
    print()

    mgr = get_user_manager()

    # Check if already set up
    if mgr.get_user_count() > 0:
        print(f"[INFO] Registry already has {mgr.get_user_count()} user(s):")
        for u in mgr.get_all_users():
            print(f"  - {u.display_name} ({u.username}) [{u.role}]")
        print()

        choice = input("Add another user? (y/n): ").strip().lower()
        if choice != "y":
            print("Done.")
            return

        # Add a user
        name = input("Display name: ").strip()
        if not name:
            print("No name entered. Aborting.")
            return

        passphrase = input("Passphrase: ").strip()
        if not passphrase:
            print("No passphrase entered. Aborting.")
            return

        role = input("Role (user/admin) [user]: ").strip().lower() or "user"
        if role not in ("user", "admin"):
            role = "user"

        profile = mgr.create_user(name, passphrase, role=role)
        if profile:
            print(f"\n✓ Created {profile.display_name} ({profile.username}) as {profile.role}")
            print(f"  Data directory: {profile.data_dir}")
        else:
            print("\n✗ Failed to create user. Username may already exist.")
        return

    # ═══ FIRST RUN ═══
    print("No users found. Let's set up your admin account.\n")

    # Admin account
    print("── STEP 1: Your Admin Account (God Mode) ──")
    print("This will be your main account with full access.\n")

    admin_name = input("Your display name [Adam]: ").strip() or "Adam"
    admin_pass = input("Pick a passphrase: ").strip()
    while not admin_pass:
        print("You need a passphrase. Even a simple one is fine.")
        admin_pass = input("Pick a passphrase: ").strip()

    admin = mgr.setup_first_admin(admin_name, admin_pass)
    if not admin:
        print("\n✗ Failed to create admin account. Check errors above.")
        return

    print(f"\n✓ Admin account created: {admin.display_name} ({admin.username})")
    print(f"  Data directory: {admin.data_dir}")
    print(f"  Your existing SignalBot data has been migrated.\n")

    # Kid accounts
    print("── STEP 2: Family Accounts (Optional) ──")
    print("Create accounts for your kids? They each get their own")
    print("SignalBot memory space. You can see everything in god mode.\n")

    while True:
        name = input("Add a user (or press Enter to skip): ").strip()
        if not name:
            break

        passphrase = input(f"Passphrase for {name}: ").strip()
        if not passphrase:
            print("Skipped (no passphrase).\n")
            continue

        profile = mgr.create_user(name, passphrase, role="user")
        if profile:
            print(f"  ✓ Created {profile.display_name} ({profile.username})")
            print(f"    Data directory: {profile.data_dir}\n")
        else:
            print(f"  ✗ Failed. Username might be taken.\n")

    # Summary
    print("\n" + "=" * 60)
    print("SETUP COMPLETE")
    print("=" * 60)
    print(f"\nTotal users: {mgr.get_user_count()}")
    for u in mgr.get_all_users():
        print(f"  [{u.role.upper():6s}] {u.display_name} ({u.username})")

    print(f"\nUser data lives in: {admin.data_dir.parent}/")
    print(f"Registry file: user_registry.json")
    print()
    print("NEXT STEPS:")
    print("  1. Your existing data was COPIED, not moved.")
    print("     Originals are still in the root directory as backup.")
    print("  2. Run the web app: python3 app.py")
    print("  3. Log in with your admin account.")
    print()


if __name__ == "__main__":
    main()
