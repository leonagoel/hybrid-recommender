import os
from dotenv import load_dotenv


def check_env_variables() -> None:
    """Check required environment variables."""
    load_dotenv()

    required_vars = [
        "SUPABASE_URL",
        "SUPABASE_ANON_KEY",
        "SUPABASE_SERVICE_KEY",
    ]

    missing_vars = []

    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)

    if missing_vars:
        print("❌ Missing environment variables:")
        for var in missing_vars:
            print(f" - {var}")
    else:
        print("✅ Environment setup looks good")


if __name__ == "__main__":
    check_env_variables()