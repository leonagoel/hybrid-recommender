import os
import unittest
from unittest.mock import MagicMock, patch

from src.data import db


class TestSupabaseGateway(unittest.TestCase):
    def setUp(self):
        # Cache current environment variables to restore them after each test
        self.original_env = {
            "SUPABASE_URL": os.environ.get("SUPABASE_URL"),
            "SUPABASE_ANON_KEY": os.environ.get("SUPABASE_ANON_KEY"),
            "SUPABASE_SERVICE_KEY": os.environ.get("SUPABASE_SERVICE_KEY"),
        }
        # Reset the module-level singletons to ensure test isolation
        db._client = None
        db._admin_client = None

    def tearDown(self):
        # Restore original environment variables
        for key, value in self.original_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        # Reset module-level singletons
        db._client = None
        db._admin_client = None

    @patch("src.data.db.create_client")
    def test_get_supabase_success(self, mock_create_client):
        """Test successful initialization and caching of get_supabase (anon client)."""
        os.environ["SUPABASE_URL"] = "https://mock-project.supabase.co"
        os.environ["SUPABASE_ANON_KEY"] = "mock-anon-key"

        mock_client_instance = MagicMock()
        mock_create_client.return_value = mock_client_instance

        # First call: initializes client
        client1 = db.get_supabase()
        self.assertIs(client1, mock_client_instance)
        mock_create_client.assert_called_once_with("https://mock-project.supabase.co", "mock-anon-key")

        # Second call: returns cached singleton without re-creating
        client2 = db.get_supabase()
        self.assertIs(client2, mock_client_instance)
        mock_create_client.assert_called_once()

    @patch("src.data.db.create_client")
    def test_get_supabase_admin_success(self, mock_create_client):
        """Test successful initialization and caching of get_supabase_admin (admin client)."""
        os.environ["SUPABASE_URL"] = "https://mock-project.supabase.co"
        os.environ["SUPABASE_SERVICE_KEY"] = "mock-service-key"

        mock_admin_instance = MagicMock()
        mock_create_client.return_value = mock_admin_instance

        # First call: initializes client
        client1 = db.get_supabase_admin()
        self.assertIs(client1, mock_admin_instance)
        mock_create_client.assert_called_once_with("https://mock-project.supabase.co", "mock-service-key")

        # Second call: returns cached singleton without re-creating
        client2 = db.get_supabase_admin()
        self.assertIs(client2, mock_admin_instance)
        mock_create_client.assert_called_once()

    def test_get_supabase_missing_env(self):
        """Test that RuntimeError is raised when anon client env vars are missing."""
        os.environ.pop("SUPABASE_URL", None)
        os.environ.pop("SUPABASE_ANON_KEY", None)

        with self.assertRaises(RuntimeError) as ctx:
            db.get_supabase()
        self.assertIn("SUPABASE_URL and SUPABASE_ANON_KEY must be set in .env", str(ctx.exception))

    def test_get_supabase_admin_missing_env(self):
        """Test that RuntimeError is raised when admin client env vars are missing."""
        os.environ.pop("SUPABASE_URL", None)
        os.environ.pop("SUPABASE_SERVICE_KEY", None)

        with self.assertRaises(RuntimeError) as ctx:
            db.get_supabase_admin()
        self.assertIn("SUPABASE_URL and SUPABASE_SERVICE_KEY must be set in .env", str(ctx.exception))

    @patch("src.data.db.create_client")
    def test_get_supabase_admin_creation_failure(self, mock_create_client):
        """Test that RuntimeError is raised when create_client raises an exception."""
        os.environ["SUPABASE_URL"] = "https://mock-project.supabase.co"
        os.environ["SUPABASE_SERVICE_KEY"] = "mock-service-key"

        mock_create_client.side_effect = Exception("Connection Refused")

        with self.assertRaises(RuntimeError) as ctx:
            db.get_supabase_admin()
        self.assertIn("Failed to initialise Supabase admin client: Connection Refused", str(ctx.exception))

    @patch("src.data.db.create_client")
    def test_mocked_empty_result_set(self, mock_create_client):
        """Test empty result set handling on mocked table query chain."""
        os.environ["SUPABASE_URL"] = "https://mock-project.supabase.co"
        os.environ["SUPABASE_ANON_KEY"] = "mock-anon-key"

        # Mock the query chain: client.table().select().execute() -> response
        mock_client = MagicMock()
        mock_table = MagicMock()
        mock_select = MagicMock()
        mock_response = MagicMock()

        mock_create_client.return_value = mock_client
        mock_client.table.return_value = mock_table
        mock_table.select.return_value = mock_select
        mock_select.execute.return_value = mock_response

        # Return empty list from execute
        mock_response.data = []

        client = db.get_supabase()
        result = client.table("products").select("*").execute()

        # Verify calls and empty response
        mock_client.table.assert_called_with("products")
        mock_table.select.assert_called_with("*")
        mock_select.execute.assert_called_once()
        self.assertEqual(result.data, [])

    @patch("src.data.db.create_client")
    def test_mocked_network_timeout(self, mock_create_client):
        """Test query chain behavior when simulating network timeouts or exception side effects."""
        os.environ["SUPABASE_URL"] = "https://mock-project.supabase.co"
        os.environ["SUPABASE_ANON_KEY"] = "mock-anon-key"

        mock_client = MagicMock()
        mock_table = MagicMock()
        mock_select = MagicMock()

        mock_create_client.return_value = mock_client
        mock_client.table.return_value = mock_table
        mock_table.select.return_value = mock_select

        # Simulate network timeout exception from execute()
        mock_select.execute.side_effect = Exception("Gateway Timeout")

        client = db.get_supabase()
        
        with self.assertRaises(Exception) as ctx:
            client.table("products").select("*").execute()
            
        self.assertEqual(str(ctx.exception), "Gateway Timeout")
        mock_select.execute.assert_called_once()
