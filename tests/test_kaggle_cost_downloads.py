"""Download failure tests use tiny in-memory payloads, never network or weights."""
import hashlib
import http.client
import io
from pathlib import Path
import ssl
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from src.kaggle_cost_downloads import download_verified, matches, reuse_verified


class DownloadTests(unittest.TestCase):
    payload = b"original-test-checkpoint"
    digest = hashlib.sha256(payload).hexdigest()
    urls = ["https://origin.example/model.pt", "https://cdn.example/model.pt"]

    def test_ssl_failure_uses_other_origin_and_checks_original_hash(self):
        with tempfile.TemporaryDirectory() as folder, \
                patch("src.kaggle_cost_downloads.urllib.request.urlopen",
                      side_effect=[ssl.SSLEOFError("TLS EOF"), io.BytesIO(self.payload)]) as open_url, \
                patch("sys.stdout", new=io.StringIO()):
            target = Path(folder) / "model.pt"
            download_verified(self.urls, target, self.digest, attempts=1)
            self.assertEqual(target.read_bytes(), self.payload)
            self.assertEqual([c.args[0].full_url for c in open_url.call_args_list], self.urls)
            self.assertFalse(target.with_name("model.pt.part").exists())

    def test_incomplete_response_retries_cleanly(self):
        class BrokenResponse(io.BytesIO):
            def read(self, size=-1):
                if self.tell():
                    raise http.client.IncompleteRead(b"bad")
                return super().read(size)
        with tempfile.TemporaryDirectory() as folder, \
                patch("src.kaggle_cost_downloads.urllib.request.urlopen",
                      side_effect=[BrokenResponse(b"partial"), io.BytesIO(self.payload)]), \
                patch("sys.stdout", new=io.StringIO()):
            target = Path(folder) / "model.pt"
            download_verified(self.urls, target, self.digest, attempts=1)
            self.assertEqual(target.read_bytes(), self.payload)

    def test_successful_http_with_wrong_bytes_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder, \
                patch("src.kaggle_cost_downloads.urllib.request.urlopen", return_value=io.BytesIO(b"wrong")), \
                patch("src.kaggle_cost_downloads.shutil.which", return_value=None), \
                patch("sys.stdout", new=io.StringIO()):
            target = Path(folder) / "model.pt"
            with self.assertRaisesRegex(RuntimeError, "SHA256 mismatch"):
                download_verified(self.urls[:1], target, self.digest, attempts=1)
            self.assertFalse(target.exists())
            self.assertFalse(target.with_name("model.pt.part").exists())

    def test_curl_fallback_preserves_tls_checks(self):
        def curl(command, **kwargs):
            self.assertNotIn("--insecure", command)
            self.assertNotIn("-k", command)
            self.assertTrue(kwargs["check"])
            Path(command[command.index("--output") + 1]).write_bytes(self.payload)
            return subprocess.CompletedProcess(command, 0)
        with tempfile.TemporaryDirectory() as folder, \
                patch("src.kaggle_cost_downloads.urllib.request.urlopen", side_effect=ssl.SSLEOFError("EOF")), \
                patch("src.kaggle_cost_downloads.shutil.which", return_value="curl"), \
                patch("src.kaggle_cost_downloads.subprocess.run", side_effect=curl) as run, \
                patch("sys.stdout", new=io.StringIO()):
            target = Path(folder) / "model.pt"
            download_verified(self.urls, target, self.digest, attempts=1)
            self.assertEqual(target.read_bytes(), self.payload)
            run.assert_called_once()

    def test_verified_target_is_never_downloaded_again(self):
        with tempfile.TemporaryDirectory() as folder, \
                patch("src.kaggle_cost_downloads.urllib.request.urlopen") as open_url:
            target = Path(folder) / "model.pt"
            target.write_bytes(self.payload)
            self.assertEqual(download_verified(self.urls, target, self.digest), target)
            open_url.assert_not_called()

    def test_reuse_skips_partial_files_and_keeps_the_completed_source(self):
        with tempfile.TemporaryDirectory() as folder, patch("sys.stdout", new=io.StringIO()):
            root = Path(folder)
            bad, good = root / "partial.pt", root / "completed.pt"
            bad.write_bytes(b"partial")
            good.write_bytes(self.payload)
            target = root / "new_bundle/model.pt"
            self.assertTrue(reuse_verified(target, [bad, good], self.digest))
            self.assertEqual(target.read_bytes(), self.payload)
            self.assertEqual(good.read_bytes(), self.payload)
            self.assertFalse((target.parent / "model.pt.part").exists())

    def test_reused_small_config_is_checked_against_git_blob(self):
        with tempfile.TemporaryDirectory() as folder, patch("sys.stdout", new=io.StringIO()):
            root = Path(folder)
            source = root / "config.json"
            data = b'{"projection_dim":512}'
            source.write_bytes(data)
            blob = hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()
            target = root / "copy/config.json"
            self.assertTrue(reuse_verified(target, [source], blob_id=blob))
            self.assertTrue(matches(target, blob_id=blob))


if __name__ == "__main__":
    unittest.main()
