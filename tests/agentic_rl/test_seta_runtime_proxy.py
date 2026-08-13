from __future__ import annotations

import json
import os
import stat
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RENDER = ROOT / "deploy" / "workers" / "render_seta_runtime_proxy.py"
CONFIGURE = ROOT / "deploy" / "workers" / "configure_seta_runtime_proxy.py"
PROXY_UNIT = ROOT / "deploy" / "workers" / "lightrl-seta-runtime-proxy.service"
WORKER_DROPIN = ROOT / "deploy" / "workers" / "lightrl-pool53-runtime-proxy.conf"
WORKER_POOL_CONFIG = ROOT / "deploy" / "workers" / "lightrl-pool53-proxy.conf"


class SetaRuntimeProxyTest(unittest.TestCase):
    def test_rendered_upstream_credentials_are_root_only(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "tinyproxy.conf"
            env = dict(os.environ)
            env["HTTPS_PROXY"] = "http://alice:s3cret@proxy.example:3128"
            completed = subprocess.run(
                [
                    sys.executable,
                    str(RENDER),
                    "--output",
                    str(output),
                    "--user",
                    "probe-user",
                    "--group",
                    "probe-group",
                    "--log-file",
                    str(Path(directory) / "proxy.log"),
                ],
                env=env,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertEqual(completed.stdout, "")
            self.assertNotIn("s3cret", completed.stderr)
            self.assertEqual(stat.S_IMODE(output.stat().st_mode), 0o600)
            config = output.read_text(encoding="utf-8")
            self.assertIn("Listen 172.17.0.1", config)
            self.assertIn("Allow 172.16.0.0/12", config)
            self.assertIn("Allow 192.168.0.0/16", config)
            self.assertIn("User probe-user", config)
            self.assertIn("Group probe-group", config)
            self.assertIn(
                "Upstream http alice:s3cret@proxy.example:3128", config
            )

    def test_render_rejects_missing_authentication(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "tinyproxy.conf"
            env = dict(os.environ)
            env["HTTPS_PROXY"] = "http://proxy.example:3128"
            completed = subprocess.run(
                [sys.executable, str(RENDER), "--output", str(output)],
                env=env,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertNotEqual(completed.returncode, 0)
            self.assertFalse(output.exists())

    def test_docker_proxy_config_is_scoped_and_preserves_existing_fields(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / ".docker-seta" / "config.json"
            config.parent.mkdir()
            config.write_text(
                json.dumps({"auths": {"registry.example": {"auth": "opaque"}}}),
                encoding="utf-8",
            )
            completed = subprocess.run(
                [
                    sys.executable,
                    str(CONFIGURE),
                    "--docker-config",
                    str(config),
                ],
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertEqual(stat.S_IMODE(config.stat().st_mode), 0o600)
            payload = json.loads(config.read_text(encoding="utf-8"))
            self.assertEqual(
                payload["auths"], {"registry.example": {"auth": "opaque"}}
            )
            proxy = payload["proxies"]["default"]
            self.assertEqual(proxy["httpProxy"], "http://172.17.0.1:3129")
            self.assertEqual(proxy["httpsProxy"], "http://172.17.0.1:3129")
            self.assertIn("192.168.0.0/16", proxy["noProxy"])
            self.assertNotIn("@", json.dumps(proxy))

    def test_systemd_scope_does_not_change_root_docker_config(self) -> None:
        unit = PROXY_UNIT.read_text(encoding="utf-8")
        dropin = WORKER_DROPIN.read_text(encoding="utf-8")
        pool_config = WORKER_POOL_CONFIG.read_text(encoding="utf-8")
        configure = CONFIGURE.read_text(encoding="utf-8")
        self.assertIn("Listen", RENDER.read_text(encoding="utf-8"))
        self.assertIn("lightrl-seta-runtime-proxy.service", dropin)
        self.assertIn("DOCKER_CONFIG=/home/puyuan/lightrl_worker/.docker-seta", dropin)
        self.assertIn("SKIP_PROXY_ENV=1", dropin)
        self.assertIn("HTTP_PROXY=http://172.17.0.1:3129", dropin)
        self.assertIn("http_proxy=http://172.17.0.1:3129", dropin)
        self.assertIn("EnvironmentFile=/etc/seta_build_proxy.env", unit)
        self.assertNotIn("EnvironmentFile=/etc/seta_build_proxy.env", pool_config)
        self.assertNotIn("@", dropin)
        self.assertNotIn("/root/.docker/config.json", configure)
        self.assertNotIn("/root/.docker/config.json", dropin)


if __name__ == "__main__":
    unittest.main()
