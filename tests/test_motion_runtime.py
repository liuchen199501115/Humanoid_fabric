import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from motion_generation import runtime


class RuntimeTests(unittest.TestCase):
    def test_manifest_launch_preserves_arguments_and_avoids_loop(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root/'source-python'; source.touch()
            target = root/'runtime-python'; target.touch()
            packages = root/'site-packages'; packages.mkdir()
            manifest = root/'manifest.json'
            manifest.write_text(json.dumps(dict(source_executable=str(source), runtime_python=target.name,
                                                source_site_packages=str(packages))))
            with patch.object(runtime, 'PROJECT_ROOT', root), patch.object(runtime, 'RUNTIME_CONFIG', manifest):
                with patch.object(runtime.sys, 'executable', str(source)), patch.object(runtime.sys, 'argv', ['train', '--epochs', '100']), patch.object(runtime.os, 'execve') as execute:
                    runtime.maybe_reexec()
                    executable, command, env = execute.call_args.args
                    self.assertEqual(executable, str(target))
                    self.assertEqual(command[-2:], ['--epochs', '100'])
                    self.assertIn(str(packages), env['PYTHONPATH'])
                    self.assertEqual(env['PYTHONNOUSERSITE'], '1')
                with patch.object(runtime.sys, 'executable', str(target)), patch.object(runtime.os, 'execve') as execute:
                    runtime.maybe_reexec()
                    execute.assert_not_called()
                target.unlink()
                with patch.object(runtime.sys, 'executable', str(source)):
                    with self.assertRaisesRegex(RuntimeError, 'incomplete'):
                        runtime.maybe_reexec()

    def test_no_local_manifest_does_not_change_other_installations(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(runtime, 'RUNTIME_CONFIG', Path(tmp)/'missing'), patch.object(runtime.os, 'execve') as execute:
                runtime.maybe_reexec()
                execute.assert_not_called()


if __name__ == '__main__':
    unittest.main()
