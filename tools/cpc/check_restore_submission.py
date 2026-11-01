"""Small end-to-end checks of archive restoration and refusal paths."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import zipfile
from restore_submission import restore


class RestoreChecks(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(); self.root=Path(self.tmp.name)
        self.archive=self.root/'sample.zip'; self.dest=self.root/'restored'
        self.dest.mkdir()

    def tearDown(self):
        self.tmp.cleanup()

    def bundle(self, name='data/value.bin', content=b'original', bad_hash=False):
        value='0'*64 if bad_hash else hashlib.sha256(content).hexdigest()
        with zipfile.ZipFile(self.archive,'w') as z:
            z.writestr(name,content)
            z.writestr('SHA256.json',json.dumps({name:value}))

    def test_exact_restore_and_idempotent_restore(self):
        self.bundle(); restore(self.archive,self.dest,'SHA256.json')
        restore(self.archive,self.dest,'SHA256.json')
        self.assertEqual((self.dest/'data/value.bin').read_bytes(),b'original')

    def test_checksum_failure_prevents_payload_write(self):
        self.bundle(bad_hash=True)
        with self.assertRaises(AssertionError):restore(self.archive,self.dest,'SHA256.json')
        self.assertFalse((self.dest/'data/value.bin').exists())

    def test_path_escape_is_rejected(self):
        self.bundle('../escaped.bin')
        with self.assertRaises(AssertionError):restore(self.archive,self.dest,'SHA256.json')
        self.assertFalse((self.root/'escaped.bin').exists())

    def test_existing_different_file_is_preserved(self):
        self.bundle(); p=self.dest/'data/value.bin'; p.parent.mkdir(); p.write_bytes(b'user edit')
        with self.assertRaises(RuntimeError):restore(self.archive,self.dest,'SHA256.json')
        self.assertEqual(p.read_bytes(),b'user edit')

    def test_explicit_readme_overlay(self):
        self.bundle('data/README.md'); p=self.dest/'data/README.md'; p.parent.mkdir(); p.write_bytes(b'old')
        restore(self.archive,self.dest,'SHA256.json',overlay_readme=True)
        self.assertEqual(p.read_bytes(),b'original')


if __name__=='__main__':unittest.main(verbosity=2)
