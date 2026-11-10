"""Explicit test-table lossless storage; decoding returns unchanged uint8 bytes.

This codec has no scientific acceptance, dataset selection or training authority.
"""
import hashlib
import io
import re
import zipfile
import zlib

VERSION = 'test_table_uint8_npz_deflate6_v1'
LEVEL = 6
MAX_RAW_BYTES = 48 * 1024 * 1024
POLICY = {'version': VERSION, 'partition': 'test', 'encoder': 'numpy_npy_zip_deflate',
          'compression_level': LEVEL, 'dtype': 'uint8', 'array_order': 'C',
          'array_name': 'payload.npy', 'lossless': True,
          'scope': 'new selected test table arrays only; original native PNG evidence unchanged'}


def encode(data):
    import numpy as np
    if not isinstance(data, bytes) or not 0 < len(data) <= MAX_RAW_BYTES:
        raise ValueError('Compressed table payload requires bounded nonempty original bytes')
    raw = io.BytesIO()
    np.save(raw, np.frombuffer(data, dtype=np.uint8), allow_pickle=False)
    out = io.BytesIO()
    info = zipfile.ZipInfo('payload.npy', date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o600 << 16
    with zipfile.ZipFile(out, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=LEVEL) as archive:
        archive.writestr(info, raw.getvalue(), compress_type=zipfile.ZIP_DEFLATED, compresslevel=LEVEL)
    body = out.getvalue()
    fields = {'encoding': dict(POLICY), 'encoder_versions': {'numpy': np.__version__, 'zlib': zlib.ZLIB_VERSION},
              'raw_sha256': hashlib.sha256(data).hexdigest(), 'raw_bytes': len(data),
              'sha256': hashlib.sha256(body).hexdigest(), 'bytes': len(body)}
    if decode(body, fields) != data:
        raise ValueError('Immediate compression roundtrip changed original bytes')
    return body, fields


def decode(body, reference):
    import numpy as np
    if (not isinstance(body, bytes) or reference.get('encoding') != POLICY
            or type(reference.get('bytes')) is not int or len(body) != reference['bytes']
            or hashlib.sha256(body).hexdigest() != reference.get('sha256')
            or type(reference.get('raw_bytes')) is not int
            or not 0 < reference['raw_bytes'] <= MAX_RAW_BYTES
            or not isinstance(reference.get('encoder_versions'), dict)
            or set(reference['encoder_versions']) != {'numpy', 'zlib'}
            or any(not isinstance(v, str) or not v for v in reference['encoder_versions'].values())):
        raise ValueError('Unknown codec or compressed bytes/length/SHA mismatch')
    with zipfile.ZipFile(io.BytesIO(body), 'r') as archive:
        entries = archive.infolist()
        if (len(entries) != 1 or entries[0].filename != 'payload.npy'
                or entries[0].compress_type != zipfile.ZIP_DEFLATED or entries[0].flag_bits & 1
                or not reference['raw_bytes'] < entries[0].file_size <= reference['raw_bytes'] + 1024):
            raise ValueError('Unexpected NPZ contents, encryption, or unbounded raw array')
        encoded_array = archive.read(entries[0])  # ZIP CRC and declared size checked by zipfile.
    stream = io.BytesIO(encoded_array)
    version = np.lib.format.read_magic(stream)
    if version != (1, 0):
        raise ValueError('Only the encoder\'s bounded NPY1 uint8 array is permitted')
    shape, fortran, dtype = np.lib.format.read_array_header_1_0(stream, max_header_size=1024)
    if shape != (reference['raw_bytes'],) or fortran or dtype != np.dtype('uint8') or dtype.hasobject:
        raise ValueError('Compressed payload is not its exact one-dimensional non-pickle uint8 bytes')
    data = stream.read(reference['raw_bytes'] + 1)
    if len(data) != reference['raw_bytes'] or hashlib.sha256(data).hexdigest() != reference.get('raw_sha256'):
        raise ValueError('Decoded original bytes/length/SHA mismatch')
    return data


def require_test_table(scene, source):
    match = re.fullmatch(r'FloorPlan(\d+)', str(scene))
    number = int(match.group(1)) if match else -1
    if (not any(base + 21 <= number <= base + 30 for base in (0, 200, 300, 400))
            or not isinstance(source, dict) or source.get('g01_partition') != 'test'):
        raise ValueError('Lossless compressed storage is explicitly restricted to new test tables')
