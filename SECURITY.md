# Security and privacy

Do not commit API keys, camera serial numbers, private network addresses,
runtime databases, recorded images, or deployment credentials.

The application uploads no image by default. Future online scene understanding
must use an explicit per-request consent path. Local object detection, depth
estimation, and guidance keep the current frame on the device.

Run `python3 tools/check_public_secrets.py` before publishing.

Report security or privacy issues privately to the repository owner.

