
# Converts a rgba color into a single integer
def rgb_to_hex(r, g, b, a = 255):
    """Converts a 8-bit RGBA color to a single integer matching (0xRRGGBBAA).

    Args:
        r (int): Red color channel
        g (int): Green color channel
        b (int): Blue color channel
        a (int, optional): Alpha color channel. Defaults to 255.

    Returns:
        int: An integer describing the given color as 32-bit integer (0xRRGGBBAA).
    """
    col_bytes = bytearray([r, g, b, a])
    col_hex = int.from_bytes(col_bytes, byteorder='big')
    return col_hex