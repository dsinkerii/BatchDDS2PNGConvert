import os
import sys
from wand import image
import argparse
from PIL import Image
import struct
import numpy as np

parser = argparse.ArgumentParser("main.py")
parser.add_argument("folder_in", help="Folder that contains DDS images.", type=str)
parser.add_argument("folder_out", help="Folder that will output PNG images to.", type=str)

import numpy as np
import struct

def decompress_bc4(data, width, height):
    blocks_x = (width + 3) // 4
    blocks_y = (height + 3) // 4
    decompressed = np.zeros((height, width), dtype=np.uint8)

    for by in range(blocks_y):
        for bx in range(blocks_x):
            block_start = (by * blocks_x + bx) * 8
            block_end = block_start + 8
            if block_end > len(data):
                break
            block_data = data[block_start:block_end]
            min_value, max_value = struct.unpack('<2B', block_data[:2])
            indices = np.unpackbits(np.frombuffer(block_data[2:], dtype=np.uint8))

            if min_value > max_value:
                values = [min_value, max_value, 
                          (6 * min_value + 1 * max_value) // 7,
                          (5 * min_value + 2 * max_value) // 7,
                          (4 * min_value + 3 * max_value) // 7,
                          (3 * min_value + 4 * max_value) // 7,
                          (2 * min_value + 5 * max_value) // 7,
                          (1 * min_value + 6 * max_value) // 7]
            else:
                values = [min_value, max_value, 
                          (4 * min_value + 1 * max_value) // 5,
                          (3 * min_value + 2 * max_value) // 5,
                          (2 * min_value + 3 * max_value) // 5,
                          (1 * min_value + 4 * max_value) // 5,
                          0, 255]

            for y in range(4):
                for x in range(4):
                    if by * 4 + y < height and bx * 4 + x < width:
                        decompressed[by * 4 + y, bx * 4 + x] = values[indices[y * 4 + x]]

    return decompressed

def decompress_dx10(data, width, height):
    dxgi_format, dimension, misc_flag, array_size, misc_flags2 = struct.unpack('<5I', data[:20])
    
    data = data[20:]
    
    format_map = {
        28: np.uint8,  # DXGI_FORMAT_R8G8B8A8_UNORM
        87: np.uint8,  # DXGI_FORMAT_B8G8R8A8_UNORM
        41: np.uint16, # DXGI_FORMAT_R16G16B16A16_FLOAT
        10: np.float32, # DXGI_FORMAT_R32G32B32A32_FLOAT
        80: np.uint8,  # DXGI_FORMAT_BC4_UNORM
    }
    
    if dxgi_format not in format_map:
        print(f"Unsupported DXGI format: {dxgi_format}")
        return None
    
    dtype = format_map[dxgi_format]
    channels = 4
    
    compressed_formats = [80]

    if dxgi_format in compressed_formats:
        block_size = 8
        blocks_x = (width + 3) // 4
        blocks_y = (height + 3) // 4
        expected_size = blocks_x * blocks_y * block_size
    else:
        bytes_per_pixel = np.dtype(dtype).itemsize * channels
        expected_size = width * height * bytes_per_pixel

    if len(data) < expected_size:
        print(f"Warning: Not enough data. Expected {expected_size} bytes, got {len(data)} bytes.")
        expected_size = len(data)

    if dxgi_format == 80:
        image = decompress_bc4(data[:expected_size], width, height)
        image = np.stack([image, image, image, np.full_like(image, 255)], axis=-1)
    else:
        pixels = np.frombuffer(data[:expected_size], dtype=dtype)
        image = pixels.reshape((height, width, channels))
    
    if dtype != np.uint8:
        if dtype == np.uint16:
            image = (image / 65535.0 * 255.0).astype(np.uint8)
        elif dtype == np.float32:
            image = (np.clip(image, 0, 1) * 255.0).astype(np.uint8)
    
    if dxgi_format == 87:
        image = image[:, :, [2, 1, 0, 3]]
    
    return image


def get_dds_format(file_path):
    with open(file_path, 'rb') as f:
        magic = f.read(4)
        if magic != b'DDS ':
            return 'Not a DDS file', 0, 0
        
        f.seek(12)
        height = struct.unpack('<I', f.read(4))[0]
        width = struct.unpack('<I', f.read(4))[0]
        
        f.seek(76)
        pixel_format_data = f.read(32)
        
        try:
            flags, fourcc = struct.unpack('<2I', pixel_format_data[:8])
            
            if flags & 0x4:  # DDPF_FOURCC
                return fourcc.decode('ascii', errors='ignore'), width, height
            elif flags & 0x40:  # DDPF_RGB
                rgb_bit_count, = struct.unpack('<I', pixel_format_data[8:12])
                return f'RGB{rgb_bit_count}', width, height
            else:
                return 'Unknown format', width, height
        except struct.error:
            return 'Unable to parse format', width, height

def convert_dds_to_png(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    compression_types = [
        'dxt1', 'dxt2', 'dxt3', 'dxt4', 'dxt5',
        'dx10', 'cao',
        'bc1', 'bc2', 'bc3', 'bc4', 'bc5', 'bc6h', 'bc6s', 'bc7',
        'atc', 'atc_rgb', 'atc_rgba_explicit', 'atc_rgba_interpolated',
        'etc1', 'etc2', 'etc2a',
        'pvrtc', 'pvrtc2',
        'rgtc1', 'rgtc2',
        'bptc',
        's3tc',
        'astc_4x4', 'astc_5x4', 'astc_5x5', 'astc_6x5', 'astc_6x6', 'astc_8x5', 'astc_8x6', 'astc_8x8', 'astc_10x5', 'astc_10x6', 'astc_10x8', 'astc_10x10', 'astc_12x10', 'astc_12x12'
    ]

    global success_conv
    global success_using_custom_DX10_conv
    global failed_conv
    success_conv = 0
    success_using_custom_DX10_conv = 0
    failed_conv = 0

    for filename in os.listdir(input_folder):
        if filename.lower().endswith('.dds'):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename[:-4] + '.png')

            attempt = 1
            success = False
            for compression in compression_types:
                try:
                    with image.Image(filename=input_path) as img:
                        img.compression = compression
                        img.save(filename=output_path)
                    success = True
                    print(f"Converted {filename} using {compression} compression.")
                    break
                except Exception as e:
                    attempt += 1
                    continue
            if(success):
                success_conv += 1
                print(f"Successfully converted {filename} after {attempt} attempt(s).")
            else:
                print(f"\n\n!!!\tFailed to convert {filename} using any compression type.\t!!!")
                print("!!!\tTrying to use custom decompression with DX10....")
                try:
                    with open(input_path, 'rb') as f:
                        header = f.read(128)
                        data = f.read()
                    
                    dds_format, width, height = get_dds_format(input_path)
                    
                    print("!!!\tDetected file width:", width, "height:", height)
                    
                    decompressed = decompress_dx10(data, width, height)
                    if decompressed is None:
                        print(f"Custom DX10 decompression failed for {filename}.")
                        failed_conv += 1
                        broken_image = np.zeros((height, width, 4), dtype=np.uint8)
                        img = Image.fromarray(broken_image, 'RGBA')
                        img.save(output_path)
                        print(f"Output a broken texture for {filename}.")
                        continue
                    
                    img = Image.fromarray(decompressed, 'RGBA')
                    img.save(output_path)
                    success_using_custom_DX10_conv += 1
                    print(f"!!!\tConverted {filename} using custom DX10 decompression. Please note that this decompression may not be accurate due to the fact your system couldn't convert it.\n\n")
                except Exception as e:
                    failed_conv += 1
                    broken_image = np.zeros((height, width, 4), dtype=np.uint8)
                    img = Image.fromarray(broken_image, 'RGBA')
                    img.save(output_path)
                    print(f"!!!\tError during custom DX10 decompression: {str(e)}")
                    print(f"!!!\tOutput a broken texture for {filename}.")
                continue

if __name__ == "__main__":
    if(len(sys.argv) < 3):
        print("Usage: main.py <input_folder> <output_folder>\narguments:\n\t<input_folder> - Folder that contains DDS images.\n\t<output_folder> - Folder that will output PNG images to.")
        exit()
    input_folder = sys.argv[1]
    output_folder = sys.argv[2]

    convert_dds_to_png(input_folder, output_folder)
    os.system('cls')
    os.system('clear')
    global success_conv
    global success_using_custom_DX10_conv
    global failed_conv
    print(f"\n\nSuccessfully converted: {success_conv}")
    print(f"\n\nSuccessfully converted using custom DX10 decompression: {success_using_custom_DX10_conv}")
    print(f"\n\nFailed: {failed_conv}")