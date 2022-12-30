from io import BytesIO
import PIL.Image
import skvideo.io
import subprocess
import numpy as np
import tempfile
import importlib
import time

ffmpeg_spec = importlib.util.find_spec("ffmpeg")

if ffmpeg_spec is not None:
    import ffmpeg


def jpeg_compress(image, jpeg_quality):
    buffer = BytesIO()  # TODO: check if subsampling needs to be changed for JPEG....
    image.save(buffer, "JPEG", subsampling=0, quality=jpeg_quality)  # main compression step
    buffer.seek(0)
    compressed_image = PIL.Image.open(buffer)
    return compressed_image


def jm_compress(image, qpi, jm_command, yuv_loc, comp_loc, verbose=False, max_tries=3, retry_delay=0.001):
    init_params = 'QPISlice=%d -p SourceHeight=%d -p SourceWidth=%d' % (0, 0, 0)

    # Adjusting JM command for image dimensions and selected QPI
    l_w, l_h = image.size
    new_params = 'QPISlice=%d -p SourceHeight=%d -p SourceWidth=%d' % (qpi, l_h, l_w)
    command_full = jm_command.replace(init_params, new_params)

    for try_counter in range(max_tries):
        try:
            # convert im to YUV file
            vid = skvideo.utils.vshape(np.array(image))
            skvideo.io.vwrite(yuv_loc, vid, verbosity=1 if verbose else 0, outputdict={"-pix_fmt": "yuv420p"})

            # Perform JM compression
            process = subprocess.Popen(command_full,
                                       shell=True,
                                       stdout=subprocess.PIPE,
                                       stderr=subprocess.PIPE)

            stdout, stderr = process.communicate()

            if verbose:
                print('%%%%%%%%%%%%%%%%%')
                print('JM Error Output:')
                print(stderr)
                print('%%%%%%%%%%%%%%%%%')

            # Restore image from video
            vid = skvideo.io.vread(comp_loc, height=l_h, width=l_w,
                                   inputdict={"-pix_fmt": "yuv420p"}, verbosity=1 if verbose else 0)
            # recover resulting image to PIL array
            output = PIL.Image.fromarray(vid[0, :, :, :].astype(np.uint8))
            break       # as soon as it works, break out of the loop
        except (AssertionError, OSError, IndexError) as e:
            if try_counter == max_tries-1:
                raise RuntimeError('Error when running the JM compression system. Original exception: ' + repr(e))
            else:
                time.sleep(retry_delay)
                continue

    return output


def ffmpeg_compress(image, encoder_args, decoder_args, verbose=False):
    # Convert the PIL image into a bytes buffer to pass it to the FFMPEG through a pipe
    if ffmpeg_spec is None:
        raise RuntimeError('FFMPEG not installed - compression cannot run.')

    byte_buffer = BytesIO()
    image.save(byte_buffer, 'PNG')
    bytes_object = byte_buffer.getvalue()

    if verbose:
        log_level_value = 'verbose'
    else:
        log_level_value = 'error'

    # Create a temporary file in memory to store the output of the FFMPEG encoder
    temp_file = tempfile.SpooledTemporaryFile(suffix='.yuv')

    # Convert the PNG images from the bytes buffer to a yuv image compressed using libx264
    out_encoder, _ = (
        ffmpeg
        .input('pipe:')
        .output('pipe:', **encoder_args)
        .global_args('-loglevel', log_level_value)
        .global_args('-y')
        .run(input=bytes_object, capture_stdout=True)
    )

    # Write the output into the temporary file and set to pointer to the start
    temp_file.write(out_encoder)
    temp_file.seek(0)

    # Convert the yuv image back to a PNG and get the output as a buffer
    out_decoder, _ = (
        ffmpeg
        .input('pipe:')
        .output('pipe:', **decoder_args)
        .global_args('-loglevel', log_level_value)
        .global_args('-y')
        .run(input=temp_file.read(), capture_stdout=True)
    )

    temp_file.close()

    # Decode and reshape the bytes buffer
    vid = skvideo.utils.vshape(np.array(image))
    comp_vid = np.frombuffer(out_decoder, np.uint8).reshape(vid.shape)

    # Recover the resulting image to PIL array
    output = PIL.Image.fromarray(comp_vid[0, :, :, :].astype(np.uint8))

    return output
