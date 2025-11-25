import os
import subprocess
import numpy as np
from PIL import Image

import newton.viewer


def bin_to_mp4(
    input_bin,
    output_mp4="output.mp4",
    temp_dir="frames",
    img_format="png",
    fps=60,
    width=1920,
    height=1080,
):
    # ------------------------------------------------------------
    # Load recording
    # ------------------------------------------------------------
    vf = newton.viewer.ViewerFile(input_bin)
    vf.show_particles = True
    # load recording into the viewer/recorder and populate internal state
    vf.load_recording(input_bin)
    nframes = vf.get_frame_count()
    print(f"Loaded recording: {input_bin}")
    print(f"Frames: {nframes}")

    # ------------------------------------------------------------
    # Prepare temporary folder for images
    # ------------------------------------------------------------
    os.makedirs(temp_dir, exist_ok=True)

    # ------------------------------------------------------------
    # Create a visible viewer (OpenGL) but off-screen
    # ------------------------------------------------------------
    viewer = newton.viewer.ViewerGL(width=width, height=height, headless=False)

    # Attempt to reconstruct a Model from the recording and set it on the viewer
    try:
        model = newton.Model()
        vf._recorder.playback_model(model)
        viewer.set_model(model)
    except Exception:
        # best-effort: if playback_model fails leave the viewer without a model
        viewer.set_model(None)

    print("Rendering frames...")

    for i in range(nframes):
        # playback recorded arrays into a fresh State object
        state = newton.State()
        vf._recorder.playback(state, i)

        # approximate time for rendering (recordings don't currently store time)
        frame_time = float(i) / float(fps)

        viewer.begin_frame(frame_time)
        viewer.log_state(state)
        viewer.end_frame()

        # Read back the rendered color buffer (H x W x 4 uint8)
        # ViewerGL doesn't expose a direct get_color_buffer() API; use pyglet's buffer manager.
        try:
            import pyglet

            buf = pyglet.image.get_buffer_manager().get_color_buffer()
            img_data = buf.get_image_data()
            raw = img_data.get_data("RGBA", img_data.width * 4)
            arr = np.frombuffer(raw, dtype=np.uint8)
            # pyglet image data is bottom-to-top; reshape and flip vertically
            color = arr.reshape((img_data.height, img_data.width, 4))[::-1]
        except Exception:
            # Fallback to a blank image if readback fails
            fb_w, fb_h = viewer.renderer.window.get_framebuffer_size()
            color = np.zeros((fb_h, fb_w, 4), dtype=np.uint8)

        # Save as image
        img = Image.fromarray(color)
        img.save(f"{temp_dir}/frame_{i:06d}.{img_format}")
        print('saved to '+f"{temp_dir}/frame_{i:06d}.{img_format}")

        if i % 50 == 0:
            print(f"Rendered {i}/{nframes} frames")

    print("All frames rendered.")

    # ------------------------------------------------------------
    # Encode the video using ffmpeg
    # ------------------------------------------------------------
    print("Encoding mp4 with ffmpeg...")

    ffmpeg_cmd = [
        "ffmpeg",
        "-y",
        "-framerate", str(fps),
        "-i", f"{temp_dir}/frame_%06d.{img_format}",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        output_mp4,
    ]

    subprocess.run(ffmpeg_cmd, check=True)

    print(f"Video saved to: {output_mp4}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python bin_to_mp4.py recording.bin [output.mp4]")
        exit(1)

    input_bin = sys.argv[1]
    output_mp4 = sys.argv[2] if len(sys.argv) >= 3 else "output.mp4"

    bin_to_mp4(input_bin, output_mp4=output_mp4)


# TOOD: Should fix recorder code to properly save particles in .bin files