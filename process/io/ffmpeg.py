import subprocess

def cut_video(input_path, start_time, duration):
    output_path = "cut_input.mp4"
    cmd = ["ffmpeg", "-y", "-ss", str(start_time), "-i", input_path,
           "-t", str(duration), "-c", "copy", output_path]
    subprocess.run(cmd, check=True)
    return output_path

def convert_video_to_safe_format(input_path):
    safe_path = "safe_input.mp4"
    cmd = ["ffmpeg", "-y", "-i", input_path,
           "-movflags", "faststart", "-pix_fmt", "yuv420p", safe_path]
    subprocess.run(cmd, check=True)
    return safe_path
