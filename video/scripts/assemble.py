"""Assemble the silent walkthrough: normalize scenes, composite the mobile
scene onto a 1920x1080 canvas, chain 0.5s crossfades, crf 18 +faststart."""
import json, subprocess, sys

SCE = "/home/riju279/Documents/code/AIRA/histogate/video/scenes"
OUT = "/home/riju279/Documents/code/AIRA/histogate/video/out"
XFADE = 0.5

# order, max duration (None = full)
PLAN = [("s1", None), ("s2", None), ("s3", None), ("s4", 21.0), ("s5", None),
        ("s6", None), ("s7", None), ("s8", None), ("s9", None), ("s10", None)]

def dur(p):
    r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                        "-of", "csv=p=0", p], capture_output=True, text=True)
    return float(r.stdout.strip())

durs = {}
for name, cap in PLAN:
    d = dur(f"{SCE}/{name}.mp4")
    durs[name] = min(d, cap) if cap else d
    print(name, round(durs[name], 3))

parts, inputs = [], []
for i, (name, _) in enumerate(PLAN):
    inputs += ["-i", f"{SCE}/{name}.mp4"]
    trim = f"trim=duration={durs[name]:.3f},setpts=PTS-STARTPTS," if durs[name] < dur(f"{SCE}/{name}.mp4") - 0.01 else ""
    if name == "s7":  # 390x844 mobile -> centered card on dark canvas
        f = (f"[{i}:v]{trim}fps=30,scale=445:962:flags=lanczos,"
             f"pad=449:966:2:2:color=#26282e,"
             f"pad=1920:1080:(ow-iw)/2:(oh-ih)/2:color=#08090a,"
             f"format=yuv420p,setsar=1[a{i}]")
    else:
        f = (f"[{i}:v]{trim}fps=30,format=yuv420p,setsar=1[a{i}]")
    parts.append(f)

prev = "a0"; cum = durs[PLAN[0][0]]
for i in range(1, len(PLAN)):
    off = cum - XFADE
    out = f"x{i}"
    parts.append(f"[{prev}][a{i}]xfade=transition=fade:duration={XFADE}:offset={off:.3f}[{out}]")
    prev = out
    cum = cum + durs[PLAN[i][0]] - XFADE

fc = ";".join(parts)
cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", *inputs,
       "-filter_complex", fc, "-map", f"[{prev}]",
       "-c:v", "libx264", "-preset", "medium", "-crf", "18",
       "-pix_fmt", "yuv420p", "-movflags", "+faststart",
       f"{OUT}/histogate_walkthrough_silent.mp4"]
print("total duration:", round(cum, 2), "s")
subprocess.run(cmd, check=True)
print("WROTE", f"{OUT}/histogate_walkthrough_silent.mp4")
