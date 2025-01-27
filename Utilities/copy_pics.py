import shutil
import os
import argparse
from datetime import date

""""
Copies every k'th image from the hard drive to the pc using CLI
"""

parser = argparse.ArgumentParser()
parser.add_argument('-k', type=int, default=10, help='step size for copying images.')
parser.add_argument('-date', type=str, default=date.today().strftime('%d_%m_%Y'), help='Date for dirnames.')
parser.add_argument('-calib', action='store_true', help='Is Calibration flag (for dir names)')
parser.add_argument('-lim', type=int, default=1e8, help='max samples to take')
args = parser.parse_args()
k, date, is_calibration, lim = args.k, args.date, args.calib, args.lim

print(
    "RUNNING WITH THE FOLLOWING ARGUMENTS:\n"
    f"--> k: {k} (choosing every {k}{'st' if k == 1 else 'nd' if k == 2 else 'rd' if k == 3 else 'th'} image)\n"
    f"--> date: {date}\n"
    f"--> is_calibration: {is_calibration} \n"
    f"--> lim: {lim}"
)

parent_dirname = 'calibrations' if is_calibration else 'experiments'
src_cam2 = f"E:\\Hadar\\{parent_dirname}\\{date}\\cam2"
src_cam3 = f"E:\\Hadar\\{parent_dirname}\\{date}\\cam3"
dest_cam2 = f"E:\\Hadar\\{parent_dirname}\\{date}\\cam2cpy"
dest_cam3 = f"E:\\Hadar\\{parent_dirname}\\{date}\\cam3cpy"

shutil.rmtree(dest_cam2, ignore_errors=True)
shutil.rmtree(dest_cam3, ignore_errors=True)

os.makedirs(dest_cam2)
os.makedirs(dest_cam3)

counter = 0
print(f"Copy from: [{src_cam2}] to: [{dest_cam2}]")
cam2_files = os.listdir(src_cam2)
for i, file_name in enumerate(cam2_files, start=1):
    if i % k == 0:
        src_file = os.path.join(src_cam2, file_name)
        dst_file = os.path.join(dest_cam2, file_name)
        shutil.copy(src_file, dst_file)
        print(f"CAM 2: [{counter}/{min((len(cam2_files) // k) - 1, lim // k - 1)}] Done...", end='\r')
        counter += 1
        if counter >= lim // k - 1: break

counter = 0
print(f"Copy from: [{src_cam3}] to: [{dest_cam3}]")
cam3_files = os.listdir(src_cam3)
for i, file_name in enumerate(cam3_files, start=1):
    if i % k == 0:
        src_file = os.path.join(src_cam3, file_name)
        dst_file = os.path.join(dest_cam3, file_name)
        shutil.copy(src_file, dst_file)
        print(f"CAM 3: [{counter}/{min((len(cam2_files) // k) - 1, lim // k - 1)}] Done...", end='\r')
        counter += 1
        if counter >= lim // k - 1: break

print('\nAll Done!')
