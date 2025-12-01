ffmpeg -i dataset/EgoWalk_samples/2025_01_22__15_52_08_custom_segment_061_130/2025_01_22__15_52_08_custom_segment_061_130_video.mp4 -c:v libx264 -c:a aac dataset/EgoWalk_samples/2025_01_22__15_52_08_custom_segment_061_130/gt_fixed.mp4
ffmpeg -i dataset/EgoWalk_samples/2025_01_22__15_52_08_custom_segment_061_130/gt_fixed.mp4 -vf "select='between(n\,0\,8)'" -vsync vfr dataset/EgoWalk_samples/2025_01_22__15_52_08_custom_segment_061_130/first_current.mp4
ffmpeg -i dataset/EgoWalk_samples/2025_01_22__15_52_08_custom_segment_061_130/first_current.mp4 -vf "scale=1280:720,fps=30" -c:a copy dataset/EgoWalk_samples/2025_01_22__15_52_08_custom_segment_061_130/first_current_fixed.mp4
cd dataset/EgoWalk_samples/2025_01_22__15_52_08_custom_segment_061_130
echo "file 'first_current_fixed.mp4'" > list.txt
echo "file 'Move_closer_to_202509041321_fmyhz.mp4'" >> list.txt
ffmpeg -f concat -safe 0 -i list.txt -c copy combined_2.mp4
ffprobe -v error -count_frames -select_streams v:0 -show_entries stream=nb_read_frames -of default=nokey=1:noprint_wrappers=1 first_current_fixed.mp4
ffprobe -v error -count_frames -select_streams v:0 -show_entries stream=nb_read_frames -of default=nokey=1:noprint_wrappers=1 Move_closer_to_202509041321_fmyhz.mp4
ffprobe -v error -count_frames -select_streams v:0 -show_entries stream=nb_read_frames -of default=nokey=1:noprint_wrappers=1 combined_2.mp4