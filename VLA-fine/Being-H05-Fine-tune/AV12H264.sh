# !/bin/bash 

for ds in /home/hyl/Being-H/origin_data_v21/koch_follower/*/; do
  for cam in observation.images.above observation.images.wrist; do
    video_dir="${ds}videos/chunk-000/${cam}"
    [ -d "$video_dir" ] && for f in "$video_dir"/*.mp4; do
      ffmpeg -y -i "$f" -c:v libx264 -preset fast -crf 23 "${f}.h264.mp4" 2>/dev/null && mv "${f}.h264.mp4" "$f"
    done
  done
  echo "Done: $(basename $ds)"
done