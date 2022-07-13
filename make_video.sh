# See https://trac.ffmpeg.org/wiki/Create%20a%20mosaic%20out%20of%20several%20input%20videos%20using%20xstack

out_fn="${1}"
fps=${FPS:=25}
height=${HEIGHT:=1080}
plot_mode="${PLOTMODE:=rho}"

if [ ${plot_mode} = "rho" ]; then
  prefix1="rho_fit_nostars"
  prefix2="rho_diff_nostars"
elif [ ${plot_mode} = "stars" ]; then
  prefix1="ln_rho_fit_stars"
  prefix2="ln_rho_diff_stars"
elif [ ${plot_mode} = "fourier" ]; then
  prefix1="fourier"
  prefix2="power"
else
  echo "Unrecognized PLOTMODE: ${PLOTMODE}"
  return 1
fi

echo ${plot_mode}

ffmpeg \
  -y \
  -r ${fps} \
  -pattern_type glob \
  -i ''"${prefix1}"'_step?????.png' \
  -r ${fps} \
  -pattern_type glob \
  -i ''"${prefix2}"'_step?????.png' \
  -filter_complex " \
    [0:v] setpts=PTS-STARTPTS, scale=-1:${height} [a0]; \
    [1:v] setpts=PTS-STARTPTS, scale=-1:${height} [a1]; \
    [a0][a1]hstack=inputs=2[out] \
    " \
  -map "[out]" \
  -c:v libx264 \
  -pix_fmt yuv420p \
  -movflags +faststart \
  -r ${fps} \
  "${out_fn}"
