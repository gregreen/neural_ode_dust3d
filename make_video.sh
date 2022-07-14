# See https://trac.ffmpeg.org/wiki/Create%20a%20mosaic%20out%20of%20several%20input%20videos%20using%20xstack

out_fn="${1}"
fps=${FPS:=25}
height=${HEIGHT:=1080}
plot_mode="${PLOTMODE:=rho}"

if [ ${plot_mode} = "rho" ]; then
  prefix1="rho_fit_nostars"
  prefix2="rho_diff_nostars"
  stackmode="h"
elif [ ${plot_mode} = "stars" ]; then
  prefix1="ln_rho_fit_stars"
  prefix2="ln_rho_diff_stars"
  stackmode="h"
elif [ ${plot_mode} = "fourier" ]; then
  prefix1="fourier"
  prefix2="power"
  stackmode="h"
elif [ ${plot_mode} = "sky_close" ]; then
  prefix1="A_sky_close_fit"
  prefix2="A_sky_close_diff"
  stackmode="v"
elif [ ${plot_mode} = "sky_far" ]; then
  prefix1="A_sky_far_fit"
  prefix2="A_sky_far_diff"
  stackmode="v"
else
  echo "Unrecognized PLOTMODE: ${PLOTMODE}"
  return 1
fi

if [ ${stackmode} = "v" ]; then
  height=$(( ${height} / 2 ));
fi

echo "plot mode: ${plot_mode}"

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
    [a0][a1]${stackmode}stack=inputs=2[out] \
    " \
  -map "[out]" \
  -c:v libx264 \
  -pix_fmt yuv420p \
  -movflags +faststart \
  -r ${fps} \
  "${out_fn}"
