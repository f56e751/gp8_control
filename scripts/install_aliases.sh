#!/bin/bash
# Add short commands for THIS workspace to ~/.bashrc (idempotent):
#   gp8_env  gp8_build  gp8_robot  gp8_app  gp8_camera  gp8_s2r
D="$(cd "$(dirname "$0")" && pwd)"
MARK="# >>> gp8_control aliases >>>"
if grep -qF "$MARK" ~/.bashrc; then
  sed -i "/$MARK/,/# <<< gp8_control aliases <<</d" ~/.bashrc
fi
cat >> ~/.bashrc <<EOT
$MARK
alias gp8_env='source $D/env.sh'
alias gp8_build='$D/build.sh'
alias gp8_robot='$D/robot.sh'
alias gp8_app='$D/app.sh'
alias gp8_camera='$D/camera.sh'
alias gp8_s2r='$D/s2r_run.sh'
# <<< gp8_control aliases <<<
EOT
echo "added gp8_* aliases for $D — open a new terminal or: source ~/.bashrc"
