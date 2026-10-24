#!/usr/bin/env bash
# Deploy the latest code from git:  sudo bash /srv/knoxis/app/deploy/update.sh
set -euo pipefail
cd /srv/knoxis/app
sudo -u knoxis git pull --ff-only
sudo -u knoxis /srv/knoxis/venv/bin/pip install -r requirements.txt
install -m 644 deploy/knoxis.service /etc/systemd/system/knoxis.service
systemctl daemon-reload
systemctl restart knoxis
sleep 4
curl -fsS http://127.0.0.1:8100/health && echo
