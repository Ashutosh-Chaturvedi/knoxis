#!/usr/bin/env bash
# One-time setup on a fresh Ubuntu 22.04 / 24.04 VM (Azure or any other).
#   sudo bash setup_vm.sh https://github.com/<you>/knoxis.git
# Safe to re-run. It does not touch other nginx sites and does not issue the certificate (see DEPLOY.md).
set -euo pipefail

REPO_URL="${1:?usage: sudo bash setup_vm.sh <git repo url>}"
BASE=/srv/knoxis
[ "$(id -u)" -eq 0 ] || { echo "run with sudo"; exit 1; }

export DEBIAN_FRONTEND=noninteractive
apt-get update -y
# libgomp1: OpenMP runtime that LightGBM needs
apt-get install -y nginx python3-venv python3-pip git ufw fail2ban unattended-upgrades \
                   certbot python3-certbot-nginx libgomp1 rsync curl

id knoxis >/dev/null 2>&1 || useradd --system --home-dir "$BASE" --shell /usr/sbin/nologin knoxis
mkdir -p "$BASE"/data/days "$BASE"/data/live "$BASE"/data/model
chown -R knoxis:knoxis "$BASE"

if [ ! -d "$BASE/app/.git" ]; then
  sudo -u knoxis git clone "$REPO_URL" "$BASE/app"
fi
[ -f "$BASE/app/api/main.py" ] || { echo "api/main.py not found in the cloned repo: push the api/ folder first"; exit 1; }

sudo -u knoxis python3 -m venv "$BASE/venv"
sudo -u knoxis "$BASE/venv/bin/pip" install --upgrade pip
sudo -u knoxis "$BASE/venv/bin/pip" install -r "$BASE/app/requirements.txt"

if [ ! -f /etc/knoxis.env ]; then
  install -m 640 -o root -g knoxis "$BASE/app/deploy/knoxis.env.example" /etc/knoxis.env
  sed -i "s/^KNOXIS_ADMIN_TOKEN=.*/KNOXIS_ADMIN_TOKEN=$(openssl rand -hex 24)/" /etc/knoxis.env
  echo "created /etc/knoxis.env with a random admin token"
fi

install -m 644 "$BASE/app/deploy/knoxis.service" /etc/systemd/system/knoxis.service
systemctl daemon-reload
systemctl enable knoxis

install -m 644 "$BASE/app/deploy/nginx-knoxis.conf" /etc/nginx/sites-available/knoxis
ln -sf /etc/nginx/sites-available/knoxis /etc/nginx/sites-enabled/knoxis
nginx -t
systemctl reload nginx

ufw allow OpenSSH
ufw allow 'Nginx Full'
ufw --force enable

systemctl restart knoxis
sleep 4
curl -fsS http://127.0.0.1:8100/health && echo
echo
echo "Setup done. Next: copy the data to $BASE/data (see DEPLOY.md), point DNS at this VM, then run:"
echo "  sudo certbot --nginx -d knoxis.in -d www.knoxis.in"
