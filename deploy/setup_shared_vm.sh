#!/usr/bin/env bash
# Add Knoxis to a VM that already serves another site through nginx (for example chat-app.live).
#   sudo bash setup_shared_vm.sh https://github.com/<you>/knoxis.git            # no swap changes
#   sudo bash setup_shared_vm.sh https://github.com/<you>/knoxis.git --swap 2G  # also create a 2 GB swap file
#
# Differences from setup_vm.sh (which is for a fresh VM):
#   - does not install, restart or reconfigure nginx; it only adds one site file and reloads nginx,
#     and only after `nginx -t` passes with the new site in place (otherwise it removes the site again)
#   - does not touch the firewall, fail2ban or unattended-upgrades
#   - caps Knoxis memory (systemd MemoryHigh/MemoryMax) so a big day build cannot starve the other site
#   - refuses to continue if port 8100 is already in use
set -euo pipefail

REPO_URL="${1:?usage: sudo bash setup_shared_vm.sh <git repo url> [--swap SIZE]}"
SWAP=""
if [ "${2:-}" = "--swap" ]; then SWAP="${3:?give a size, for example 2G}"; fi
BASE=/srv/knoxis

[ "$(id -u)" -eq 0 ] || { echo "run with sudo"; exit 1; }
command -v nginx >/dev/null || { echo "nginx is not installed here; use setup_vm.sh on a fresh VM instead"; exit 1; }

if ss -ltnH | awk '{print $4}' | grep -qE '[:.]8100$'; then
  echo "port 8100 is already in use. Change it in deploy/knoxis.service and deploy/nginx-knoxis.conf first."
  exit 1
fi

export DEBIAN_FRONTEND=noninteractive
apt-get update -y
# libgomp1: OpenMP runtime that LightGBM needs
apt-get install -y python3-venv python3-pip git libgomp1 rsync curl

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

# systemd service plus a memory cap, so Knoxis is restarted before the other site is starved
install -m 644 "$BASE/app/deploy/knoxis.service" /etc/systemd/system/knoxis.service
mkdir -p /etc/systemd/system/knoxis.service.d
cat > /etc/systemd/system/knoxis.service.d/memory.conf <<'EOF'
[Service]
MemoryHigh=1500M
MemoryMax=2000M
EOF
systemctl daemon-reload
systemctl enable knoxis

# optional swap file as a safety net (a VM with no swap kills processes when memory runs out)
if [ -n "$SWAP" ] && [ -z "$(swapon --show --noheadings)" ]; then
  fallocate -l "$SWAP" /swapfile
  chmod 600 /swapfile
  mkswap /swapfile
  swapon /swapfile
  grep -q '^/swapfile' /etc/fstab || echo '/swapfile none swap sw 0 0' >> /etc/fstab
  echo "swap file of $SWAP enabled"
fi

# nginx: add only the Knoxis site, in whichever style this nginx loads sites
if grep -qs 'sites-enabled' /etc/nginx/nginx.conf; then
  SITE=/etc/nginx/sites-available/knoxis
  ENABLED=/etc/nginx/sites-enabled/knoxis
  install -m 644 "$BASE/app/deploy/nginx-knoxis.conf" "$SITE"
  ln -sf "$SITE" "$ENABLED"
else
  ENABLED=/etc/nginx/conf.d/knoxis.conf
  install -m 644 "$BASE/app/deploy/nginx-knoxis.conf" "$ENABLED"
fi
if ! nginx -t; then
  rm -f "$ENABLED"
  echo "nginx rejected the config with Knoxis added. The Knoxis site was removed again and nginx was NOT reloaded."
  exit 1
fi
systemctl reload nginx

systemctl restart knoxis
sleep 4
curl -fsS http://127.0.0.1:8100/health && echo
echo
echo "Knoxis is installed next to the existing site. Next:"
echo "  1. copy the data and model into $BASE/data (see DEPLOY.md step 4)"
echo "  2. point DNS for knoxis.in and www at this VM's IP"
echo "  3. sudo certbot --nginx -d knoxis.in -d www.knoxis.in      (needs certbot; the existing site's certificate is untouched)"
echo "  4. watch memory: free -h ; systemctl status knoxis"
