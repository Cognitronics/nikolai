#!/usr/bin/env bash
# NikolAI public page: nikolai.cognitronics.ai -> mini PC port 8092 (read-only view)
# Run on the VPS:  bash nikolai-public.sh
# Does: 1) checks DNS  2) checks the mini PC can be reached (offers to install Tailscale)
#       3) creates the proxy host (or points the existing one to the mini PC)  4) offers to stop the VPS NikolAI
#       – i.e. creates/updates the proxy host in Nginx Proxy Manager (with a Let's Encrypt certificate) via its API.
set -u
DOMAIN="nikolai.cognitronics.ai"
MINIPC="100.103.103.100"
PORT=8092
NPM="http://127.0.0.1:81"

echo "== 1/3 DNS"
IP_VPS=$(curl -s -m 5 https://api.ipify.org || true)
IP_DNS=$(getent hosts "$DOMAIN" | awk '{print $1}' | head -1)
echo "   VPS: ${IP_VPS:-?}   $DOMAIN -> ${IP_DNS:-(not found)}"
if [ -z "$IP_DNS" ] || [ "$IP_DNS" != "$IP_VPS" ]; then
  echo "   !! The DNS entry is missing or not active yet. At Hostinger: type A, name 'nikolai', target ${IP_VPS:-your VPS IP}."
  echo "   !! Without it there will be no certificate. Run this script again in a few minutes."
  DNS_OK=0
else
  echo "   ok"; DNS_OK=1
fi

echo "== 2/3 Can the VPS reach the mini PC?"
if curl -s -m 5 "http://$MINIPC:8091/zustand" >/dev/null; then
  echo "   ok (NikolAI answers on $MINIPC)"
  curl -s -m 5 "http://$MINIPC:$PORT/zustand" >/dev/null && echo "   public port $PORT: ok" \
    || echo "   !! Port $PORT does not answer yet – install Neuro 13 on the mini PC first (the proxy entry is created anyway)."
else
  echo "   !! No answer from $MINIPC."
  if ! command -v tailscale >/dev/null; then
    read -r -p "   Install Tailscale on this VPS now? [y/N] " a
    if [ "${a:-n}" = "y" ]; then
      curl -fsSL https://tailscale.com/install.sh | sh && sudo tailscale up
      echo "   Open the link above and approve the VPS, then run this script again."
    fi
    exit 1
  else
    echo "   Tailscale is installed. Is it logged in? ->  sudo tailscale status"
    echo "   Is NikolAI running on the mini PC? ->  systemctl is-active nikolai-neuro"
    exit 1
  fi
fi

echo "== 3/3 Nginx Proxy Manager ($NPM)"
read -r -p "   NPM login e-mail: " EMAIL
read -r -s -p "   NPM password: " PASS; echo
python3 - "$NPM" "$EMAIL" "$PASS" "$DOMAIN" "$MINIPC" "$PORT" "$DNS_OK" <<'PY'
import json, sys, urllib.request
npm, email, pw, domain, host, port, dns_ok = sys.argv[1:8]
def call(method, path, data=None, token=None):
    req = urllib.request.Request(npm + path, method=method, data=json.dumps(data).encode() if data else None,
                                 headers={"Content-Type": "application/json", **({"Authorization": "Bearer " + token} if token else {})})
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            return json.loads(r.read() or b"{}")
    except urllib.error.HTTPError as e:
        raise SystemExit(f"   !! NPM said {e.code}: {e.read().decode()[:300]}")
    except urllib.error.URLError as e:
        raise SystemExit(f"   !! NPM not reachable at {npm}: {e}")
token = call("POST", "/api/tokens", {"identity": email, "secret": pw})["token"]
for h in call("GET", "/api/nginx/proxy-hosts", token=token):
    if domain in h.get("domain_names", []):          # already there: point it to the mini PC, keep SSL
        call("PUT", f"/api/nginx/proxy-hosts/{h['id']}",
             {"forward_scheme": "http", "forward_host": host, "forward_port": int(port)}, token)
        raise SystemExit(f"   ok – existing proxy host {h['id']} now points to {host}:{port} (SSL unchanged)")
host_cfg = {"domain_names": [domain], "forward_scheme": "http", "forward_host": host, "forward_port": int(port),
            "access_list_id": 0, "certificate_id": 0, "ssl_forced": False, "caching_enabled": False,
            "block_exploits": True, "advanced_config": "", "allow_websocket_upgrade": False,
            "http2_support": False, "hsts_enabled": False, "hsts_subdomains": False, "locations": [], "meta": {}}
if dns_ok == "1":
    host_cfg.update(certificate_id="new", ssl_forced=True, http2_support=True,
                    meta={"letsencrypt_email": email, "letsencrypt_agree": True, "dns_challenge": False})
r = call("POST", "/api/nginx/proxy-hosts", host_cfg, token)
print(f"   ok – proxy host {r.get('id')} created" + (" with HTTPS" if dns_ok == "1" else " (HTTP only – add SSL in NPM once DNS works)"))
PY
echo "== 4/4 The old NikolAI on this VPS"
if systemctl is-active --quiet nikolai 2>/dev/null; then
  read -r -p "   Stop and disable it (the page now shows the mini PC)? Its brain stays in /opt/nikolai-data. [y/N] " a
  [ "${a:-n}" = "y" ] && systemctl disable --now nikolai && echo "   stopped"
fi
echo "== done: https://$DOMAIN"
