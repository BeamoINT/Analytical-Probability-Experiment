## Secrets + “clone and run” setup (safe)

This repo **does not** commit private keys or API secrets.
Instead it supports:

- **Committed defaults**: `env.live` (safe; no secrets)
- **Machine secrets**: `~/.polyb0t/secrets.env` (NOT in git)
- **Local override link**: `.env.local` → `~/.polyb0t/secrets.env`

### One-time setup per machine

```bash
cd "/path/to/repo"
bash scripts/bootstrap_env.sh
nano ~/.polyb0t/secrets.env
```

### Run in background (log to file)

```bash
cd "/path/to/repo"
nohup "$HOME/.local/bin/poetry" run polyb0t run --live > live_run.log 2>&1 &
```

### Monitor logs

```bash
cd "/path/to/repo"
tail -f live_run.log
```

### Auto-start on boot (systemd)

Create:

```bash
sudo nano /etc/systemd/system/polymarket-bot.service
```

Example (edit `User=` and paths):

```ini
[Unit]
Description=Polymarket Auto Trading Bot
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=YOUR_USER
WorkingDirectory=/path/to/repo
Environment=PATH=/home/YOUR_USER/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
EnvironmentFile=/home/YOUR_USER/.polyb0t/secrets.env
ExecStart=/home/YOUR_USER/.local/bin/poetry run polyb0t run --live
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Enable + start:

```bash
sudo systemctl daemon-reload
sudo systemctl enable polymarket-bot
sudo systemctl start polymarket-bot
```

Logs:

```bash
sudo journalctl -u polymarket-bot -f
```


