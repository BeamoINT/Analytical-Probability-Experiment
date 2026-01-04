# 🐧 Ubuntu Quick Start - One-Command Setup

## 🚀 Fastest Way to Get Started on Ubuntu

This guide is for **Ubuntu/Debian** users who want a fully automated setup.

---

## ⚡ One-Command Installation

```bash
# Clone the repo
git clone https://github.com/BeamoINT/Analytical-Probability-Experiment.git
cd Analytical-Probability-Experiment

# Run the auto-setup script
./ubuntu_setup.sh
```

That's it! The script will:
- ✅ Check and install Python 3.11+
- ✅ Install system dependencies (build tools, SQLite, etc.)
- ✅ Install pip
- ✅ Install Poetry
- ✅ Install all PolyB0t dependencies
- ✅ Initialize the database
- ✅ Create `.env` from template

**Time:** ~5-10 minutes (depending on what's already installed)

---

## 📋 What Gets Installed

### System Packages (via apt)
```
build-essential     # C/C++ compiler for native extensions
curl, wget, git     # Download tools
python3.11          # Python runtime
python3.11-venv     # Virtual environment support
python3.11-dev      # Python headers for native modules
libssl-dev          # SSL/TLS support
libffi-dev          # Foreign function interface
sqlite3             # Database engine
libsqlite3-dev      # SQLite headers
```

### Python Tools
```
pip                 # Package manager
poetry              # Dependency manager
```

### PolyB0t Dependencies
```
All packages from pyproject.toml:
  - py-clob-client  # Polymarket API
  - web3            # Ethereum/Polygon
  - lightgbm        # Machine learning
  - scikit-learn    # ML utilities
  - pandas, numpy   # Data processing
  - (and 20+ more)
```

---

## 🔧 After Installation

### 1. Configure Environment

Edit your `.env` file:

```bash
nano .env
```

Required settings:

```bash
# Basic Settings
POLYBOT_MODE=live
POLYBOT_USER_ADDRESS=0xYourWalletAddress
POLYBOT_POLYGON_RPC_URL=https://polygon-rpc.com
POLYBOT_DRY_RUN=true

# Trading Parameters
POLYBOT_LOOP_INTERVAL_SECONDS=10
POLYBOT_MIN_ORDER_USD=1.0
POLYBOT_MAX_ORDER_USD=100.0
POLYBOT_MAX_TOTAL_EXPOSURE_USD=500.0
```

### 2. Test the Installation

```bash
# Verify installation
poetry run polyb0t --help

# Run diagnostics
poetry run polyb0t doctor

# Check status
poetry run polyb0t status
```

### 3. Run the Bot (Dry-Run Mode)

```bash
# Safe mode - no real orders
poetry run polyb0t run --live
```

### 4. (Optional) Enable Live Trading

See `README_L2_SETUP.md` for L2 credential setup.

---

## 🐛 Troubleshooting

### "poetry: command not found"

The script adds Poetry to your PATH, but you need to reload your shell:

```bash
# Option 1: Reload bashrc
source ~/.bashrc

# Option 2: Restart terminal
exit
# (open new terminal)
```

### "Python version too old"

The script installs Python 3.11 from deadsnakes PPA. If it fails:

```bash
# Manually add PPA and install
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update
sudo apt install python3.11 python3.11-venv python3.11-dev

# Tell Poetry to use it
poetry env use python3.11
poetry install
```

### "Permission denied" on ubuntu_setup.sh

Make it executable:

```bash
chmod +x ubuntu_setup.sh
./ubuntu_setup.sh
```

### Dependencies fail to install

Try locking and reinstalling:

```bash
poetry lock --no-update
poetry install
```

### LightGBM installation fails

Install additional build dependencies:

```bash
sudo apt install -y cmake libboost-dev
poetry install
```

---

## 🖥️ Tested Ubuntu Versions

✅ **Ubuntu 22.04 LTS** (Recommended)  
✅ **Ubuntu 20.04 LTS**  
✅ **Ubuntu 24.04 LTS**  
✅ **Debian 11+**  
✅ **Linux Mint 20+**  

---

## 🚀 Cloud Deployment

Perfect for VPS/Cloud servers:

### DigitalOcean Droplet
```bash
# Recommended specs:
- Ubuntu 22.04 LTS
- 2 vCPU
- 8 GB RAM
- 50 GB SSD

# After SSH:
git clone <repo>
cd Analytical-Probability-Experiment
./ubuntu_setup.sh
```

### AWS EC2
```bash
# Recommended instance:
- t3.large (2 vCPU, 8GB RAM)
- Ubuntu 22.04 AMI
- 50 GB EBS volume

# After SSH:
git clone <repo>
cd Analytical-Probability-Experiment
./ubuntu_setup.sh
```

### Google Cloud VM
```bash
# Recommended machine:
- e2-standard-2 (2 vCPU, 8GB RAM)
- Ubuntu 22.04 LTS
- 50 GB disk

# After SSH:
git clone <repo>
cd Analytical-Probability-Experiment
./ubuntu_setup.sh
```

---

## 🔒 Running as a Service (Optional)

Keep the bot running 24/7:

### Create systemd service

```bash
# Create service file
sudo nano /etc/systemd/system/polyb0t.service
```

Add this content:

```ini
[Unit]
Description=PolyB0t Trading Bot
After=network.target

[Service]
Type=simple
User=your_username
WorkingDirectory=/home/your_username/Analytical-Probability-Experiment
Environment="PATH=/home/your_username/.local/bin:/usr/bin"
ExecStart=/home/your_username/.local/bin/poetry run polyb0t run --live
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Enable and start:

```bash
# Reload systemd
sudo systemctl daemon-reload

# Enable on boot
sudo systemctl enable polyb0t

# Start now
sudo systemctl start polyb0t

# Check status
sudo systemctl status polyb0t

# View logs
sudo journalctl -u polyb0t -f
```

---

## 🛡️ Security Best Practices

### Don't run as root
```bash
# Create dedicated user
sudo useradd -m -s /bin/bash polyb0t
sudo su - polyb0t

# Run setup as this user
git clone <repo>
./ubuntu_setup.sh
```

### Secure your .env file
```bash
chmod 600 .env
```

### Use SSH keys (not passwords)
```bash
# On your local machine:
ssh-copy-id user@your-server

# On server, disable password auth:
sudo nano /etc/ssh/sshd_config
# Set: PasswordAuthentication no
sudo systemctl restart ssh
```

### Enable firewall
```bash
sudo ufw allow ssh
sudo ufw enable
```

---

## 📊 Resource Usage

### Idle (waiting for next cycle)
- CPU: 1-2%
- RAM: 200-400 MB
- Disk I/O: Minimal

### Active (processing cycle)
- CPU: 20-40%
- RAM: 400-800 MB
- Disk I/O: Low

### ML Training (background, every 6 hours)
- CPU: 80-100% (for 5-15 minutes)
- RAM: 2-4 GB
- Disk I/O: Moderate

---

## 🎯 Next Steps

After installation:

1. ✅ Configure `.env`
2. ✅ Run `poetry run polyb0t doctor`
3. ✅ Test with `poetry run polyb0t run --live` (dry-run)
4. ✅ Monitor logs and metrics
5. ✅ Read `ML_SYSTEM_GUIDE.md` to enable ML

---

## 📚 Additional Documentation

- `README.md` - Main documentation
- `README_L2_SETUP.md` - L2 credentials for live trading
- `ML_SYSTEM_GUIDE.md` - Machine learning system
- `DATA_RETENTION_UPGRADE.md` - Data retention (15GB capacity)
- `BROAD_MARKET_LEARNING.md` - Broad market data collection

---

## 🆘 Getting Help

If something goes wrong:

1. Check logs: `poetry run polyb0t status`
2. Run diagnostics: `poetry run polyb0t doctor`
3. Check GitHub issues
4. Review error messages carefully

---

**Happy Trading! 🚀**

