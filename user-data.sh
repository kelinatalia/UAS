#!/bin/bash
set -eu

GIT_REPO="https://github.com/kelinatalia/UAS.git"
SUBFOLDER=""
APP_FILE="app_streamlit_cloud.py"
S3_BUCKET="sagemaker-us-east-1-488236761809"

REGION="us-east-1"
APP_DIR="/opt/credit-app"
VENV_DIR="/opt/streamlit-venv"

if [ -z "$SUBFOLDER" ]; then
  APP_PATH="$APP_DIR"
else
  APP_PATH="$APP_DIR/$SUBFOLDER"
fi

dnf update -y
dnf install -y python3 python3-pip git

git clone "$GIT_REPO" "$APP_DIR"
chown -R ec2-user:ec2-user "$APP_DIR"

if [ ! -f "$APP_PATH/$APP_FILE" ]; then
  echo "FATAL: $APP_PATH/$APP_FILE not found."
  find "$APP_DIR" -maxdepth 4 -type f | head -40
  exit 1
fi

python3 -m venv "$VENV_DIR"
"$VENV_DIR/bin/pip" install --upgrade pip
"$VENV_DIR/bin/pip" install -r "$APP_PATH/requirements.txt"

cat >/etc/systemd/system/streamlit.service <<EOF
[Unit]
Description=Streamlit App
After=network.target

[Service]
Type=simple
User=ec2-user
WorkingDirectory=$APP_PATH
Environment=S3_BUCKET=$S3_BUCKET
Environment=AWS_REGION=$REGION
ExecStart=$VENV_DIR/bin/streamlit run $APP_FILE \\
  --server.address 0.0.0.0 \\
  --server.port 8501 \\
  --server.headless true \\
  --server.enableCORS false \\
  --server.enableXsrfProtection false
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable --now streamlit.service

sleep 5
if systemctl is-active --quiet streamlit; then
  touch "$APP_DIR/.userdata-success"
  chown ec2-user:ec2-user "$APP_DIR/.userdata-success"
else
  echo "FATAL: streamlit service failed to start."
  journalctl -u streamlit -n 30 --no-pager || true
  exit 1
fi
