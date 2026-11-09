#!/bin/sh
# picarx-netcut を設置し、実行した利用者にだけパスワードなしの sudo を許可する。
#
#   sudo sh install-netcut.sh [picarx-netcut.sh のパス]
#
# sudoers は検査してから置くので、書式の誤りで sudo が使えなくなることはない。
# 取り消し: sudo rm /etc/sudoers.d/picarx-netcut /usr/local/sbin/picarx-netcut
set -eu

SRC=${1:-/tmp/picarx-netcut}
DEST=/usr/local/sbin/picarx-netcut
RULE=/etc/sudoers.d/picarx-netcut

if [ "$(id -u)" -ne 0 ]; then
    echo "sudo で実行してください" >&2
    exit 1
fi
USER_NAME=${SUDO_USER:-}
if [ -z "$USER_NAME" ] || [ "$USER_NAME" = root ]; then
    echo "一般の利用者から sudo で実行してください" >&2
    exit 1
fi

install -o root -g root -m 755 "$SRC" "$DEST"

TMP=$(mktemp)
trap 'rm -f "$TMP"' EXIT
echo "$USER_NAME ALL=(root) NOPASSWD: $DEST" >"$TMP"
visudo -cf "$TMP"
install -o root -g root -m 440 "$TMP" "$RULE"
echo "installed: $DEST (sudo without password for $USER_NAME)"
