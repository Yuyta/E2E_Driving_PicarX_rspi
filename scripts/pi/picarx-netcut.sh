#!/bin/sh
# 走行サーバー（TCP 8080）への受信だけを一定時間破棄し、時間が来たら元に戻す。
# PC の Wi-Fi を切らずに、Pi から見た「指令が途絶えた」状態を再現する（M1 の Wi-Fi 切断の試験）。
#
#   sudo picarx-netcut <秒 1〜30>   遮断して、秒数の経過後に戻す
#   sudo picarx-netcut clear        遮断を即座に解く
#
# 設置: /usr/local/sbin/picarx-netcut（root 所有、755）。sudoers でこのファイルだけ
# パスワードなしを許可する（ROADMAP の M1 の手順4）。規則は永続化しないので、再起動でも消える。
set -u

TABLE=picarx_netcut
NFT=/usr/sbin/nft

restore() {
    "$NFT" delete table inet "$TABLE" 2>/dev/null
    return 0
}

case "${1:-}" in
    clear)
        restore
        echo "restored"
        exit 0
        ;;
    '' | *[!0-9]*)
        echo "usage: picarx-netcut <seconds 1-30> | clear" >&2
        exit 2
        ;;
esac

secs=$1
if [ "$secs" -lt 1 ] || [ "$secs" -gt 30 ]; then
    echo "seconds must be 1-30" >&2
    exit 2
fi

restore
# 時間切れでも、SSH の切断や中断でも、必ず元に戻す
trap restore EXIT
trap 'exit 1' HUP INT TERM

"$NFT" -f - <<EOF || exit 1
table inet $TABLE {
    chain input {
        type filter hook input priority -10; policy accept;
        tcp dport 8080 drop
    }
}
EOF

echo "cut tcp/8080 for ${secs}s"
sleep "$secs"
echo "restored"
