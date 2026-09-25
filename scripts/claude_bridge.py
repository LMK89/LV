"""claude_bridge.py — Script kết nối tự động giữa Antigravity và Claude Code qua Edge CDP (Port 9222).

Sử dụng:
    python scripts/claude_bridge.py status
    python scripts/claude_bridge.py send "Nội dung cần gửi cho Claude"
"""
import argparse
import base64
import json
import os
import socket
import struct
import sys
import time
import urllib.parse
import urllib.request

if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass


class ClaudeCDPClient:
    def __init__(self, port: int = 9222):
        self.port = port
        self._msg_id = 0
        self.sock = None
        self.connect()

    def connect(self):
        try:
            tabs = json.loads(urllib.request.urlopen(f"http://127.0.0.1:{self.port}/json", timeout=5).read().decode())
        except Exception as e:
            raise ConnectionError(
                f"Không kết nối được cổng {self.port} trên Edge. Hãy đảm bảo Edge đã chạy với:\n"
                f'msedge.exe --remote-debugging-port=9222 --user-data-dir="C:\\edge_debug_profile"\n'
                f"Lỗi: {e}"
            )

        claude = next((t for t in tabs if "Claude" in t.get("title", "") or "claude.ai/code" in t.get("url", "")), None)
        if not claude:
            available = [t.get("title") for t in tabs if t.get("title")]
            raise RuntimeError(f"Không tìm thấy tab Claude Code trên Edge. Các tab đang mở: {available}")

        self.tab_info = claude
        parsed = urllib.parse.urlparse(claude["webSocketDebuggerUrl"])
        self.sock = socket.create_connection((parsed.hostname, parsed.port), timeout=15)
        key = base64.b64encode(os.urandom(16)).decode("ascii")
        headers = [
            f"GET {parsed.path} HTTP/1.1",
            f"Host: {parsed.hostname}:{parsed.port}",
            "Upgrade: websocket",
            "Connection: Upgrade",
            f"Sec-WebSocket-Key: {key}",
            "Sec-WebSocket-Version: 13",
            "\r\n",
        ]
        self.sock.sendall("\r\n".join(headers).encode("ascii"))
        resp = b""
        while b"\r\n\r\n" not in resp:
            resp += self.sock.recv(1024)
        assert b"101 " in resp, f"WebSocket Handshake failed: {resp}"

    def send_cmd(self, method: str, params: dict = None) -> dict:
        self._msg_id += 1
        msg = {"id": self._msg_id, "method": method, "params": params or {}}
        data = json.dumps(msg).encode("utf-8")
        length = len(data)
        mask = os.urandom(4)
        masked = bytearray(b ^ mask[i % 4] for i, b in enumerate(data))
        header = bytearray([0x81])
        if length <= 125:
            header.append(0x80 | length)
        elif length <= 65535:
            header.append(0x80 | 126)
            header.extend(struct.pack("!H", length))
        else:
            header.append(0x80 | 127)
            header.extend(struct.pack("!Q", length))
        header.extend(mask)
        self.sock.sendall(header + masked)
        return self._recv()

    def _recv(self) -> dict:
        head = self.sock.recv(2)
        if len(head) < 2:
            return {}
        length = head[1] & 0x7F
        if length == 126:
            length = struct.unpack("!H", self.sock.recv(2))[0]
        elif length == 127:
            length = struct.unpack("!Q", self.sock.recv(8))[0]
        data = b""
        while len(data) < length:
            chunk = self.sock.recv(length - len(data))
            if not chunk:
                break
            data += chunk
        return json.loads(data.decode("utf-8", errors="replace"))

    def evaluate(self, expr: str):
        res = self.send_cmd("Runtime.evaluate", {"expression": expr, "returnByValue": True})
        return res.get("result", {}).get("result", {}).get("value")

    def status(self) -> dict:
        js = """
        (() => {
            const stopBtn = document.querySelector('button[aria-label="Stop generating"], button[aria-label="Stop Response"], button[aria-label="Stop"]');
            const editor = document.querySelector('div[aria-label="Prompt"]');
            const sendBtn = document.querySelector('button[aria-label="Send"]');
            return {
                url: window.location.href,
                title: document.title,
                isGenerating: !!stopBtn,
                hasEditor: !!editor,
                sendDisabled: sendBtn ? sendBtn.disabled : true
            };
        })()
        """
        return self.evaluate(js)

    def send_message(self, text: str) -> dict:
        js_insert = f"""
        (() => {{
            const editor = document.querySelector('div[aria-label="Prompt"], div[contenteditable="true"], textarea');
            if (!editor) return {{ status: 'ERROR', message: 'Không tìm thấy ô nhập Prompt' }};
            editor.focus();
            document.execCommand('selectAll', false, null);
            document.execCommand('delete', false, null);
            const text = {json.dumps(text)};
            document.execCommand('insertText', false, text);
            editor.dispatchEvent(new Event('input', {{ bubbles: true, composed: true }}));
            editor.dispatchEvent(new Event('change', {{ bubbles: true, composed: true }}));

            // Tìm nút send theo nhiều tiêu chí
            const sendSelectors = [
                'button[aria-label="Send"]',
                'button[aria-label="Send Message"]',
                'button[aria-label="Send Prompt"]',
                'button[aria-label="Send prompt"]',
                'button[aria-label="Send message"]'
            ];
            let sendBtn = null;
            for (const sel of sendSelectors) {{
                const b = document.querySelector(sel);
                if (b) {{ sendBtn = b; break; }}
            }}
            if (!sendBtn) {{
                // fallback: tìm button chứa SVG mũi tên hoặc có text gửi
                const allButtons = Array.from(document.querySelectorAll('button'));
                sendBtn = allButtons.find(b => {{
                    const label = (b.getAttribute('aria-label') || '').toLowerCase();
                    return label.includes('send') || (b.innerText || '').toLowerCase().includes('send');
                }});
            }}

            if (sendBtn) {{
                if (!sendBtn.disabled) {{
                    sendBtn.click();
                    return {{ status: 'SENT' }};
                }} else {{
                    return {{ status: 'TYPED_BUT_DISABLED', sendBtnAria: sendBtn.getAttribute('aria-label') }};
                }}
            }}

            return {{ status: 'TYPED_NO_BUTTON' }};
        }})()
        """
        return self.evaluate(js_insert)

    def close(self):
        if self.sock:
            try:
                self.sock.close()
            except Exception:
                pass


def main():
    parser = argparse.ArgumentParser(description="Bridge giữa Antigravity và Claude Code qua Edge Port 9222")
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("status", help="Kiểm tra trạng thái tab Claude Code")

    send_parser = subparsers.add_parser("send", help="Gửi tin nhắn sang Claude Code")
    send_parser.add_argument("message", nargs="?", default="", type=str, help="Nội dung cần gửi")
    send_parser.add_argument("--file", "-f", type=str, default=None, help="Đọc nội dung từ file text")

    args = parser.parse_args()
    if not args.command or args.command == "status":
        try:
            client = ClaudeCDPClient()
            info = client.status()
            print("[OK] Kết nối Edge 9222 thành công!")
            print(json.dumps(info, ensure_ascii=False, indent=2))
            client.close()
        except Exception as e:
            print(f"[ERROR] {e}")
    elif args.command == "send":
        try:
            msg = args.message
            if args.file and os.path.exists(args.file):
                with open(args.file, "r", encoding="utf-8") as f:
                    msg = f.read()
            if not msg.strip():
                print("[ERROR] Tin nhắn trống, không thể gửi.")
                sys.exit(1)
            client = ClaudeCDPClient()
            res = client.send_message(msg)
            print(f"Kết quả gửi: {res}")
            client.close()
        except Exception as e:
            print(f"[ERROR] {e}")


if __name__ == "__main__":
    main()
