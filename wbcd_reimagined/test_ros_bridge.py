import socket

HOST = '0.0.0.0'
PORT = 5001

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
sock.bind((HOST, PORT))
sock.listen(1)

print(f'Listening on {HOST}:{PORT}')
conn, addr = sock.accept()
print('Desktop connected from', addr)

while True:
    s = input('Enter x y z: ').strip()
    if not s:
        continue
    conn.sendall((s + '\n').encode())
    # 0.3 0.25 0.25 180.0 45.0 90.0 0.0
    # 0.3 0.15 0.2 180.0 0.0 90.0 0.0