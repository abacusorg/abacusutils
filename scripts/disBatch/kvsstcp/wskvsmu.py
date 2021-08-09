#!/usr/bin/env python2
from __future__ import print_function
import argparse, base64, hashlib, json, os, SimpleHTTPServer, socket, SocketServer, struct as S, sys
from threading import current_thread, Lock, Thread
import pkg_resources

try:
    import .kvsclient
    from .kvscommon import recvall
except:
    import kvsclient
    from kvscommon import recvall

# The web monitor implements two conventions wrt keys:
#
# 1) A key starting with '.' is not displayed.
#
# 2) A wait for a key of the form "key?fmt" creates an input
# box for the key. fmt will be used to format the input string
# to produce a "put" value for the key.
#
# It implements three conventions wrt to values:
#
# 1) An ASCII String encoded value that starts with '<HTML>' (case
# insensitive) will be sent directly to the web client after removing
# the '<HTML>' pefix. I.e., this combination permits the value to be
# render as HTML.
#
# 2) An ASCII String encoded value that does not start with '<HTML>'
# is sent HTML escaped and encoded (distinct from the KVS encoding
# scheme). This combination prevents the value from being interpreted
# as HTML.
#
# 3) In all other cases, python's 'repr' of the (KVS encoding, value)
# tuple is sent to the web client HTML escaped and encoded.
#

def jsonDefault(o):
    if isinstance(o, bytearray):
        return o.decode('latin-1')
    return repr(o)

def dump2json(kvsc):
    r = json.dumps(kvsc.dump(), ensure_ascii = False, check_circular = False, encoding = 'latin-1', default = jsonDefault)
    if isinstance(r, unicode): r = r.encode('latin-1')
    return r

class KVSWaitThread(Thread):
    def __init__(self, kvsaddr, wslist, mk, spec, frontend, name='KVSClientThread'):
        Thread.__init__(self, name=name)
        self.daemon = True
        self.mk = mk
        self.wslist = wslist
        self.kvsc = kvsclient.KVSClient(kvsaddr)
        self.kvsc.monkey(mk, spec)
        self.frontend = frontend
        self.start()

    def run(self):
        try: 
            while 1:
                r = self.kvsc.get(self.mk, False)
                j = dump2json(self.kvsc)
                self.wslist.broadcast(j)
        finally:
            self.wslist.broadcast('bye', True)
            self.kvsc.close()
            self.frontend.stop()

class WebSocketServer(object):
    def __init__(self, kvsaddr, wslist, ws):
        self.client = ws
        self.lock = Lock()
        self.active = True

        kvsc = kvsclient.KVSClient(kvsaddr)
        wslist.add(self)

        try:
            while self.active:
                op = self.recv()
                if not op: pass
                elif op == 'bye': break
                elif op == 'dump':
                    j = dump2json(kvsc)
                    self.send(j)
                elif op.startswith('put\x00'):
                    d, key, v = op.split('\x00')
                    if '?' in key:
                        k, fmt = key.split('?')
                        if fmt: v = fmt%v
                    kvsc.put(key, v, False)
                else: raise Exception('Unknown op from websocket: "%s".'%repr(op))
        finally:
            wslist.remove(self)
            kvsc.close()
            self.close()

    def send(self, p, end = False):
        try:
            with self.lock:
                op = 8 if end else 1
                lp = len(p)
                if lp < 126:
                    self.client.sendall(S.pack('BB', 0x80 | op, lp))
                elif lp < 2**16:
                    self.client.sendall(S.pack('!BBH', 0x80 | op, 126, lp))
                else:
                    self.client.sendall(S.pack('!BBQ', 0x80 | op, 127, lp))
                self.client.sendall(p)
        except socket.error as msg:
            self.active = False

    def recv(self):
        r = self.client.recv(1)
        if not r: return 'bye'
        b = S.unpack('B', r)[0]
        op = b & 0xF
        if op == 0x8:
            assert b & 0x80
            return 'bye'
        elif op == 0x9:
            print('Got a ping', file=sys.stderr)
            #TODO: Do something here
            return ''
        elif op == 0XA:
            print('Got a pong', file=sys.stderr)
            #TODO: Do something here?
            return ''
        elif op == 0x1:
            assert b & 0x80 # For this test, this frame must be marked FIN
            b = S.unpack('B', recvall(self.client, 1))[0]
            assert b & 0x80 # Masking must be on for frames from clients.
            plen = b & 0x7F
            assert plen < 126 # Limit on payload for this test.
            mb = S.unpack('4B', recvall(self.client, 4))
            p = recvall(self.client, plen)
            decode = ''
            for i in xrange(plen): decode += chr(ord(p[i]) ^ mb[i%4])
            return decode
        else:
            print('ws recv header byte: %02x'%b, file=sys.stderr)
            return ''

    def close(self):
        self.client.close()

class WebSocketList(object):
    def __init__(self):
        self.lock = Lock()
        self.wss = set() # of WebWaitThread
    
    def add(self, ws):
        with self.lock:
            self.wss.add(ws)

    def remove(self, ws):
        with self.lock:
            self.wss.remove(ws)

    def __iter__(self):
        with self.lock:
            return iter(self.wss.copy())

    def broadcast(self, p, end = False):
        for ws in self:
            # TODO: add timeout for sending to avoid blocking, or lift to separate thread
            ws.send(p, end)

class FrontEnd(SimpleHTTPServer.SimpleHTTPRequestHandler):
    html = pkg_resources.resource_filename('kvsstcp', 'wskvspage.html')

    wsmagic = '258EAFA5-E914-47DA-95CA-C5AB0DC85B11'

    handshake = '\
HTTP/1.1 101 Switching Protocols\r\n\
Upgrade: websocket\r\n\
Connection: Upgrade\r\n\
Sec-WebSocket-Accept: %s\r\n\r\n\
'

    def send_head(self):
        if self.headers.get('upgrade', None) == 'websocket' and self.headers.get('sec-websocket-protocol', None) == 'kvs' and self.headers.get('origin') == self.server.base_url:
            k = self.headers.get('sec-websocket-key')
            reply = FrontEnd.handshake%(base64.b64encode(hashlib.sha1(k + FrontEnd.wsmagic).digest()))
            self.request.sendall(reply)
            WebSocketServer(self.server.kvsserver, self.server.wslist, self.request)
            return None

        elif self.path == '/kvsviewer':
            f = open(FrontEnd.html, 'r')
            self.send_response(200)
            self.send_header('Content-type', 'text/html; charset=utf-8')
            fs = os.fstat(f.fileno())
            self.send_header("content-length", str(fs.st_size))
            self.send_header("last-modified", self.date_time_string(fs.st_mtime))
            self.end_headers()
            return f
            
        else:
            self.send_error(404, 'Not recognized')
            return None

class ThreadingTCPServer(SocketServer.ThreadingTCPServer):
    daemon_threads = True

class FrontEndThread(Thread):
    def __init__(self, kvsserver, addr, urlfile, wslist, name='FrontEndThread'):
        Thread.__init__(self, name=name)
        self.daemon = True
        self.httpd = ThreadingTCPServer(addr, FrontEnd)
        if addr[1]: self.httpd.allow_reuse_address = True
        self.httpd.base_url = 'http://%s:%s'%(self.httpd.server_address)

        myurl = self.httpd.base_url + '/kvsviewer'
        print('front end at: '+myurl, file=sys.stderr)
        if urlfile:
            urlfile.write('%s\n'%(myurl))
            urlfile.close()

        self.httpd.kvsserver = kvsserver
        self.httpd.wslist = wslist
        self.start()

    def run(self):
        self.httpd.serve_forever()
        self.httpd.server_close()

    def stop(self):
        self.httpd.shutdown()

def main(kvsserver, urlfile=None, monitorkey='.webmonitor', monitorspec=':w', addr=(socket.gethostname(), 0), **args):
    wslist = WebSocketList()
    fe = FrontEndThread(kvsserver, addr, urlfile, wslist)
    kvs = KVSWaitThread(kvsserver, wslist, monitorkey, monitorspec, fe)
    return (fe, kvs)

if '__main__' == __name__:
    argp = argparse.ArgumentParser(description='Start a web monitor for a key-value storage server.')
    argp.add_argument('-m', '--monitorkey', default='.webmonitor', help='Key to use for the monitor.')
    argp.add_argument('-s', '--monitorspec', default=':w', help='What to monitor: comma separted list of "[key]:[gpvw]" specifications.')
    argp.add_argument('-u', '--urlfile', default=None, type=argparse.FileType('w'), help='Write url to this file.')
    argp.add_argument('-b', '--bind', default=socket.gethostname(), type=str, help='HTTP IP to listen on (hostname).')
    argp.add_argument('-p', '--port', default=0, type=int, help='HTTP port to listen on (random).')
    kvsclient.addKVSServerArgument(argp)
    args = argp.parse_args()

    args.addr = (args.bind, args.port)
    (fe, kvs) = main(**args.__dict__)
    try:
        while fe.isAlive():
            fe.join(60)
    finally:
        fe.stop()
    fe.join()
